import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omarchy_cluster import agent, cli, facts  # noqa: E402

GIB = 1 << 30


def _nodes(**names):
    return {n: {"gpu_backend": "vulkan", **extra} for n, extra in names.items()}


def test_attach_uses_a_cap_the_driver_refused_at(monkeypatch, capsys):
    posted = []

    def fake_post(node, path, payload, timeout=30):
        posted.append((path, payload))
        return {"alloc_cap_bytes": 8 * GIB, "stopped_at_limit": False}

    monkeypatch.setattr(cli, "_agent_post", fake_post)
    nodes = _nodes(m1={})
    cli._attach_gpu_caps({"m1": {"ip": "x"}}, nodes)
    assert nodes["m1"]["gpu_alloc_cap_bytes"] == 8 * GIB
    assert posted == [("/v1/gpu-cap", {"headroom_gib": 2})]
    assert "asking its agent" in capsys.readouterr().out


def test_attach_ignores_a_floor_and_says_so(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_agent_post", lambda *a, **k: {"alloc_cap_bytes": 10 * GIB, "stopped_at_limit": True})
    nodes = _nodes(m1={})
    cli._attach_gpu_caps({"m1": {"ip": "x"}}, nodes)
    assert "gpu_alloc_cap_bytes" not in nodes["m1"]
    assert "floor" in capsys.readouterr().out


def test_attach_reports_why_a_node_has_no_cap(monkeypatch, capsys):
    answers = iter([{"alloc_cap_bytes": None, "error": "vkCreateDevice failed"},
                    {"alloc_cap_bytes": None, "skipped": "a rank is running on this node"},
                    OSError("timed out")])

    def fake_post(*a, **k):
        r = next(answers)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(cli, "_agent_post", fake_post)
    nodes = _nodes(a={}, b={}, c={})
    cli._attach_gpu_caps({n: {"ip": n} for n in nodes}, nodes)
    out = capsys.readouterr().out
    assert "vkCreateDevice failed" in out and "a rank is running" in out and "timed out" in out
    assert not any("gpu_alloc_cap_bytes" in v for v in nodes.values())


def test_attach_skips_nodes_without_a_vulkan_gpu(monkeypatch):
    monkeypatch.setattr(cli, "_agent_post", lambda *a, **k: (_ for _ in ()).throw(AssertionError("probed")))
    cli._attach_gpu_caps({"mac": {"ip": "x"}}, {"mac": {"gpu_backend": "metal"}})


def test_cached_gpu_cap_ignores_a_floor_and_another_boot(tmp_path, monkeypatch):
    path = tmp_path / "gpu-cap.json"
    monkeypatch.setattr(facts, "GPU_CAP_FILE", str(path))
    monkeypatch.setattr(facts, "boot_id", lambda: "boot-1")
    path.write_text(json.dumps({"boot_id": "boot-1", "alloc_cap_bytes": 8 * GIB, "stopped_at_limit": False}))
    assert facts.cached_gpu_cap() == 8 * GIB
    path.write_text(json.dumps({"boot_id": "boot-1", "alloc_cap_bytes": 8 * GIB, "stopped_at_limit": True}))
    assert facts.cached_gpu_cap() is None
    path.write_text(json.dumps({"boot_id": "boot-0", "alloc_cap_bytes": 8 * GIB, "stopped_at_limit": False}))
    assert facts.cached_gpu_cap() is None
    path.unlink()
    assert facts.cached_gpu_cap() is None


class _Probe:
    def __init__(self, result):
        self.result = result
        self.runs = []

    def __call__(self, cmd, **kw):
        self.runs.append(cmd)

        class P:
            returncode = 0
            stdout = json.dumps(self.result)
            stderr = ""
        return P()


def _agent_env(tmp_path, monkeypatch, free_gib, probe, rank_running=False):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(agent.sys, "platform", "linux")
    monkeypatch.setattr(agent.facts_mod, "boot_id", lambda: "boot-1")
    monkeypatch.setattr(agent.facts_mod, "memory_free_bytes", lambda: free_gib * GIB)
    monkeypatch.setattr(agent.facts_mod, "memory_total_bytes", lambda: 16 * GIB)
    monkeypatch.setattr(agent, "_gpu_holder_running", lambda: rank_running)
    monkeypatch.setattr(agent.subprocess, "run", probe)


def test_agent_probe_leaves_the_requested_headroom_and_reprobes_a_floor(tmp_path, monkeypatch):
    # 13 GiB free: the default 6 GiB headroom allows 7 blocks, a request for 2 GiB allows 11
    probe = _Probe({"device": "gpu", "heap_bytes": 16 * GIB, "alloc_cap_bytes": 7 * GIB, "stopped_at_limit": True})
    _agent_env(tmp_path, monkeypatch, 13, probe)
    first = agent.gpu_cap({})
    assert probe.runs[-1][-1] == "7" and first["stopped_at_limit"] and first["max_gib"] == 7
    agent.gpu_cap({})                      # same limit: the cached floor answers
    assert len(probe.runs) == 1
    probe.result = {"device": "gpu", "heap_bytes": 16 * GIB, "alloc_cap_bytes": 8 * GIB, "stopped_at_limit": False}
    second = agent.gpu_cap({"headroom_gib": 2})
    assert probe.runs[-1][-1] == "11" and second["alloc_cap_bytes"] == 8 * GIB
    agent.gpu_cap({"headroom_gib": 2})     # a measured cap is cached for the boot
    assert len(probe.runs) == 2


def test_agent_does_not_probe_while_a_gpu_holder_runs(tmp_path, monkeypatch):
    probe = _Probe({"alloc_cap_bytes": 8 * GIB, "stopped_at_limit": False})
    _agent_env(tmp_path, monkeypatch, 13, probe, rank_running=True)
    res = agent.gpu_cap({})
    assert res["alloc_cap_bytes"] is None and "GPU-holding server" in res["skipped"]
    assert probe.runs == []


def test_gpu_holder_matches_the_program_not_a_substring():
    holders = [b"/usr/bin/python3\0-m\0omarchy_cluster.rank\0--model\0m", b"ggml-rpc-server\0-p\050060",
               b"/opt/llama/bin/llama-server\0-m\0x.gguf", b"/usr/local/bin/rpc-server\0-H\00.0.0.0",
               b"/usr/bin/sushi-serve\0--port\08000", b"python3\0-u\0/srv/sushi-serve.py", b"python3\0-m\0sushi.serve"]
    others = [b"python\0-m\0omarchy_cluster.agent", b"nvim\0/srv/src/sushi/README.md",
              b"tail\0-f\0/var/log/llama-server.log", b"less\0sushi-serve.log", b"bash\0run-rpc-server-test.sh", b"",
              b"grep\0-r\0omarchy_cluster.rank\0/srv/src"]
    assert all(agent._is_gpu_holder(c) for c in holders)
    assert not any(agent._is_gpu_holder(c) for c in others)


def test_gpu_holder_scan_reads_proc(tmp_path, monkeypatch):
    procs = tmp_path / "proc"
    for pid, cmdline in {"11": b"nvim\0sushi/README.md", "12": b"ggml-rpc-server\0-p\050060"}.items():
        (procs / pid).mkdir(parents=True)
        (procs / pid / "cmdline").write_bytes(cmdline)
    real_listdir, real_open = os.listdir, open
    monkeypatch.setattr(agent.os, "listdir", lambda p: real_listdir(procs if p == "/proc" else p))
    monkeypatch.setattr("builtins.open", lambda p, *a, **k: real_open(
        str(p).replace("/proc/", str(procs) + "/"), *a, **k))
    assert agent._gpu_holder_running()
    (procs / "12" / "cmdline").write_bytes(b"idle")
    assert not agent._gpu_holder_running()


def test_rank_layer_cap_is_the_capped_figure_not_raw_free_memory(monkeypatch):
    """The 13-inch M1: 12 GB free, GPU cap 7.5 GB. The rank's measured split must size its
    layer cap from the cap, or the cap never reaches the engine."""
    from omarchy_cluster import planner, rank
    info = {"ref": "m", "path": "/x", "layers": 27, "hidden": 2048, "kv_heads": 16,
            "head_dim": 128, "weights_bytes": int(8.8e9)}
    monkeypatch.setattr(facts, "memory_free_bytes", lambda: int(12e9))
    monkeypatch.setattr(facts, "cached_gpu_cap", lambda: int(7.5e9))
    capped = rank._layer_cap(info)
    monkeypatch.setattr(facts, "cached_gpu_cap", lambda: None)
    raw = rank._layer_cap(info)
    assert raw == planner.max_layers(int(12e9), info)
    assert capped == planner.max_layers(int(7.5e9), info) and capped < raw


def _stub_measured_split(monkeypatch, info, cap_bytes):
    """Run rank._measured_split for one rank with mlx, the child process and the model
    metadata stubbed, and the node's free memory and measured GPU cap set."""
    import types
    from omarchy_cluster import planner, rank

    class Rows(list):
        def tolist(self):
            return list(self)

    mx = types.ModuleType("mlx.core")
    mx.float32 = None
    mx.cpu = None
    mx.array = lambda rows, dtype=None: Rows(rows)
    mx.distributed = types.SimpleNamespace(all_gather=lambda a, group=None, stream=None: a)
    pkg = types.ModuleType("mlx")
    pkg.core = mx
    monkeypatch.setitem(sys.modules, "mlx", pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setattr(planner, "model_info", lambda ref: info)
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout="5.0\n"))
    monkeypatch.setattr(rank, "_fastest_layer_ms", lambda key, ms: ms)
    monkeypatch.setattr(facts, "memory_free_bytes", lambda: int(12e9))
    monkeypatch.setattr(facts, "cached_gpu_cap", lambda: cap_bytes)
    return rank._measured_split("m", types.SimpleNamespace(rank=lambda: 0))


def test_measured_split_refuses_a_model_the_gpu_cap_cannot_hold(monkeypatch):
    import pytest
    info = {"ref": "m", "path": "/x", "layers": 27, "hidden": 2048, "kv_heads": 16,
            "head_dim": 128, "weights_bytes": int(8.8e9)}
    split, _ = _stub_measured_split(monkeypatch, info, None)  # raw free memory: all 27 layers fit
    assert split == [27]
    with pytest.raises(RuntimeError, match="cannot hold 27 layers"):
        _stub_measured_split(monkeypatch, info, int(7.5e9))  # GPU cap 7.5 GB: they do not
