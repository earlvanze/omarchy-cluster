# One machine, DeepSeek-Coder-V2-Lite 4-bit, current omarchy-mlx build

2026-10-11. The speed table in the main README had one single-machine laptop number, 2.45 tok/s on the
M1 Max, from an earlier omarchy-mlx build. This receipt adds a run on the current build and records what
happens on the 13-inch M1.

## Setup

- Model: `mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit`, 27 layers, 8.8 GB of weights.
- Machines: MacBook Pro 16-inch M1 Max (64 GB) and MacBook Pro 13-inch M1 (16 GB), both Omarchy Linux,
  Vulkan through omarchy-mlx.
- Software: the cluster rank venv (a separate venv from the laptops' main one), omarchy-mlx wheel
  `0.32.4.dev202610102049+93131d1b`, mlx-lm `0.31.4.dev132`; the main venv on each laptop held
  `0.32.4.dev202610111010+4ba8fff` that day. omarchy-cluster is the tree at `3c82f0a` or a later
  docs-only commit: the run log records a SHA-256 over the package's `.py` files (sorted by path)
  of `bd6e3b22828d3db8`, and the same hash computed on a checkout of `3c82f0a` matches (the hash
  covers only the package `.py` files). The same wheel and
  the same weights (two safetensors files of 5316545548 and 3523543836 bytes) are on both machines.
- Run: `omarchy-cluster serve MODEL --nodes NODE --stages 1`, one rank, gateway on the same machine.
  One warm-up request of 8 tokens, then three requests, temperature 0, `max_tokens` 64, prompt "Write a
  haiku about shared memory." Raw responses: [data](data).
- The run was a guard ticket: no other GPU job ran on the machine. The one-minute load average at the
  start was 1.01 on both laptops; the cause of that load was not determined.

## M1 Max alone

| request | completion tokens | decode tok/s | prefill ms | decode ms |
|---|---|---|---|---|
| 1 | 22 | 15.73 | 639.2 | 1401.8 |
| 2 | 22 | 15.70 | 650.4 | 1404.5 |
| 3 | 22 | 15.74 | 639.0 | 1401.3 |

Median 15.73 tok/s. The speed column is the `tokens_per_s` field the gateway returns, which is mlx-lm's
own `generation_tps`. It is not completion tokens divided by decode ms (22 / 1.4018 s is 15.69).

All three requests returned the same text (SHA-256 prefix `b011eaef717fb898`): "Threads in harmony, /
Shared memory binds us, a dance / Of data, diverse and free." The model stops after 22 tokens, so this is
the decode speed over a 22-token reply. The 2.45 tok/s row in the main README came from an earlier
omarchy-mlx build and this receipt does not record that run's reply length, so the two numbers show the
build change only if the replies had the same length.

## 13-inch M1 alone

On the run at 09:49Z the rank died while the engine booted, and the gateway answered 502. The rank log
(not in this directory; kept with the run records) ends with this line:

```
RuntimeError: engine boot failed: RuntimeError: [omarchy] vkAllocateMemory failed: VK_ERROR_OUT_OF_DEVICE_MEMORY (request=92274688 bytes; cache released)
```

A second run at 09:58Z also failed: its warm-up request got connection refused. Its rank log was not
kept, so that run does not show the cause. On the one run that recorded it, the 8.8 GB model failed to
load on the 13-inch M1. `omarchy-cluster place` reports that all 27 layers fit, and it is wrong for this
machine.

A later run (17:40Z) loaded the weights in-process in chunks and recorded mlx's memory counters
([data](data/m1-13inch-weight-load.txt)). The load failed with the same `vkAllocateMemory` error when mlx had
8.07 GB allocated (8.84 GB of parameters in total), about half of the 16.2 GB Vulkan heap the driver reports.
At that moment 5.09 GB of RAM was still available, and mlx's peak equalled its active memory, so the failure
is neither the machine running out of RAM nor a transient double copy while loading. A separate probe that
allocates 1 GiB Vulkan blocks went to 10 GiB without a refusal, under the agent's default driver and under the
driver the ranks use, so that probe does not see this limit. What sets the roughly 8 GB limit is not known.

## What this does not show

On the current build (omarchy-mlx wheel `93131d1b`, omarchy-cluster `3c82f0a`, 2026-10-11) no split of
this model across the two laptops has completed a request. An earlier run did: on 2026-10-06, with
omarchy-cluster at `c95baa0` and an earlier omarchy-mlx build, a 1 + 26 split over the M1 16 GB and the
M1 Max completed requests at 1.79 and 1.80 tok/s and returned the same text as the M1 Max alone. That
run is documented only in the main README's speed table and in the commit message that added its row
(`cdded18`); this repository has no receipt directory for it.

On the 2026-10-11 runs, a 13 + 14 split loaded its 13 layers on the 13-inch M1 and brought the engine up
(seen in the live rank log at 10:06Z; that log was not saved), and the ticket reached its time cap before
a request was sent. The planner's own split failed on the M1 Max side with a lost ring peer, and the
13-inch M1's rank log for that run was overwritten by the next run, so the cause is not recorded. A paired
run that keeps the rank logs is queued.
