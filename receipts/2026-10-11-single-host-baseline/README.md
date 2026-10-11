# One machine, DeepSeek-Coder-V2-Lite 4-bit, current omarchy-mlx build

2026-10-11. The speed table in the main README had one single-machine laptop number, 2.45 tok/s on the
M1 Max, from an earlier omarchy-mlx build. This receipt adds a run on the current build and records what
happens on the 13-inch M1.

## Setup

- Model: `mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit`, 27 layers, 8.8 GB of weights.
- Machines: MacBook Pro 16-inch M1 Max (64 GB) and MacBook Pro 13-inch M1 (16 GB), both Omarchy Linux,
  Vulkan through omarchy-mlx.
- Software: omarchy-mlx wheel `0.32.4.dev202610102049+93131d1b`, mlx-lm `0.31.4.dev132`, omarchy-cluster
  at `3c82f0a`. The same wheel and the same weights (two safetensors files of 5316545548 and 3523543836
  bytes) are on both machines.
- Run: `omarchy-cluster serve MODEL --nodes NODE --stages 1`, one rank, gateway on the same machine.
  One warm-up request of 8 tokens, then three requests, temperature 0, `max_tokens` 64, prompt "Write a
  haiku about shared memory." Raw responses: [data](data).
- The run was a guard ticket: no other GPU job ran on the machine.

## M1 Max alone

| request | completion tokens | decode tok/s | prefill ms | decode ms |
|---|---|---|---|---|
| 1 | 22 | 15.73 | 639.2 | 1401.8 |
| 2 | 22 | 15.70 | 650.4 | 1404.5 |
| 3 | 22 | 15.74 | 639.0 | 1401.3 |

Median 15.73 tok/s. All three requests returned the same text (SHA-256 prefix `b011eaef717fb898`):
"Threads in harmony, / Shared memory binds us, a dance / Of data, diverse and free." The model stops
after 22 tokens, so this is the decode speed over a 22-token reply. The 2.45 tok/s row in the main README
came from an earlier omarchy-mlx build and this receipt does not record that run's reply length, so the
two numbers show the build change only if the replies had the same length.

## 13-inch M1 alone

The rank dies while the engine boots:

```
RuntimeError: engine boot failed: RuntimeError: [omarchy] vkAllocateMemory failed:
VK_ERROR_OUT_OF_DEVICE_MEMORY (request=92274688 bytes; cache released)
```

The gateway then answers 502. This happened on two runs, 09:49Z and 09:58Z. The 8.8 GB model does not fit
on that GPU. `omarchy-cluster place` reports that all 27 layers fit, because it checks free RAM and not
the GPU's allocation limit. That check is wrong for this machine and is open.

## What this does not show

No split of this model across the two laptops has completed a request. On a 13 + 14 split the 13-inch M1
loaded its 13 layers and the engine came up; the ticket reached its time cap before a request was sent.
The planner's own split failed with a lost ring peer, and the rank log that would say why was overwritten
by the next run. A paired run that keeps the rank logs is queued.
