# KV reservation for a hybrid model: planner before and after (2026-10-15)

Input: the `config.json` and the safetensors file sizes of Qwen3.8-27B (`model_type` qwen3_5): 64 decoder layers, `layer_types` names 16 as
`full_attention` and 48 as `linear_attention`, 4 KV heads, head dim 256, bf16 weights 51.75 GiB. No node, no network, no GPU: the planner functions
read only the config and file sizes, so these numbers are exact for this input and say nothing about speed. `measure.py` is the script.

Before (main at 08e4f97): the planner reserved KV for every layer.

```
layers 64 kv_layers (absent: all layers) kv_heads 4 head_dim 256
kv_bytes_per_token 262144 = 256.0 KiB
KV reserved at ctx   2048:    0.500 GiB
KV reserved at ctx  32768:    8.000 GiB
KV reserved at ctx 131072:   32.000 GiB
weights 51.75 GiB
max_layers at free 24 GiB, ctx 2048: 26 of 64; ctx 32768: 23
max_layers at free 40 GiB, ctx 2048: 44 of 64; ctx 32768: 38
max_layers at free 56 GiB, ctx 2048: 61 of 64; ctx 32768: 53
```

After (this change): KV reserved for the 16 attention layers only.

```
layers 64 kv_layers 16 kv_heads 4 head_dim 256
kv_bytes_per_token 65536 = 64.0 KiB
KV reserved at ctx   2048:    0.125 GiB
KV reserved at ctx  32768:    2.000 GiB
KV reserved at ctx 131072:    8.000 GiB
weights 51.75 GiB
max_layers at free 24 GiB, ctx 2048: 26 of 64; ctx 32768: 25
max_layers at free 40 GiB, ctx 2048: 44 of 64; ctx 32768: 42
max_layers at free 56 GiB, ctx 2048: 62 of 64; ctx 32768: 60
```

| | before | after |
|---|---|---|
| KV per token | 256.0 KiB | 64.0 KiB |
| KV reserved at 2048 tokens | 0.500 GiB | 0.125 GiB |
| KV reserved at 32768 tokens | 8.000 GiB | 2.000 GiB |
| KV reserved at 131072 tokens | 32.000 GiB | 8.000 GiB |
| layers that fit (of 64) in 24 GiB free, 2048 / 32768 tokens | 26 / 23 | 26 / 25 |
| layers that fit (of 64) in 40 GiB free, 2048 / 32768 tokens | 44 / 38 | 44 / 42 |
| layers that fit (of 64) in 56 GiB free, 2048 / 32768 tokens | 61 / 53 | 62 / 60 |

`max_layers` charges the KV of n layers for the n contiguous layers that hold the most attention layers (a stage owns a contiguous slice, and
the attention layers are every fourth layer), so a one-layer stage that lands on an attention layer is charged a whole layer's KV. For this
model the rows below are the same as with a model-wide average.

At the default 2048-token context the cap moves by at most one layer, because the weights dominate. It matters for long contexts. The 48
linear-attention layers keep a fixed-size recurrent state per sequence instead of a per-token cache; the planner does not add that state and
I did not measure its size.

Cause and cross-check: oMLX PR 4553 (merged 2026-10-15) made the same correction in its planner for hybrid checkpoints. This repo's
`kv_bytes_per_token` multiplied by `num_hidden_layers`; it now multiplies by the number of attention layers, taken from `layer_types` or, when that is
absent, from `full_attention_interval` as mlx-lm builds the layers.
