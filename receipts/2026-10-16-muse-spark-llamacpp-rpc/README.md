# Muse Spark Open 816B-A42B (UD-IQ1_S) across a Mac Studio and two Omarchy Linux MacBook Pros (2026-10-16)

One llama.cpp endpoint for a 175.6 GB mixture-of-experts model that no machine here can hold. `omarchy-cluster serve --engine llamacpp` split
the 62 layers over three machines with llama.cpp RPC. This is the GGUF route (IQ1_S, 1.56 bits per weight). It is not an MLX run. The MLX route
needs a quantization that fits the same machines and is not measured here.

## Setup

| Item | Value |
|---|---|
| Model | `unsloth/Muse-Spark-Open-816B-A42B-GGUF`, `UD-IQ1_S`, 5 shards, 175,639,574,392 bytes, every shard sha256 checked against Hugging Face |
| Layers | 62, each 2,795,767,296 bytes (read from the shard headers); 16 full-attention and 46 sliding-window layers (window 2048) |
| omarchy-cluster | `4750ca7` (llama.cpp engine with `--llama-arg`, sliding-window KV charge) |
| llama.cpp | Muse support is pull request 30616. Laptops: `ggml-rpc-server` built from `335916ad3f0cab876a7a3cf16faaaaea28bf9a96` (Vulkan, RPC), sha256 `f620b98ab3bca2a0e9f4870d828ba9cef999a1cca8bf2e96f83f66376e7c8618`. Gateway: `7206269`, the same commit plus one line that turns off whole-mapping read-ahead. |
| Gateway flags | `--no-warmup -fa on -ub 128 -b 128`, context 2048, the output matrix stays on the gateway CPU |
| Prompts | raw `/completion`, temperature 0, `cache_prompt` off. The prompt is the model's own chat template, rendered, and ends `<|start|>assistant to=user<|message|>` so the model answers without reasoning first. Files in `prompts/`. |

## Layer-to-node map

| Node | Backend | Layers | Weights | Link to the gateway |
|---|---|---|---|---|
| Mac Studio M1 Ultra 128 GB, macOS (gateway) | CPU, layers read from its own disk by mmap | 0-12 (13) | 36.3 GB | local |
| 14-inch MacBook Pro M2 Max 96 GB, Omarchy, server 1 | Vulkan (Honeykrisp), `HK_SYSMEM=45000000000` | 13-27 (15) | 41.9 GB | 2.5 Gb Ethernet |
| 14-inch MacBook Pro M2 Max, server 2 | Vulkan (Honeykrisp), `HK_SYSMEM=45000000000` | 28-42 (15) | 41.9 GB | 2.5 Gb Ethernet |
| 16-inch MacBook Pro M1 Max 64 GB, Omarchy | Vulkan (Honeykrisp), `HK_SYSMEM=57000000000` | 43-61 (19) | 53.1 GB | Thunderbolt |

The laptops hold no copy of the weights. They receive them by RPC at load. The M2 Max runs two servers because one Honeykrisp process cannot map
more than about 63 GiB. Command: [src/muse-serve.sh](src/muse-serve.sh).

## Results

Run on 2026-10-16. Greedy decoding. Raw output: [data/](data), one file per request with every token and its arrival time.

| Measure | Value |
|---|---|
| Load, launch to healthy | 857 s |
| First request, time to first token | 72.3 s (161 prompt tokens, cold) |
| Later requests, time to first token | 25.5 to 26.8 s (158 to 161 prompt tokens, `cache_n` 0) |
| Prompt processing, later requests | 5.9 to 6.2 tokens/s |
| Decode, 107 tokens (code answer) | 3.04 tokens/s (329 ms per token) |
| Decode, 300 tokens (reasoning prompt, stopped at the limit) | 3.17 tokens/s |
| Gateway swap during the load | 10,938 MB at start, -8 MB change; free memory 43 to 51 percent |

Answers, with the answer channel forced:

| Prompt | Text | Tokens | Stop |
|---|---|---|---|
| What is 17 times 23? Answer with only the number. (4 runs) | ` 391` | 2 | end of turn |
| What is the capital of Australia? Answer in one sentence. | `The capital of Australia is Canberra.` | 7 | end of turn |
| Write a Python function is_prime(n). Code only. | a trial-division function (6k plus or minus 1 step) in a code block | 107 | end of turn |

An independent check ran the returned code on 13 cases (primes, 0, 1, negatives, 7917, 91): 13 correct.

With the reasoning channel open (`prompts/canary.medium.think.txt`) the model wrote private reasoning, computed 17 x 20 + 17 x 3 = 391 twice, and
then kept restating that it should output 391 until the 300-token limit. It did not reach an answer. Treat that as a soft loop.

## Limits

- One greedy sample per prompt and three short prompts. This shows the route works and the answers are coherent. It is not a quality benchmark.
- The numbers are for IQ1_S through llama.cpp. They say nothing about an MLX build.
- The first request is slower because the gateway reads its layers from disk the first time. Later requests find them in memory.
- Other jobs ran on the laptops that night. The load average was not recorded for this run.
- The gateway was a Mac Studio that also served another model. A guard stopped the run on a swap rise above 6 GB, free memory below 10 percent or a slow
  router; none of them fired.
