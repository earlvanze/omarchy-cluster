"""w7G: planner KV reservation for a real hybrid config, before and after the fix. Usage: python3 measure.py <repo dir> <model dir>
Reads only config.json and the safetensors file sizes. No node facts, no network."""
import sys

sys.path.insert(0, sys.argv[1])
from omarchy_cluster import planner

info = planner.model_info(sys.argv[2])
GiB = 1024 ** 3
kv_tok = planner.kv_bytes_per_token(info)
print("layers", info["layers"], "kv_layers", info.get("kv_layers", "(absent: all layers)"), "kv_heads", info["kv_heads"], "head_dim", info["head_dim"])
print("kv_bytes_per_token", kv_tok, "= %.1f KiB" % (kv_tok / 1024))
for ctx in (2048, 32768, 131072):
    print("KV reserved at ctx %6d: %8.3f GiB" % (ctx, kv_tok * ctx / GiB))
print("weights %.2f GiB" % (info["weights_bytes"] / GiB))
for free_gib in (24, 40, 56):
    print("max_layers at free %2d GiB, ctx 2048: %2d of %d; ctx 32768: %2d" % (
        free_gib, planner.max_layers(free_gib * GiB, info), info["layers"],
        planner.max_layers(free_gib * GiB, info, 32768)))
