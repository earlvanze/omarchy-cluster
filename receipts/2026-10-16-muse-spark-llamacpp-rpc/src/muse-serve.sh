#!/usr/bin/env bash
# Serve Muse Spark Open (UD-IQ1_S GGUF) over llama.cpp RPC with omarchy-cluster. Run on the gateway machine, the one that has the GGUF on
# its own disk. Every node already runs ggml-rpc-server (see the README). Layers go in the order of the --rpc-node options, after the first
# HOST_LAYERS layers that stay on this machine's CPU.
#
#   GGUF=/models/Muse-Spark-Open-816B-A42B-UD-IQ1_S-00001-of-00005.gguf \
#   LLAMA_SERVER=~/src/llama.cpp/build/bin/llama-server \
#   NODES="NODE_A:50060=43 NODE_A:50061=43 NODE_B:50060=54" HOST_LAYERS=13 ./muse-serve.sh
#
# `=GB` after a node is its memory budget in GB. The run in this receipt used HOST_LAYERS=13, two ggml-rpc-server processes on one 94 GB
# laptop (43 GB each) and one on a 62 GB laptop (54 GB); the planner puts whole layers (2.8 GB each) on each.
set -euo pipefail
: "${GGUF:?path of the first shard, -00001-of-00005.gguf}"
: "${LLAMA_SERVER:?path of a llama-server that has the muse-spark architecture}"
: "${NODES:?space separated host:port=GB list of RPC nodes, in layer order}"
HOST_LAYERS="${HOST_LAYERS:-0}"
args=()
for n in $NODES; do args+=(--rpc-node "$n"); done
omarchy-cluster stop > /dev/null 2>&1 || true
exec omarchy-cluster serve "$GGUF" --engine llamacpp --llama-server "$LLAMA_SERVER" --host-layers "$HOST_LAYERS" "${args[@]}" --ctx 2048 \
  --port 8020 --engine-port 8031 \
  --llama-arg=--no-warmup --llama-arg=-fa --llama-arg=on --llama-arg=-ub --llama-arg=128 --llama-arg=-b --llama-arg=128
