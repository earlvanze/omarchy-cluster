# Driver and build

2026-10-11. What the records show about the software behind the speed-table rows dated 2026-10-11,
and what they do not show.

The run scripts pin the rank ICD through `VK_DRIVER_FILES` to a Honeykrisp build directory whose
build record is `6543eeb7df7` (each laptop keeps several driver builds). The record names the Mesa
commit `6543eeb7df702b6bcaa4e02a18425571305d4e05` ("hk: expose VK_KHR_shader_bfloat16 with bf16
cooperative matrices", 2026-10-07). The runs' provenance logs record the hash of the pinned ICD
JSON, not the bytes of the library behind it, so the library the 2026-10-11 runs loaded is not
separately recorded.

Two records tie the library to the commit. The weight-load run on the 13-inch M1 the same day
logged, for the ICD it loaded, `icd <the pinned JSON> lib <the Honeykrisp ICD library> sha256
3546bcafe8ed3995`. After the runs, the copies of the pinned build directory on both laptops hold
byte-identical Honeykrisp ICD libraries with sha256
`3546bcafe8ed3995d091debcc2fb00803087e0074a88d7ebfe681071c58e57d0` (checked over ssh), and both
directories carry the same build record `6543eeb7df7`.

Software. The runs used the cluster rank venv (a separate venv from the laptops' main one):

- omarchy-mlx wheel: `0.32.4.dev202610102049+93131d1b` (the pip list line
  `wheel mlx-omarchy 0.32.4.dev202610102049+93131d1b` is in the weight-load log; the laptops' main
  venvs held `0.32.4.dev202610111010+4ba8fff` that day).
- mlx-lm: `0.31.4.dev132`.
- omarchy-cluster: the provenance line `omarchy-cluster src bd6e3b22828d3db8` (a hash over the
  package's `.py` files, kept in [data](data)). It matches the tree at `3c82f0a`, or a later
  docs-only commit (the hash covers only the package `.py` files; every commit after `3c82f0a`
  touches docs only).

The run and the model are in
[receipts/2026-10-11-single-host-baseline](../2026-10-11-single-host-baseline).
