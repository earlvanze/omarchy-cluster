# Driver and build

2026-10-11. Records the Vulkan driver, the omarchy-mlx wheel and the omarchy-cluster tree that the
speed-table rows dated 2026-10-11 ran on.

The Mesa Vulkan driver that omarchy-mlx loads was built from the joshuaswarren/mesa-1 checkout at
commit `6543eeb7df702b6bcaa4e02a18425571305d4e05` ("hk: expose VK_KHR_shader_bfloat16 with bf16
cooperative matrices", 2026-10-07). The 11-character prefix `6543eeb7df7` is the build record written
next to the library on every laptop that ran the table. The library `libvulkan_asahi.so` has SHA-256
`3546bcafe8ed3995d091debcc2fb00803087e0074a88d7ebfe681071c58e57d0` and is byte-identical on the
16-inch M1 Max MacBook Pro and the 13-inch M1 MacBook Pro. The rank wrapper selects it through
`VK_DRIVER_FILES`.

Software:

- omarchy-mlx wheel: `0.32.4.dev202610102049+93131d1b`.
- mlx-lm: `0.31.4.dev132`.
- omarchy-cluster tree: `3c82f0a` (the run SHA-256 over the package's `.py` files matches the
  checkout).

The run and the model are in
[receipts/2026-10-11-single-host-baseline](receipts/2026-10-11-single-host-baseline).
