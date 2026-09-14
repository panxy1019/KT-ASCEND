# Backend Build and Provenance

Container-only packages:

- OpenBLAS `0.3.20+ds-1`, pthread
- BLIS `0.8.1-2`, OpenMP
- ATLAS `3.10.3-12ubuntu1`
- Arm Compute Library `20.08+dfsg-5`
- `libhwloc15` / `libhwloc-dev` for kt-kernel rebuild

Compiler: GCC/G++ `11.4.0`. The OpenBLAS experimental kt-kernel extension built
with the existing frozen CMake build tree. Its experimental SO SHA256 was
`094cca95b633dc2cabc7c6e4e9690c19b0b1a8071fb6f569a41a29e2760959f9`;
the pre-experiment SO was preserved inside the container. No artifact is
shipped because P1 failed.

Library SHA256 values and exact runtime paths are recorded in
`backend_manifests/`. The ACL shim build command is:

```bash
g++ -std=c++17 -O2 -fPIC -shared -I/usr/include/aarch64-linux-gnu \
  tools/arm_compute_cblas_shim.cpp -L/usr/lib/aarch64-linux-gnu \
  -larm_compute -larm_compute_core -lpthread \
  -o libround4a2_acl_cblas.so
```

No untracked `LD_LIBRARY_PATH` setting was used.
