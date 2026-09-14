# CANN Runtime API Audit

## Headers and library

- header: `/usr/local/Ascend/cann-9.0.0/aarch64-linux/include/acl/acl_rt.h`
- runtime: `/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libacl_rt.so`
- CMake discovery root: `ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0`
- production adapter includes only the public CANN Runtime header and links only `acl_rt`.

## Verified public APIs

An aarch64 compile/link/run probe exercised `aclInit`, stream creation/destruction, `aclrtLaunchHostFunc`, `aclrtMallocHost`, `aclrtFreeHost`, and `aclrtMemcpyAsync`; compile, link and execution all returned success.

The installed signature is:

```cpp
using aclrtHostFunc = void (*)(void *);
aclError aclrtLaunchHostFunc(aclrtStream stream, aclrtHostFunc fn, void *userData);
```

`cpu_backend/vendors/ascend.h` maps this to the common vendor interface and checks both null handles and every CANN return code. Errors retain the API name, numeric error code and C++ source location. No CANN error is silently discarded.

## Build isolation

`KTRANSFORMERS_USE_ASCEND` is a single opt-in switch. When ON, CMake finds the public header and `libacl_rt`; when OFF, `ASCEND_HOME_PATH` is empty, the cache records `KTRANSFORMERS_USE_ASCEND:BOOL=OFF`, and `readelf -d` shows no ACL/Ascend dynamic dependency.

Evidence: `cann-api-symbols.log`, `cann-api-compile-final.log`, `ascend-build-second.log`, `cpu-only-build.log`.
