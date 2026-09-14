# B Round 1.5 — KML Runtime Availability

## Status

**AVAILABLE package; exact A3 OS compatibility UNKNOWN / NOT VERIFIED**

This distinction matters. Huawei's official site exposes KML 2.5.0 artifacts and documents standard headers/libraries, but the inspected HPCKit 24.0 support matrix does not list openEuler 24.03 LTS-SP1. Round 1.5 did not install KML or modify the A3 host.

## Official package evidence

Huawei's [Kunpeng BoostKit KML component page](https://www.hikunpeng.com/boostkit/library/detail?subtab=%E6%95%B0%E5%AD%A6%E5%BA%93&version=2.3.0) identifies the BoostKit 24.0.0 formal KML component as version 2.5.0 and exposes GCC/BiSheng download entries. The official object path is named `KML_2.5.0.zip`; the page also exposes an `.asc` signature artifact.

Therefore `KML 2.5.0 package availability = AVAILABLE` from an official Huawei source. Round 1.5 only verified discovery; it did not download, hash, unpack, or execute that package.

The [Huawei Cloud Kunpeng KML archive](https://mirrors.huaweicloud.com/kunpeng/archive/HPC/KML/) currently lists other releases, including 2.3.0 and 26.1.RC1, but not 2.5.0 in its browsable index. This is fragmented discoverability, not proof that the official 2.5.0 object is absent.

## Installation layout and API evidence

Huawei's [KML overview](https://www.hikunpeng.com/document/detail/zh/kunpengaccel/math-lib/devg-kml/kunpengaccel_kml_0001.html) describes KML as a Kunpeng-targeted math library family.

Official examples document both packaging styles:

- a binary/RPM workflow and KBLAS static libraries, described in the [Kunpeng optimization guide](https://www.hikunpeng.com/document/detail/en/perftuning/progtuneg/kunpengprogramming_05_0042.html);
- a runtime-link workflow using `#include <kblas.h>`, include path `/usr/local/kml/include`, library path `/usr/local/kml/lib`, and `-lkml_rt`, shown in the [KML_BLAS example](https://www.hikunpeng.com/document/detail/zh/hpchistory/hpckit/develop2400/kunpengaccel_kml_1034.html).

The [HPCKit module-directory documentation](https://www.hikunpeng.com/document/detail/zh/hpchistory/hpckit/install2400/KunpengHPCKit_install_013.html) also documents KML 2.5.0 under `/opt/HPCKit/24.12.30/kml/{gcc,bisheng}` with compiler-specific include/library trees. Hence `/usr/local/kml` is a documented conventional layout, but not the only official layout.

Expected items for any future controlled image audit:

```text
KML_2.5.0.zip
kblas.h
libkml_rt.so or equivalent runtime library
KBLAS static/shared libraries appropriate to compiler and ISA
/usr/local/kml/include and /usr/local/kml/lib, or HPCKit module paths
package signature and vendor documentation
```

## openEuler compatibility boundary

The official [HPCKit 24.0 installation/support documentation](https://www.hikunpeng.com/document/detail/zh/hpchistory/hpckit/install2400) lists openEuler 20.03 SP3 and 22.03 SP2/SP3 among supported systems. It does not list the A3's openEuler 24.03 LTS-SP1.

Consequently:

```text
official KML 2.5.0 artifact:             AVAILABLE
documented headers/library conventions:  AVAILABLE
reliable package source:                 AVAILABLE, but fragmented
official openEuler 24.03 SP1 support:     NOT VERIFIED
A3 load/link/runtime behavior:            NOT TESTED
```

Documentation for newer Kunpeng products mentioning openEuler 24.03 cannot be used as proof that this specific KML 2.5.0 package supports it.

## Decision impact

KML runtime availability is not a current blocker because Route A does not use KML. If a later optimization round considers restoration, it must first download and verify the signed artifact in a disposable container, record package contents and licenses, compile a minimal `kblas.h`/`kml_rt` probe, and run numerical tests on the exact A3 OS/kernel environment. No host installation is justified by the present evidence.
