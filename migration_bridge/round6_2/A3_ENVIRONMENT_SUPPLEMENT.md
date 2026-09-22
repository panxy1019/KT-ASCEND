# A3 environment supplement, 2026-09-15

The version collector's optional `version.cfg` paths were absent. This does
not mean the installed CANN version is unknown or that the running metadata
file was edited. A separate read-only check located:

`/usr/local/Ascend/cann-9.0.0/opp/version.info`

It reports `Version=9.0.0`, build timestamp `20260428_134817545`.
File SHA256: `3f6997bfb998c24cf107258e4cd4e76f6ffc520e0382770e75520083c20b162a`.

The actual extension tested in the container is:

`/workspace/results/venv-r6/lib/python3.11/site-packages/kt_kernel/kt_kernel_ext.cpython-311-aarch64-linux-gnu.so`

SHA256: `3fbb95c00c5906b905e8fd689346f4e78f86714fddb6372fa0e481e0da7ce43f`.

During QD2 baseline generation, read-only `npu-smi info` reported NPU0 chip0
Health OK and HBM 40054/65536 MiB. `docker top` attributed host scheduler PID
2397441 and launcher PID 2396763 to this project's history server. The separate
python3.10 workload, including PID 1510639 on chip0, remained running. Other
cards/chips were observed only; no workload was launched or signaled there.

Shared utilization is descriptive and not an isolated throughput benchmark.
No host libraries, driver settings or business containers were modified.
