# Round8 isolated environment decision

Use the already isolated container image's pinned system torch/torch_npu/CANN
stack, not a blind pip upgrade to the latest official dependencies. Actual Qwen
imports, real expert and NPU GDN/convolution primitives have run on this stack.

Reference venv-r6 itself uses include-system-site-packages=true; its private
packages occupy19MiB. Therefore a new Round8 venv can inherit the SAME isolated
container image's system packages while owning its own private package directory.
This does not share writable packages with Round7 containers. Source mounts stay
read-only and no host or Round7 Python installation is modified.

Create `/workspace/round8/venv` with system-site-packages and copy the frozen
backing-fix native KT package into its private site-packages. Verify native hash
and actual imported paths. This is a provenance-preserving binary transplant,
NOT a fresh source build. Any future native modification needs independent build
and regression. Record image/system package inventory; no claim of a wheel-only
reproducible clean build yet. All future installs/build outputs must be private.

Environment entry log `independent-env-entry-v1.log`; reference package inventory
`reference-package-list-v1.json` on .22. Re-run representative primitive/operator
checks using the new venv before full-model qualification.

Entry verified: torch2.9.0+cpu,torch_npu2.9.0.post2; actual KT import is the
new `/workspace/round8/venv/lib/python3.11/site-packages/kt_kernel/__init__.py`.
Copied native SHA46182176279b072b4ab52d21bb8cc0ecf4d217399d486155c7fcd4f0690e21e2
matches the frozen source binary. Started private-env LLAMAFILE C64 probe then
GDN continuous-state replay, log `private-env-regression-v1.log`; results
`llamafile-private-env-v1/result.json`, `gdn-private-env-v1.json`. Inspect before
relaunch. Full checkpoint v2 download driverPID2254561 is independent/healthy.
