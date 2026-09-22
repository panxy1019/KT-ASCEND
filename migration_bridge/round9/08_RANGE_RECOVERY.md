# Download interruption and bounded recovery

## Observed cause versus unresolved network attribution

Installed huggingface_hub file_download.py streaming handler catches
httpx.ConnectError and TimeoutException, but not the observed ReadError or
RemoteProtocolError. Consequently interrupted bodies escape retry handling
and terminate the whole snapshot process. The wrapper propagated failure;
manual restarts did not provide a reliable transfer strategy.

Two proxy attempts reported about2.87GB received out of3355443376bytes.
This does not establish a fixed server limit or identify who closed the
connection. Shared gigabit traffic and rx_errors were observed separately;
their causal relation to the TLS/peer-close failures remains unresolved.
The client can also reset an incomplete file if a requested Range returns200;
that behavior exists in installed code but was not captured for these failures.

References inspected: https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/file_download.py
and https://curl.se/docs/manpage.html . No wholesale library upgrade performed.

## Reproducer and transport repair

Frozen shard00048, requested bytes2862612480-2863661055 via official HTTPS
through the existing proxy: response206, content-length1048576,
content-range bytes2862612480-2863661055/3355443376. Tail is retrievable.

New range_recovery.py uses32MiB requests, HTTP1.1, Accept-Encoding:identity,
TLS verification and90s bounded requests. Each transport failure retries at
most3 times with separate evidence. Incorrect range/length hard-stops.
The original HF incomplete file is copied, not overwritten. Each accepted
chunk is appended only at its validated offset and fsynced; full official SHA
must pass before atomic no-clobber model publication. Entire model identity
and tensor census remain mandatory at end. No unbounded restart loop.

Existing completed files are not redownloaded; final audit checks their hashes.
Missing metadata is fetched from the same frozen allowlist, checked by SHA/Git
blob before publication. First attempt stopped safely on absent index; this
missing-metadata handling was added and the tooling suite rerun.

Tests: wrong/ignored range does not append; SHA mismatch cannot publish;
publication cannot overwrite; existing range/identity/endpoint tests retained.
11/11 targeted Round9 tooling tests pass. No model/runtime code changed.

## Execution

Owned .22 unit kt-round9-range-recovery-v1 replaces FAILED HF download units.
Retains existing proxy tunnel,8GiB memory,swap0,CPU400%,IO300MB/s and resource
pause guards. Stage range-recovery-v1 retains chunk headers/failures/receipts.
Controller flock prevents duplicate range controllers; do not restart any old
HF unit concurrently. Current implementation SHA:
4504dc48a8700304f7cb0fb27389b11bbdfd51c37df3586336003164ea6bbf56.

## Real recovery result

Shard00048 completed3355443376bytes and passed officialSHA:
dbbf3a6e2b81c8cf9eb152c03e07550741b6e143121fbe579e3b32cdf8836a33.
The seeded historical incomplete plus fetched tail therefore reproduced the
frozen original exactly. It was published without overwriting any model file.
At1789862358 the controller was RUNNING on missing00050 (00049 present).
This validates recovery for the previously blocked shard, not entire model
completion or removal of the underlying network cause.
