"""Bounded shutdown handshake; a missing container is never treated as an ACK."""
import json
import time


def wait_for_guard_exit(path, *, not_before, timeout=30):
    deadline = time.monotonic() + timeout
    while True:
        if path.exists():
            with path.open('rb') as stream:
                stream.seek(max(0, path.stat().st_size - 65536))
                tail = stream.read().decode(errors='replace').splitlines()
            for line in reversed(tail):
                try:
                    event = json.loads(line)
                except (ValueError, UnicodeError):
                    continue
                if (event.get('action') in ('OWNED_EXITED', 'OWNED_EXITED_DURING_SAMPLE')
                        and event.get('time', 0) >= not_before):
                    return event
        if time.monotonic() >= deadline:
            raise TimeoutError('guard did not acknowledge stopped owned container; removal must not race telemetry')
        time.sleep(0.2)
