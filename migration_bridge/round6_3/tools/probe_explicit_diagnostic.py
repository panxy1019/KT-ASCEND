"""Diagnostic-only adapter for the private-buffer replay oracle.

Production transition validators remain unchanged. Oracle uses private pending
storage, so normal pending state must remain EMPTY, even with delayed routes.
After the shared capture/history checks, independently require private output
to equal consumer pending input and require the consumer output capture.
"""
import json
from pathlib import Path
import sys


def main():
    import audit_deferred_transitions as transitions
    import run_stateful_probe as probe
    import torch
    original = transitions.validate_transition
    # Local diagnostic process only: never imported by a qualification runner.
    def oracle_transition(events, deferred_routes):
        return original(events, 0)
    transitions.validate_transition = oracle_transition
    probe.main()
    output = Path(sys.argv[sys.argv.index('--output')+1])
    result = json.loads((output/'result.json').read_text())
    private_count = 0
    for row in result['rows']:
        for step in row['steps']:
            data = torch.load(step['file'], map_location='cpu', weights_only=False, mmap=True)
            producer = 'r62.cpu.layer7.explicit_deferred_output'
            consumer = 'r62.cpu.layer8.explicit_pending_input'
            merged = 'r62.cpu.layer8.explicit_incremental_output'
            if step['current_deferred']:
                assert all(k in data for k in (producer,consumer,merged)), 'ORACLE_STAGE_MISSING'
                assert torch.equal(data[producer],data[consumer]), 'PRIVATE_PENDING_IDENTITY'
                assert torch.equal(data[merged],data['r62.cpu.layer8.cpu_output_after_sync']), 'PRIVATE_MERGE_OUTPUT'
                private_count += 1
            else:
                assert not any(k in data for k in (producer,consumer,merged)), 'STALE_PRIVATE_PENDING'
    with (output/'oracle-audit.json').open('x') as f:
        json.dump(dict(status='DIAGNOSTIC_ORACLE_TRACE_PASS',private_forwards=private_count,
            production_transition_gate='NOT_APPLICABLE_PRIVATE_ORACLE',qualification=False),f,indent=2)


if __name__ == '__main__':
    main()
