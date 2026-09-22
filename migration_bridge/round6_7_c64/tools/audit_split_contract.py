#!/usr/bin/env python3
"""Arithmetic counterexample, not an operator qualification or model result."""
import json
import struct


def bf16(value):
    bits = struct.unpack('<I', struct.pack('<f', value))[0]
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack('<f', struct.pack('<I', rounded))[0]


def main():
    partition = 1408 // 2
    covered = (partition // 256) * 256
    p0, p1 = 1.00390625, 0.00390625
    split_round = bf16(p0) + bf16(p1)
    full_round = bf16(p0 + p1)
    assert covered == 512 and partition - covered == 192
    assert split_round != full_round
    print(json.dumps({
        'evidence_state': 'ANALYTIC_DERIVED',
        'operator_executed': False,
        'partition': partition,
        'forward_many_tile': 256,
        'covered_channels': covered,
        'uncovered_channels_if_guard_bypassed': partition - covered,
        'tile32_covers_partition': partition % 32 == 0,
        'partial_down_values': [p0, p1],
        'round_partials_then_sum': split_round,
        'sum_then_round_full_down': full_round,
        'c64_qualified': False,
        'decision': 'KEEP_TP1_GUARDS_NO_LONG_C64_RUN',
    }, indent=2))


if __name__ == '__main__':
    main()
