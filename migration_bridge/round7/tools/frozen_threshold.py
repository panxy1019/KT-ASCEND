"""Verify explicit commit acknowledgment before any post-DEV threshold use.

The acknowledgment is produced only after the agent checks that the named local
commit contains the exact selected JSON and mask. A3 has no local audit git tree:
this verifies the handoff binding, not a cryptographic Git attestation.
"""
import hashlib
import json
import re


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_ack(selected, ack, selected_sha, profile_sha):
    if not re.fullmatch(r'[0-9a-f]{40}', ack.get('local_candidate_commit', '')):
        raise ValueError('MISSING_THRESHOLD_LOCAL_COMMIT_FREEZE')
    if ack.get('threshold_freeze_sha256') != selected_sha:
        raise ValueError('THRESHOLD_FREEZE_DRIFT')
    if selected.get('status') != 'FROZEN_NOT_HELDOUT_QUALIFIED':
        raise ValueError('THRESHOLD_NOT_FROZEN')
    if selected.get('threshold') not in (256, 512, 1024):
        raise ValueError('THRESHOLD_OUTSIDE_FROZEN_DEV_CANDIDATES')
    if selected.get('profile_sha256') != profile_sha or ack.get('profile_sha256') != profile_sha:
        raise ValueError('THRESHOLD_STATIC_PROFILE_DRIFT')
    if not re.fullmatch(r'[0-9a-f]{64}', selected.get('dev_sha256', '')):
        raise ValueError('THRESHOLD_DEV_PROVENANCE_MISSING')
    if ack.get('dev_sha256') != selected['dev_sha256']:
        raise ValueError('THRESHOLD_DEV_PROVENANCE_DRIFT')
    if selected.get('quality_used_for_selection') is not False or selected.get('heldout_used_for_selection') is not False:
        raise ValueError('THRESHOLD_SELECTION_CONTAMINATION')


def load_threshold(path, acknowledgment, profile):
    if path is None or acknowledgment is None:
        raise ValueError('COMMITTED_THRESHOLD_AND_ACK_REQUIRED')
    selected = json.loads(path.read_text())
    ack = json.loads(acknowledgment.read_text())
    validate_ack(selected, ack, sha(path), sha(profile))
    return selected, ack
