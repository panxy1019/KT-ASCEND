"""Receipt predicates shared by bounded HTTP drivers; no runtime mutation."""


def drained_request_ids(events, mode):
    if mode not in ('request_boundary', 'prefill_end'):
        raise ValueError('unknown Dynamic drain mode')
    drained = set()
    for event in events:
        rid = event.get('request_id')
        if not isinstance(rid, str) or not rid:
            continue
        if mode == 'prefill_end':
            complete = (event.get('status') == 'PREFILL_END_REQUEST_DRAINED'
                        and event.get('changed_layers') == 0
                        and event.get('copied_experts') == 0)
        else:
            # The established request-boundary commit has no status field.
            # A prefill commit is not a request drain, even if it names the rid.
            complete = ('status' not in event and
                        type(event.get('generation')) is int and
                        type(event.get('changed_layers')) is int and
                        type(event.get('copied_experts')) is int and
                        isinstance(event.get('plans'), list))
        if complete:
            drained.add(rid)
    return drained
