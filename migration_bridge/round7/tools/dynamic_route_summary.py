"""Decode-only exposure, rejecting unassociated/incomplete model forwards.

Prefill routed ownership is not actual CPU execution when Dynamic enters the
full-device path. Keep it out of the post-update Hybrid decode comparison.
The initial-mask lane is a counterfactual on the SAME Dynamic routes; it must
not replace measured Static-vs-Dynamic results, whose routes can differ.
"""


def correlate_updates(updates,requests):
    groups={r['id']:[] for r in requests}
    for event in updates:
        matched=[r['id'] for r in requests
                 if r['started_unix']<=event['time_unix']<=r['ended_unix']]
        if len(matched)!=1:raise ValueError('UNASSOCIATED_OR_AMBIGUOUS_MIGRATION_EVENT')
        groups[matched[0]].append(event)
    triggered=sum(bool(events) for events in groups.values())
    return dict(requests=len(requests),triggered_requests=triggered,
        actual_trigger_rate=triggered/len(requests),
        update_calls=len(updates),copied_expert_slots=sum(len(e['selected']) for e in updates),
        changed_resident_experts=sum(len(set(e['selected'])-set(e['previous_resident'])) for e in updates),
        migration_bytes=sum(e['migration_bytes'] for e in updates),
        per_request=[dict(id=rid,updated_layers=[e['layer'] for e in events]) for rid,events in groups.items()],
        scope='ACTUAL_UPDATE_EVENTS_ASSOCIATED_TO_NONOVERLAPPING_SERIAL_REQUEST_WINDOWS')


def summarize_decode(rows,kept,decode_keys=None,include_counterfactual=True):
    if decode_keys is not None and not decode_keys<=kept:
        raise ValueError('DECODE_KEYS_NOT_ATTRIBUTED')
    result=dict(assignments=0,npu_hits=0,cpu_hits=0,total_weight=0.0,cpu_weight=0.0)
    if include_counterfactual:
        result.update(initial_static_npu_hits=0,initial_static_cpu_weight=0.0)
    groups={}
    per_layer={}
    for r in rows:
        context=r['forward_context']
        key=(context['epoch'],context['forward_id'])
        if key not in kept:continue
        if decode_keys is not None:
            selected=key in decode_keys
            if 'is_decode' in context and context['is_decode'] is not selected:
                raise ValueError('DECODE_ANNOTATION_DISAGREES_WITH_HISTORY')
        elif 'is_decode' in context:
            if not isinstance(context['is_decode'],bool):raise ValueError('INVALID_DECODE_ANNOTATION')
            selected=context['is_decode']
        else:
            raw_name=context.get('mode_name',context.get('mode'))
            if not isinstance(raw_name,str):raise ValueError('UNKNOWN_MODE_REQUIRES_HISTORY_ASSOCIATION')
            name=raw_name.upper()
            if name not in ('DECODE','FORWARDMODE.DECODE','EXTEND','FORWARDMODE.EXTEND'):
                raise ValueError('NUMERIC_OR_UNKNOWN_MODE_REQUIRES_HISTORY_ASSOCIATION')
            selected=name in ('DECODE','FORWARDMODE.DECODE')
        if not selected:continue
        layers=groups.setdefault(key,{})
        if r['layer'] in layers:raise ValueError('DUPLICATE_DECODE_LAYER')
        layers[r['layer']]=r
        for field in result:result[field]+=r[field]
        layer=per_layer.setdefault(str(r['layer']),dict(assignments=0,npu_hits=0,cpu_hits=0))
        for field in layer:layer[field]+=r[field]
    if not groups:raise ValueError('NO_ATTRIBUTED_DECODE_FORWARD')
    token_rows=0;fully_npu_rows=0
    for layers in groups.values():
        if set(layers)!=set(range(1,27)):raise ValueError('DECODE_LAYER_INCOMPLETENESS')
        sizes={r['tokens'] for r in layers.values()}
        if len(sizes)!=1:raise ValueError('DECODE_ROW_SHAPE_DRIFT')
        count=next(iter(sizes));token_rows+=count
        for r in layers.values():
            if len(r['fully_npu_row_flags'])!=count or r['assignments']!=6*count:
                raise ValueError('DECODE_ROUTE_SHAPE_DRIFT')
        fully_npu_rows+=sum(all(r['fully_npu_row_flags'][i] for r in layers.values())
                           for i in range(count))
    if result['assignments']!=token_rows*26*6:raise ValueError('DECODE_ROUTE_COMPLETENESS')
    if decode_keys is not None and set(groups)!=decode_keys:
        raise ValueError('MISSING_ATTRIBUTED_DECODE_FORWARD')
    result.update(npu_hit_rate=result['npu_hits']/result['assignments'],
        cpu_route_rate=result['cpu_hits']/result['assignments'],
        cpu_weight_mass=result['cpu_weight']/result['total_weight'],
        model_token_rows=token_rows,fully_npu_model_token_rows=fully_npu_rows,
        model_token_touches_cpu_fraction=1-fully_npu_rows/token_rows,
        per_layer=per_layer,
        scope='ATTRIBUTED_HYBRID_DECODE_ONLY_PREFILL_AND_UNUSED_TAIL_EXCLUDED')
    if include_counterfactual:
        result.update(initial_static_npu_hit_rate=result['initial_static_npu_hits']/result['assignments'],
            initial_static_cpu_weight_mass=result['initial_static_cpu_weight']/result['total_weight'],
            counterfactual_scope='INITIAL_STATIC_MASK_ON_SAME_DYNAMIC_ROUTES_NOT_SEPARATE_STATIC_RUN')
    return result
