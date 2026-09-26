"""Eager compressed-QSA primitives following upstream4ccff141.

Single request, explicit ordinary KV and index tensors. No paged pool, scheduler,
RoPE substitution or dense-attention fallback. Selected tokens stay logical IDs.
"""
import math
import torch


def compress_index_keys(raw_keys,norm_weight,rope_positions,rope,*,ratio=4,eps=1e-6):
    if (type(ratio) is not int or ratio<2 or raw_keys.ndim!=2
            or norm_weight.shape!=(raw_keys.shape[-1],)
            or rope_positions.shape[-1]!=raw_keys.shape[0]):
        raise ValueError('invalid index compression layout')
    groups=raw_keys.shape[0]//ratio
    pooled=raw_keys[:groups*ratio].float().reshape(groups,ratio,raw_keys.shape[-1]).mean(1).to(raw_keys.dtype)
    v=pooled.float()
    normalized=(v*torch.rsqrt(v.square().mean(-1,keepdim=True)+eps)*(1+norm_weight.float())).to(raw_keys.dtype)
    return rope(normalized,rope_positions[..., :groups*ratio:ratio])


def select_tokens(index_query,compressed_keys,logical_positions,*,ratio=4,topk_blocks=512):
    if (type(ratio) is not int or ratio<2 or type(topk_blocks) is not int or topk_blocks<1
            or index_query.ndim!=3 or compressed_keys.ndim!=2
            or index_query.shape[-1]!=compressed_keys.shape[-1]
            or logical_positions.shape!=(index_query.shape[0],)
            or logical_positions.dtype not in (torch.int32,torch.int64)):
        raise ValueError('invalid compressed-QSA shape')
    completed=(logical_positions+1)//ratio
    if bool((logical_positions<0).any()) or bool((completed>compressed_keys.shape[0]).any()):
        raise ValueError('logical position exceeds supplied completed index groups')
    width=topk_blocks*ratio+ratio-1
    result=torch.full((index_query.shape[0],width),-1,dtype=torch.long,device=index_query.device)
    k=min(topk_blocks,compressed_keys.shape[0])
    if k:
        score=torch.einsum('thd,gd->thg',index_query.float(),compressed_keys.float()).relu().sum(1)/math.sqrt(index_query.shape[-1])
        visible=torch.arange(compressed_keys.shape[0],device=index_query.device)[None,:]<completed[:,None]
        score=score.masked_fill(~visible,float('-inf'))
        groups=torch.topk(score,k,dim=-1,sorted=True).indices
        valid=torch.arange(k,device=index_query.device)[None,:]<completed.clamp_max(k)[:,None]
        tokens=groups[:,:,None]*ratio+torch.arange(ratio,device=index_query.device)
        tokens=torch.where(valid[:,:,None],tokens,-1).flatten(1)
        result[:,:k*ratio]=tokens
    tails=torch.arange(ratio-1,device=index_query.device)[None,:]
    tail_length=(logical_positions+1)%ratio
    tail_tokens=torch.where(tails<tail_length[:,None],completed[:,None]*ratio+tails,-1)
    tail_columns=completed.clamp_max(topk_blocks)[:,None]*ratio+tails
    result.scatter_(1,tail_columns,tail_tokens)
    return result


def sparse_gqa(query,keys,values,selected):
    if (query.ndim!=3 or keys.ndim!=3 or values.shape!=keys.shape
            or query.shape[-1]!=keys.shape[-1] or keys.shape[1]<1
            or query.shape[1]%keys.shape[1] or selected.ndim!=2
            or selected.shape[0]!=query.shape[0]
            or selected.dtype not in (torch.int32,torch.int64)):
        raise ValueError('invalid sparse GQA layout')
    if bool(((selected<-1)|(selected>=keys.shape[0])).any()):
        raise ValueError('selected token outside ordinary KV cache')
    result=torch.zeros_like(query)
    repeats=query.shape[1]//keys.shape[1]
    for row in range(query.shape[0]):
        ids=selected[row];ids=ids[ids>=0].long()
        if ids.numel()==0:continue
        k=keys.index_select(0,ids).float();v=values.index_select(0,ids).float()
        q=query[row].float().reshape(keys.shape[1],repeats,query.shape[-1])
        score=torch.einsum('khd,skd->khs',q,k)/math.sqrt(query.shape[-1])
        out=torch.einsum('khs,skd->khd',score.softmax(-1),v)
        result[row]=out.reshape_as(query[row]).to(query.dtype)
    return result


def sparse_gqa_vectorized(query,keys,values,selected):
    """Evaluate the same sparse GQA contract with one batched token path."""
    if (query.ndim!=3 or keys.ndim!=3 or values.shape!=keys.shape
            or query.shape[-1]!=keys.shape[-1] or keys.shape[1]<1
            or query.shape[1]%keys.shape[1] or selected.ndim!=2
            or selected.shape[0]!=query.shape[0]
            or selected.dtype not in (torch.int32,torch.int64)):
        raise ValueError('invalid sparse GQA layout')
    width=min(selected.shape[1],keys.shape[0])
    invalid=((selected<-1)|(selected>=keys.shape[0])).any()
    if selected.shape[1]>width:
        invalid=invalid|(selected[:,width:]>=0).any()
    if bool(invalid):
        raise ValueError('selected token outside ordinary KV cache or packed prefix')
    if width==0:
        return torch.zeros_like(query)
    ids=selected[:,:width]
    valid=ids>=0
    safe_ids=ids.clamp_min(0).long()
    tokens=query.shape[0]
    kv_heads=keys.shape[1]
    repeats=query.shape[1]//kv_heads
    head_dim=query.shape[-1]
    k=keys.index_select(0,safe_ids.reshape(-1)).reshape(
        tokens,width,kv_heads,head_dim).float()
    v=values.index_select(0,safe_ids.reshape(-1)).reshape_as(k).float()
    q=query.float().reshape(tokens,kv_heads,repeats,head_dim)
    score=torch.einsum('tkrd,tskd->tkrs',q,k)/math.sqrt(head_dim)
    safe_mask=valid.clone()
    safe_mask[:,0]|=~valid.any(-1)
    score=score.masked_fill(~safe_mask[:,None,None,:],float('-inf'))
    weights=score.softmax(-1)*valid[:,None,None,:]
    out=torch.einsum('tkrs,tskd->tkrd',weights,v)
    return out.reshape_as(query).to(query.dtype)


def sparse_gqa_strategy(query_tokens):
    if type(query_tokens) is not int or query_tokens<1:
        raise ValueError('invalid sparse GQA query token count')
    return 'ROW' if query_tokens==1 else 'VECTORIZED'


def sparse_gqa_candidate(query,keys,values,selected):
    function=(sparse_gqa if sparse_gqa_strategy(query.shape[0])=='ROW'
              else sparse_gqa_vectorized)
    return function(query,keys,values,selected)


def text_rope(x,positions,*,rotary_dim=64,theta=10000000.0):
    """NeoX partial RoPE for one-axis text positions only.

    With equal text coordinates on all three MRoPE axes this equals the frozen
    model's MRoPE. Multi-axis/VL input is deliberately rejected here.
    """
    if (x.ndim not in (2,3) or positions.ndim!=1 or positions.shape[0]!=x.shape[0]
            or type(rotary_dim) is not int or rotary_dim<2 or rotary_dim%2
            or rotary_dim>x.shape[-1] or theta<=0
            or positions.dtype not in (torch.int32,torch.int64)
            or bool((positions<0).any())):
        raise ValueError('unsupported text rotary layout')
    inv=1.0/(theta**(torch.arange(0,rotary_dim,2,device=x.device,dtype=torch.float32)/rotary_dim))
    angle=positions.float()[:,None]*inv[None,:]
    cos=angle.cos().to(x.dtype);sin=angle.sin().to(x.dtype)
    if x.ndim==3:cos=cos[:,None,:];sin=sin[:,None,:]
    first,second=x[...,:rotary_dim].chunk(2,-1)
    rotated=torch.cat((first*cos-second*sin,second*cos+first*sin),-1)
    return torch.cat((rotated,x[...,rotary_dim:]),-1)
