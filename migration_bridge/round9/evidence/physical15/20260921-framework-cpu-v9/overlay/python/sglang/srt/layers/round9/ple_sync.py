"""Eager synchronous Qwen4 PLE equations with caller-owned convolution state.

Reference path only; no scheduler, speculation, asynchronous gather or graph.
Matches upstream4ccff141 Qwen4ExpPLELayer's non-fused math for unpadded TP1 rows.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .hyperconnection import GroupedGemmaRMSNorm

class SynchronousPLE(nn.Module):
    def __init__(self,*,hidden_size=2560,embedding_size=2560,kernel_size=4,dilation=3,
                 dtype=torch.bfloat16,device=None):
        super().__init__()
        if any(type(v) is not int or v<1 for v in (hidden_size,embedding_size,kernel_size,dilation)):
            raise ValueError('positive PLE dimensions required')
        self.hidden_size=hidden_size;self.embedding_size=embedding_size
        self.channels=4*hidden_size;self.state_len=(kernel_size-1)*dilation
        self.dilation=dilation
        kw=dict(bias=False,dtype=dtype,device=device)
        self.key_proj=nn.Linear(embedding_size,self.channels,**kw)
        self.value_proj=nn.Linear(embedding_size,hidden_size,**kw)
        for name in ('norm_key','norm_query','norm_conv'):
            setattr(self,name,GroupedGemmaRMSNorm(self.channels,group_size=hidden_size,device=device))
        self.conv1d=nn.Conv1d(self.channels,self.channels,kernel_size,groups=self.channels,dilation=dilation,**kw)

    def forward(self,hidden,embeddings,state=None):
        if (hidden.ndim!=2 or hidden.shape[-1]!=self.channels
                or embeddings.shape!=(hidden.shape[0],self.embedding_size)):
            raise ValueError('PLE input shape mismatch')
        if state is None:state=hidden.new_zeros(self.channels,self.state_len)
        if state.shape!=(self.channels,self.state_len) or state.device!=hidden.device or state.dtype!=hidden.dtype:
            raise ValueError('PLE convolution state identity mismatch')
        if hidden.shape[0]==0:return hidden.clone(),state.clone()
        key=self.norm_key(self.key_proj(embeddings)).unflatten(-1,(4,self.hidden_size))
        query=self.norm_query(hidden).unflatten(-1,(4,self.hidden_size))
        score=(key*query).sum(-1,keepdim=True)/math.sqrt(self.hidden_size)
        gate=torch.sigmoid(score.abs().clamp_min(1e-6).sqrt()*score.sign())
        value=self.value_proj(embeddings)
        gated=(gate*value.unsqueeze(-2)).flatten(-2)
        normalized=self.norm_conv(gated)
        sequence=torch.cat((state,normalized.T),dim=-1)
        conv=F.conv1d(sequence.unsqueeze(0),self.conv1d.weight.to(hidden.dtype),
                      dilation=self.dilation,groups=self.channels).squeeze(0).T
        next_state=sequence[:,hidden.shape[0]:].contiguous()
        return gated+F.silu(conv),next_state
