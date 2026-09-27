"""Bounded vector implementation of codec-v1, byte-identical to scalar oracle."""
from .lowbit import KVALUES,SIZES

def encode_blocks(values,fmt):
    import numpy as np
    x=np.asarray(values,dtype=np.float32)
    if fmt not in SIZES or x.size%32 or not np.isfinite(x).all():
        raise ValueError('expected finite complete32 blocks')
    x=x.reshape(-1,32)
    result=np.empty((len(x),SIZES[fmt]),dtype=np.uint8)
    # Cap nearest-code temporary at1024*32*16*8 bytes, independent of layer size.
    for start in range(0,len(x),1024):
        v=x[start:start+1024];maximum=np.max(np.abs(v),axis=1)
        target=result[start:start+len(v)]
        if fmt=='Q8_0':
            scale=(maximum/np.float32(127)).astype(np.float32)
            stored=scale.astype('<f2')
            if not np.isfinite(stored).all() or np.any((maximum!=0)&(stored==0)):
                raise ValueError('Q8_0 scale underflow/overflow')
            inverse=np.divide(np.float32(1),scale,out=np.zeros_like(scale),where=scale!=0)
            scaled=(v*inverse[:,None]).astype(np.float32)
            # scalar codec rounds the F32 product in double precision.
            d=scaled.astype(np.float64)
            q=np.where(d>=0,np.floor(d+.5),np.ceil(d-.5)).clip(-127,127).astype(np.int8)
            target[:,:2]=stored.view(np.uint8).reshape(-1,2)
            target[:,2:]=q.view(np.uint8)
        else:
            exponent=np.where(maximum==0,0,np.clip(np.frexp(maximum)[1]-1-2+127,0,254)).astype(np.uint8)
            half=np.ldexp(np.ones(len(v),dtype=np.float64),exponent.astype(np.int32)-128)
            distance=np.abs(v.astype(np.float64)[:,:,None]-half[:,None,None]*np.asarray(KVALUES,dtype=np.float64))
            codes=distance.argmin(axis=2).astype(np.uint8)
            target[:,0]=exponent
            target[:,1:]=codes[:,:16]|(codes[:,16:]<<4)
    return result


def decode_blocks(raw,fmt):
    import numpy as np
    if fmt not in SIZES:raise ValueError('unsupported block format')
    data=np.asarray(raw,dtype=np.uint8).reshape(-1,SIZES[fmt])
    if fmt=='Q8_0':
        scale=data[:,:2].copy().view('<f2').reshape(-1).astype(np.float32)
        if not np.isfinite(scale).all() or (scale<0).any():raise ValueError('invalid Q8_0 scale')
        return data[:,2:].copy().view(np.int8).astype(np.float32)*scale[:,None]
    if (data[:,0]==255).any():raise ValueError('NaN MXFP4 exponent')
    codes=np.concatenate((data[:,1:]&15,data[:,1:]>>4),axis=1)
    scale=np.ldexp(np.ones(len(data),dtype=np.float64),data[:,0].astype(np.int32)-128)
    result=(np.asarray(KVALUES)[codes]*scale[:,None]).astype(np.float32)
    if not np.isfinite(result).all():raise ValueError('nonfinite decoded block')
    return result
