"""R9 codec-v1: GGML block layouts, finite BF16 source; no native support claim.

MXFP4 follows half-block nibble order, first-code tie breaking and power-of-two
amax exponent policy. Underflow exponent is clamped to0 (never uint8 wraps).
Q8_0 uses F32 amax/127 and round-away-from-zero, stores FP16 scale; scales
not representable as finite nonzero FP16 are rejected. This is not calibration.
"""
import math
import struct

CODEC_VERSION='R9_GGML_BLOCK_CODEC_V1'
SIZES={'Q8_0':34,'MXFP4':17}
TYPES={'Q8_0':8,'MXFP4':39}
KVALUES=(0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12)


def f32(x):
    try:return struct.unpack('<f',struct.pack('<f',x))[0]
    except OverflowError:raise ValueError('value outside finite F32 range') from None


def bf16_values(raw):
    if len(raw)%2:raise ValueError('odd BF16 byte count')
    return [struct.unpack('<f',struct.pack('<I',x[0]<<16))[0] for x in struct.iter_unpack('<H',raw)]


def encode_block(values, fmt):
    if fmt not in SIZES:raise ValueError('unsupported block format')
    values=[f32(float(v)) for v in values]
    if len(values)!=32 or not all(math.isfinite(v) for v in values):raise ValueError('expected32 finite values')
    amax=max(abs(v) for v in values)
    if fmt=='Q8_0':
        if not amax:return bytes(34)
        d=f32(amax/127)
        try:scale=struct.pack('<e',d)
        except OverflowError:raise ValueError('Q8_0 scale overflow') from None
        decoded=struct.unpack('<e',scale)[0]
        if not math.isfinite(decoded) or decoded==0:raise ValueError('Q8_0 scale underflow/overflow')
        inverse=f32(1/d)
        scaled=[f32(v*inverse) for v in values]
        q=[max(-127,min(127,math.floor(v+.5) if v>=0 else math.ceil(v-.5))) for v in scaled]
        return scale+struct.pack('<32b',*q)
    exponent=max(0,min(254,math.frexp(amax)[1]-1-2+127)) if amax else 0
    half=math.ldexp(1.,exponent-128)
    codes=[min(range(16),key=lambda c:abs(v-KVALUES[c]*half)) for v in values]
    return bytes([exponent]+[codes[i]|(codes[i+16]<<4) for i in range(16)])


def decode_block(raw, fmt):
    if fmt not in SIZES or len(raw)!=SIZES[fmt]:raise ValueError('invalid block length/format')
    if fmt=='Q8_0':
        d=struct.unpack('<e',raw[:2])[0]
        if not math.isfinite(d) or d<0:raise ValueError('invalid Q8_0 scale')
        return [f32(d*q) for q in struct.unpack('<32b',raw[2:])]
    if raw[0]==255:raise ValueError('NaN MXFP4 exponent')
    d=math.ldexp(1.,raw[0]-128)
    values=[f32(KVALUES[b&15]*d) for b in raw[1:]]+[f32(KVALUES[b>>4]*d) for b in raw[1:]]
    if not all(math.isfinite(v) for v in values):raise ValueError('nonfinite decoded block')
    return values
