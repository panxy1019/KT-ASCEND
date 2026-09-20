import ctypes as c
import json
import os
import sys
import glob
r={'status':'BLOCKED','visible':os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),'compute_devices':glob.glob('/dev/davinci[0-9]*')}
a=c.CDLL('/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libascendcl.so')
a.aclInit.argtypes=[c.c_char_p]; a.aclInit.restype=c.c_int
r['aclInit']=a.aclInit(None)
try:
    n=c.c_uint32(); a.aclrtGetDeviceCount.argtypes=[c.POINTER(c.c_uint32)]; a.aclrtGetDeviceCount.restype=c.c_int
    r['count_rc']=a.aclrtGetDeviceCount(c.byref(n)); r['count']=n.value
    if r['count_rc']==0 and n.value==1:
        p=c.c_int32(-1); a.aclrtGetPhyDevIdByLogicDevId.argtypes=[c.c_int32,c.POINTER(c.c_int32)]; a.aclrtGetPhyDevIdByLogicDevId.restype=c.c_int
        r['mapping_rc']=a.aclrtGetPhyDevIdByLogicDevId(0,c.byref(p)); r['physical_id']=p.value
        if r['aclInit']==0 and r['mapping_rc']==0 and p.value==15 and r['compute_devices']==['/dev/davinci15']: r['status']='PASS_MAPPING_ONLY'
finally:
    if r['aclInit']==0: r['aclFinalize']=a.aclFinalize()
    print(json.dumps(r),flush=True)
sys.exit(0 if r['status']=='PASS_MAPPING_ONLY' else 2)
