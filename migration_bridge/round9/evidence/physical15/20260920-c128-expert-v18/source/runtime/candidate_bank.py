"""Full-bank identity binding. Qualification and resource admission are separate."""
import hashlib,json,os
from pathlib import Path
from .candidate_config import load_candidate_profile
from .probe_identity import verify_probe_identity
REVISION='de4b8e4d43b917e7706784d8bb445c9af86a3540'
RECEIPT_SHA='7d6cb3db19497303bdd8ceb24efae626e1926a610ab7fbd4406b47d05db4a239'
MANIFEST_SHA='62e6eca0f2900745dbfd9ba6ed76e60547e0d602acf2d62a9ecac3972028b38e'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def signature(path):
    s=path.stat();return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino)

def source_identity(model,receipt):
    if not os.statvfs(model).f_flag&os.ST_RDONLY:raise RuntimeError('original checkpoint must be mounted read-only')
    return verify_probe_identity(model=model,receipt=receipt,expected_model=Path('/models')/REVISION,
        expected_receipt_sha256=RECEIPT_SHA,revision=REVISION,manifest_sha256=MANIFEST_SHA)

def verify_catalog(profile,bank_root,model,receipt):
    import gguf
    candidate,_,profile_sha=load_candidate_profile(profile)
    root=Path(bank_root).resolve(strict=True)
    if not os.statvfs(root).f_flag&os.ST_RDONLY:raise RuntimeError('banks must be mounted read-only')
    identity=source_identity(model,receipt)
    catalog=root/'catalog.json';data=json.loads(catalog.read_text());rows=data['banks']
    if data['source_identity']!=identity or len(rows)!=48 or {r['layer'] for r in rows}!=set(range(48)):
        raise RuntimeError('incomplete or unrelated bank catalog')
    expected_type={'BF16':30,'Q8_0':8,'MXFP4':39}[candidate.storage]
    files=[]
    for row in sorted(rows,key=lambda r:r['layer']):
        layer=row['layer'];path=root/row['file']
        if Path(row['file']).name!=row['file'] or row['source_identity']!=identity:
            raise RuntimeError('invalid bank path/source')
        if row['storage'] not in (candidate.storage, 'BF16_ORIGINAL_BYTES' if candidate.storage=='BF16' else candidate.storage):
            raise RuntimeError('candidate storage differs from catalog')
        if path.stat().st_size!=row['bytes'] or digest(path)!=row['sha256']:
            raise RuntimeError('bank SHA or size mismatch')
        reader=gguf.GGUFReader(str(path));tensors={t.name:t for t in reader.tensors}
        names={f'blk.{layer}.ffn_{p}_exps.weight' for p in ('gate','up','down')}
        if set(tensors)!=names:raise RuntimeError('bank tensor set mismatch')
        for name,tensor in tensors.items():
            shape=(640,2560,512) if 'ffn_down_' in name else (2560,640,512)
            if int(tensor.tensor_type)!=expected_type or tuple(tensor.shape.tolist())!=shape:
                raise RuntimeError('actual bank type or geometry differs from candidate')
        files.append(dict(file=row['file'],sha256=row['sha256'],tensor_type=expected_type,**signature(path)))
        del tensors,reader
    if {str(p.relative_to(root)) for p in root.rglob('*.gguf')}!={f['file'] for f in files}:
        raise RuntimeError('unlisted GGUF may be loaded')
    return dict(status='VERIFIED48_BANK_BYTES_TYPES_SOURCE_NOT_MODEL_QUALIFIED',profile_sha256=profile_sha,
        catalog_sha256=digest(catalog),bank_root=str(root),source_identity=identity,
        source_receipt=str(Path(receipt).resolve()),banks=files)

_CACHE={}
def bind_verified_banks(profile,bank_root,model):
    """Require the exact CPU verifier receipt and unchanged read-only payload."""
    path=os.environ.get('ROUND9_BANK_RECEIPT');sha=os.environ.get('ROUND9_BANK_RECEIPT_SHA256')
    if not path or not sha or digest(path)!=sha:raise RuntimeError('verified candidate bank receipt required')
    _,_,profile_sha=load_candidate_profile(profile)
    root=Path(bank_root).resolve(strict=True);report=json.loads(Path(path).read_text())
    if (report['status']!='VERIFIED48_BANK_BYTES_TYPES_SOURCE_NOT_MODEL_QUALIFIED' or
        report['profile_sha256']!=profile_sha or report['bank_root']!=str(root) or
        report['catalog_sha256']!=digest(root/'catalog.json') or len(report['banks'])!=48):
        raise RuntimeError('candidate bank receipt binding mismatch')
    if not os.statvfs(root).f_flag&os.ST_RDONLY:raise RuntimeError('verified banks must remain read-only')
    for item in report['banks']:
        if signature(root/item['file'])!={k:item[k] for k in ('bytes','mtime_ns','inode')}:
            raise RuntimeError('bank changed after full verification')
    key=(sha,str(Path(model).resolve()))
    if key not in _CACHE:
        if source_identity(model,report['source_receipt'])!=report['source_identity']:
            raise RuntimeError('candidate original source identity mismatch')
        _CACHE[key]=True
    return dict(receipt_sha256=sha,catalog_sha256=report['catalog_sha256'],source_identity=report['source_identity'],layer0_sha256=report['banks'][0]['sha256'])
