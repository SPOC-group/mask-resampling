"""Verify the exact external pretraining shard; no corpus is distributed."""
import argparse,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent

def verify_pretraining_data(root):
    manifest=json.loads((HERE/'pretraining_data.json').read_text())
    # Dataset basename is part of the original deterministic mask RNG namespace.
    if root.name!=manifest['directory_name']:
        raise ValueError('Keep the data directory named bin-shard0-8 to retain the mask RNG keys')
    for name,record in manifest['files'].items():
        p=root/name
        if p.stat().st_size!=record['bytes']:raise ValueError(f'Wrong data file size: {name}')
        digest=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''):digest.update(chunk)
        if digest.hexdigest()!=record['sha256']:raise ValueError(f'Data checksum mismatch: {name}')
    print('Exact pretraining shard verified')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data_root',type=Path)
    verify_pretraining_data(p.parse_args().data_root.expanduser().resolve())
