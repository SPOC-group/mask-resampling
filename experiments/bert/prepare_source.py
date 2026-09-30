"""Prepare a pinned public DinkyTrain checkout with the recorded experiment changes."""
from pathlib import Path, PurePosixPath
import argparse,hashlib,json,shutil,tarfile,tempfile,urllib.request
HERE=Path(__file__).resolve().parent

def verify_source(root):
    manifest=json.loads((HERE/'source_manifest.json').read_text())
    recorded=json.loads((root/'supplement_source.json').read_text())
    if recorded!=manifest:raise ValueError('Source receipt differs from this package')
    for rel,digest in manifest['overlay_sha256'].items():
        if hashlib.sha256((root/rel).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Recorded experiment module was modified: {rel}')
    return manifest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination',type=Path,default=HERE/'vendor/dinkytrain')
    p.add_argument('--archive',type=Path,help='Previously downloaded archive; avoids network access')
    a=p.parse_args();dest=a.destination.expanduser().resolve();m=json.loads((HERE/'source_manifest.json').read_text())
    if dest.exists():
        verify_source(dest);print(f'Existing source verified: {dest}');return
    dest.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='bert-source-',dir=dest.parent) as temporary:
        tmp=Path(temporary);archive=a.archive
        if archive is None:
            archive=tmp/'upstream.tar.gz'
            with urllib.request.urlopen(m['archive_url'],timeout=60) as response,archive.open('wb') as output:
                shutil.copyfileobj(response,output)
        if hashlib.sha256(archive.read_bytes()).hexdigest()!=m['archive_sha256']:
            raise ValueError('Upstream archive SHA-256 does not match the verified release')
        source=tmp/'source';source.mkdir()
        prefix='DinkyTrain-'+m['commit']
        with tarfile.open(archive) as tar:
            for item in tar:
                path=PurePosixPath(item.name)
                if path.is_absolute() or '..' in path.parts or path.parts[0]!=prefix:
                    raise ValueError('Unsafe upstream archive path')
                relative=Path(*path.parts[1:])
                if not relative.parts:continue
                # These upstream links belong to unrelated examples; no BERT dependency uses them.
                if item.issym() or item.islnk():continue
                target=source/relative
                if item.isdir():target.mkdir(parents=True,exist_ok=True)
                elif item.isfile():
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with tar.extractfile(item) as inp,target.open('wb') as out:shutil.copyfileobj(inp,out)
                else:raise ValueError('Unsupported archive member')
        for rel,digest in m['overlay_sha256'].items():
            content=(HERE/'overlay'/rel).read_bytes()
            if hashlib.sha256(content).hexdigest()!=digest:raise ValueError(f'Overlay checksum mismatch: {rel}')
            (source/rel).parent.mkdir(parents=True,exist_ok=True);(source/rel).write_bytes(content)
        (source/'supplement_source.json').write_text(json.dumps(m,indent=2)+'\n')
        source.rename(dest)
    verify_source(dest);print(f'Prepared {dest}; install it using the README commands')

if __name__=='__main__':main()
