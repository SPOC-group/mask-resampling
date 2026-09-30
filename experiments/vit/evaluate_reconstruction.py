"""Full-validation normalized-patch MSE for the unmasked ViT checkpoints (CPU)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sitecustomize
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from main_pretrain import build_pretrain_transform
from reconstruction_core import evaluate, load_frozen_model, load_cached_result, validation_fingerprint, write_json
from run_vit import tasks, model_directory, UNMASKED_EPOCHS, validate_dataset

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--num-workers',type=int,default=0)
    p.add_argument('--epochs',type=int,nargs='+',default=list(UNMASKED_EPOCHS),help='Zero-based checkpoint indices')
    a=p.parse_args()
    if a.batch_size<1 or a.cpu_threads<1 or a.num_workers<0:p.error('Invalid batch, thread or worker count')
    if not set(a.epochs)<=set(UNMASKED_EPOCHS):p.error('Epoch must be one of the unmasked checkpoint indices')
    torch.set_num_threads(a.cpu_threads);torch.set_num_interop_threads(1);torch.manual_seed(0)
    data=a.data_root.expanduser().resolve();validate_dataset(data)
    model_dir=model_directory(a.output_root.expanduser().resolve(),tasks('unmasked')[0])
    args=json.loads((model_dir/'pretraining/args.json').read_text())
    for k,v in {'model':'mae_vit_base_patch16','mask_mode':'none','mask_ratio':0.,'reconstruction_loss_scope':'all','norm_pix_loss':True,'reg':'none','lamb':0.}.items():
        if args[k]!=v:raise ValueError(f'Incorrect unmasked configuration: {k}')
    transform=build_pretrain_transform(SimpleNamespace(input_size=224,aug_mode='no_geom'),is_train=False)
    dataset=ImageFolder(data/'val',transform=transform)
    fingerprint=validation_fingerprint(dataset)
    loader=DataLoader(dataset,batch_size=a.batch_size,shuffle=False,num_workers=a.num_workers,drop_last=False,
                      persistent_workers=a.num_workers>0)
    out=model_dir/'reconstruction_validation';out.mkdir(exist_ok=True)
    for epoch in a.epochs:
        checkpoint=model_dir/'pretraining'/f'checkpoint-{epoch}.pth'
        stat=checkpoint.stat()
        provenance={'pretraining_epoch':epoch,'num_validation_samples':len(dataset),'validation_manifest_sha256':fingerprint,
                    'checkpoint_size_bytes':stat.st_size,'checkpoint_mtime_ns':stat.st_mtime_ns,
                    'batch_size':a.batch_size,'cpu_threads':a.cpu_threads}
        path=out/f'epoch{epoch:04d}.json'
        if load_cached_result(path,provenance) is not None:continue
        model=load_frozen_model(checkpoint,epoch,args)
        mse,count,seconds=evaluate(model,loader,epoch,out/'status.json')
        write_json(path,dict(provenance,validation_reconstruction_mse=mse,num_samples_evaluated=count,elapsed_seconds=seconds))
        del model

if __name__=='__main__':main()
