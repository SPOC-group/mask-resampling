"""Independent ViT-Base/ImageNet-100 reproduction tasks on one GPU each."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RHOS = (.125, .25, .375, .5, .625, .75, .825, .95)
UNMASKED_EPOCHS = (0, 1, 2, 5, 10, 15, 20, 21, 26)
PRESETS = ('main', 'augmentation', 'longer', 'probe_no_aug', 'unmasked')

def tasks(preset):
    if preset == 'main':
        return [dict(mode=m, rho=r, augmentation='no_geom', init_seed=s, recipe='main')
                for s in range(3) for m in ('static_per_image', 'dynamic') for r in RHOS]
    if preset == 'augmentation':
        return [dict(mode=m, rho=r, augmentation=a, init_seed=0, recipe='main')
                for a in ('no_geom','standard') for m in ('static_per_image','dynamic') for r in RHOS]
    if preset == 'longer':
        return [dict(mode='dynamic', rho=.75, augmentation='no_geom', init_seed=0, recipe=r)
                for r in ('umae200','mae641')]
    if preset == 'probe_no_aug':
        return [dict(mode=m, rho=.75, augmentation='no_geom', init_seed=0, recipe='main')
                for m in ('static_per_image','dynamic')]
    if preset == 'unmasked':
        return [dict(mode='none', rho=0., augmentation='no_geom', init_seed=0, recipe='unmasked')]
    raise ValueError(preset)

def model_directory(root, row):
    token=f"{row['rho']:g}".replace('.','p')
    return root/'models'/row['recipe']/row['augmentation']/row['mode']/f"seed{row['init_seed']}"/f'rho{token}'

def pretraining_settings(row):
    epochs,warmup,batch,accum,end,val_start = {
        'main': (100,10,128 if row['augmentation']=='standard' else 64,1 if row['augmentation']=='standard' else 2,99,0),
        'umae200': (200,40,128,8,199,9),
        'mae641': (1600,40,64,64,640,9),
        'unmasked': (100,10,64,2,26,20),
    }[row['recipe']]
    return dict(epochs=epochs,warmup_epochs=warmup,batch_size=batch,accum_iter=accum,end_epoch=end,val_start_epoch=val_start)

def probe_profiles(preset,row):
    epochs=UNMASKED_EPOCHS if preset=='unmasked' else (pretraining_settings(row)['end_epoch'],)
    profiles=('standard50','standard90') if preset=='longer' else ('no_geom50',) if preset=='probe_no_aug' else ('standard50',)
    return [(e,p) for e in epochs for p in profiles]

def probe_directory(model,epoch,profile):
    return model/'probes'/f'epoch{epoch:04d}'/profile

def launch_prefix(script):
    # A single distributed process preserves the original sampler.set_epoch behavior.
    return [sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=1',str(HERE/script)]

def pretraining_command(root,data,row,workers=10):
    cfg=pretraining_settings(row);model=model_directory(root,row)
    cmd=launch_prefix('main_pretrain.py')+[
        '--model','mae_vit_base_patch16','--norm_pix_loss','--init_checkpoint',str(root/'initializations'/f"seed{row['init_seed']}.pth"),
        '--seed','0','--static_mask_seed','0','--mask_mode',row['mode'],'--mask_ratio',str(row['rho']),
        '--reconstruction_loss_scope','all' if row['mode']=='none' else 'masked','--aug_mode',row['augmentation'],
        '--epochs',str(cfg['epochs']),'--warmup_epochs',str(cfg['warmup_epochs']),
        '--batch_size',str(cfg['batch_size']),'--accum_iter',str(cfg['accum_iter']),
        '--blr','0.00015','--min_lr','0','--weight_decay','0.05','--reg','none','--lamb','0',
        '--val-start-epoch',str(cfg['val_start_epoch']),'--val-interval','1' if row['recipe']=='unmasked' else '10',
        '--num_workers',str(workers),'--data_path',str(data),'--output_dir',str(model/'pretraining'),
        '--log_dir',str(model/'pretraining'),'--wandb_mode','disabled']
    if cfg['end_epoch']!=cfg['epochs']-1:
        cmd+=['--stop-after-epoch',str(cfg['end_epoch'])]
    if row['recipe']=='unmasked':
        cmd+=['--save-epochs',*map(str,UNMASKED_EPOCHS),'--keep-intermediate-checkpoints']
    return cmd

def probe_command(root,data,row,epoch,profile,workers=10):
    model=model_directory(root,row);out=probe_directory(model,epoch,profile)
    large=profile=='standard90'
    return launch_prefix('main_linprobe.py')+[
        '--model','vit_base_patch16','--cls_token','--nb_classes','100',
        '--finetune',str(model/'pretraining'/f'checkpoint-{epoch}.pth'),
        '--batch_size','512' if large else '256','--accum_iter','32' if large else '1',
        '--epochs','90' if large else '50','--warmup_epochs','10','--blr','0.1','--min_lr','0',
        '--weight_decay','0','--aug_mode','no_geom' if profile=='no_geom50' else 'standard',
        '--seed','0','--dist_eval','--num_workers',str(workers),'--data_path',str(data),
        '--output_dir',str(out),'--log_dir',str(out),'--wandb_mode','disabled']

def completed(path,end):
    log=path/'log.txt'
    if not log.is_file():return False
    rows=[json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    return ([r['epoch'] for r in rows]==list(range(end+1))
            and (path/f'checkpoint-{end}.pth').is_file())

def run_stage(command,path,end,env,resume):
    invocation=path/'invocation.json'
    record={'command':command}
    if invocation.exists() and json.loads(invocation.read_text())!=record:
        raise ValueError(f'Existing run has different parameters: {path}')
    if completed(path,end):
        print(f'Already complete: {path}',flush=True);return
    path.mkdir(parents=True,exist_ok=True)
    if (path/'log.txt').exists():
        if not resume:raise RuntimeError(f'Incomplete run at {path}; inspect it before passing --resume')
        checkpoints=sorted(path.glob('checkpoint-*.pth'),key=lambda p:int(p.stem.split('-')[-1]))
        if not checkpoints:raise RuntimeError('No checkpoint is available for resumption')
        epoch=int(checkpoints[-1].stem.split('-')[-1])
        rows=[json.loads(s) for s in (path/'log.txt').read_text().splitlines() if s.strip()]
        if not rows or rows[-1]['epoch']!=epoch:
            raise RuntimeError('Log extends beyond the newest checkpoint; use a fresh output root to avoid mixing trajectories')
        command=command+['--resume',str(checkpoints[-1])]
    invocation.write_text(json.dumps(record,indent=2)+'\n')
    subprocess.run(command,check=True,cwd=HERE,env=env)
    if not completed(path,end):raise RuntimeError(f'Incomplete history or final checkpoint: {path}')

def validate_dataset(root):
    import sitecustomize
    from torchvision.datasets import ImageFolder
    expected=(HERE/'imagenet100_classes.txt').read_text().splitlines()
    for split,count in [('train',126689),('val',5000)]:
        dataset=ImageFolder(root/split)
        if dataset.classes!=expected or len(dataset)!=count:
            raise ValueError(f'{split}: expected the provided 100 classes and {count} images')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preset',choices=PRESETS,required=True)
    p.add_argument('--list',action='store_true')
    p.add_argument('--task',type=int)
    p.add_argument('--data-root',type=Path)
    p.add_argument('--output-root',type=Path)
    p.add_argument('--stage',choices=('pretrain','probe','both'),default='both')
    p.add_argument('--num-workers',type=int,default=10)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();grid=tasks(a.preset)
    if a.list:
        print(json.dumps([dict(task=i,**r) for i,r in enumerate(grid)],indent=2));return
    index=a.task if a.task is not None else int(os.environ.get('SLURM_ARRAY_TASK_ID','-1'))
    if not 0<=index<len(grid):p.error(f'Choose a task from 0 to {len(grid)-1}')
    if a.data_root is None or a.output_root is None:p.error('--data-root and --output-root are required')
    data=a.data_root.expanduser().resolve();root=a.output_root.expanduser().resolve();row=grid[index];model=model_directory(root,row)
    command=pretraining_command(root,data,row,a.num_workers)
    probes=[probe_command(root,data,row,e,v,a.num_workers) for e,v in probe_profiles(a.preset,row)]
    if a.preset=='probe_no_aug' and a.stage=='pretrain':p.error('probe_no_aug reuses existing pretrained encoders')
    if a.dry_run:
        print(json.dumps({'task':row,'pretrain':command,'probes':probes},indent=2));return
    validate_dataset(data)
    env=dict(os.environ,PYTHONPATH=str(HERE)+os.pathsep+os.environ.get('PYTHONPATH',''))
    # This wrapper runs inside an allocation and starts its own single-process torchrun.
    for key in ('RANK','LOCAL_RANK','WORLD_SIZE','SLURM_PROCID'):
        env.pop(key,None)
    if a.stage in ('pretrain','both') and a.preset!='probe_no_aug':
        init=root/'initializations'/f"seed{row['init_seed']}.pth"
        if not init.is_file():raise FileNotFoundError('Create the shared initialization checkpoints first (see README)')
        run_stage(command,model/'pretraining',pretraining_settings(row)['end_epoch'],env,a.resume)
    if a.stage in ('probe','both'):
        for epoch,profile in probe_profiles(a.preset,row):
            if not (model/'pretraining'/f'checkpoint-{epoch}.pth').is_file():
                raise FileNotFoundError('Required pretrained encoder is missing; run its pretraining task first')
            run_stage(probe_command(root,data,row,epoch,profile,a.num_workers),probe_directory(model,epoch,profile),
                      89 if profile=='standard90' else 49,env,a.resume)

if __name__=='__main__':main()
