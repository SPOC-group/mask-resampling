"""Final BERT-Medium presets: nine pretraining and 144 selected GLUE runs."""
from pathlib import Path
import argparse,json,os,subprocess,sys
from prepare_source import HERE,verify_source

CONDITIONS=('K1','K10','Dynamic')
FT_SEEDS={0:(0,2),1:(0,1),2:(0,2)}
# task: learning rate, batch, training examples, number of classes
TASKS={'mnli':(1e-5,32,392702,3),'qnli':(1e-5,32,104743,2),
       'qqp':(1e-5,32,363846,2),'sst2':(1e-5,32,67349,2),
       'stsb':(2e-5,16,5749,1),'mrpc':(5e-5,16,3668,2),
       'cola':(5e-5,16,8551,2),'rte':(5e-5,16,2490,2)}
TOTAL_UPDATES=99953

def tasks(stage):
    return [dict(condition=c,pt_seed=s,**extra)
            for c in CONDITIONS for s in range(3)
            for extra in ([{}] if stage=='pretrain' else
                          [dict(ft_seed=f,task=t) for f in FT_SEEDS[s] for t in TASKS])]

def pretrain_dir(root,row):return root/'pretraining'/row['condition']/f"seed{row['pt_seed']}"
def checkpoint_name(row):return 'checkpoint_best.pt' if row['condition']=='K1' else 'checkpoint_last.pt'
def finetune_dir(root,row):return root/'finetuning'/row['condition']/f"pt_seed{row['pt_seed']}"/f"ft_seed{row['ft_seed']}"/row['task']

def pretrain_command(source,data,root,row):
    dynamic=row['condition']=='Dynamic';k={'K1':1,'K10':10,'Dynamic':0}[row['condition']]
    return [sys.executable,str(source/'fairseq_cli/train.py'),str(data),
        '--save-dir',str(pretrain_dir(root,row)),'--task','masked_lm','--criterion','masked_lm',
        '--arch','roberta_medium','--max-positions','512','--encoder-normalize-before',
        '--mask-prob','0.15','--dup-factor',str(k),'--mask-seed',str(row['pt_seed']),
        '--base-sequence-limit','3756032','--leave-unmasked-prob','0','--random-token-prob','0',
        '--sample-break-mode','none','--tokens-per-sample','128',
        '--optimizer','adam','--adam-betas','(0.9,0.98)','--adam-eps','1e-6','--clip-norm','0',
        '--lr-scheduler','cosine','--lr','0.002','--batch-size','128','--update-freq','8',
        '--distributed-world-size','4','--warmup-updates','1999','--max-update',str(TOTAL_UPDATES),
        '--dropout','0.1','--attention-dropout','0.1','--weight-decay','0.01',
        '--fp16','--fp16-init-scale','8','--fp16-scale-tolerance','0.1','--fp16-scale-window','99954',
        '--keep-interval-updates','100','--save-interval-updates','9995','--validate-interval-updates','9995',
        '--save-interval','99999999','--validate-interval','99999999','--num-workers','8',
        '--seed',str(row['pt_seed']),'--log-format','simple','--log-interval','1']+(['--dynamic-masking'] if dynamic else [])

def finetune_commands(source,data,root,row):
    task=row['task'];lr,batch,ntrain,nclasses=TASKS[task];updates=10*ntrain//batch
    out=finetune_dir(root,row);checkpoint=pretrain_dir(root,row)/checkpoint_name(row)
    valid='valid,valid-mm' if task=='mnli' else 'valid';dataset=data/task/'bin'
    train=[sys.executable,str(source/'fairseq_cli/train.py'),str(dataset),
        '--finetune-from-model',str(checkpoint),'--save-dir',str(out),'--seed',str(row['ft_seed']),
        '--arch','roberta_medium','--task','sentence_prediction','--criterion','sentence_prediction',
        '--max-positions','512','--batch-size',str(batch),'--required-batch-size-multiple','1',
        '--update-freq','1','--num-classes',str(nclasses),'--total-num-update',str(updates),'--max-epoch','10',
        '--dropout','0.1','--attention-dropout','0.1','--weight-decay','0.1','--optimizer','adam',
        '--adam-betas','(0.9, 0.98)','--adam-eps','1e-6','--clip-norm','0',
        '--lr-scheduler','polynomial_decay','--lr',str(lr),'--warmup-updates',str(updates*6//100),
        '--find-unused-parameters','--keep-last-epochs','1','--encoder-normalize-before',
        '--no-epoch-checkpoints','--log-format','simple','--log-interval','10','--fp16',
        '--valid-subset',valid]+(['--regression-target'] if task=='stsb' else [])
    evaluate=[sys.executable,str(source/'fairseq_cli/validate_glue.py'),str(dataset),
        '--path',str(out/'checkpoint_last.pt'),'--task','sentence_prediction','--criterion','sentence_prediction',
        '--batch-size',str(batch),'--required-batch-size-multiple','1','--num-classes',str(nclasses),
        '--results-path',str(out/f'{task}.json'),'--max-positions','512','--fp16','--valid-subset',valid]
    return [train,evaluate]

def checkpoint_summary(path):
    import torch
    from omegaconf import OmegaConf
    checkpoint=torch.load(str(path),map_location='cpu',weights_only=False,mmap=True)
    cfg=checkpoint['cfg']
    if OmegaConf.is_config(cfg):cfg=OmegaConf.to_container(cfg,resolve=False)
    hist=checkpoint['optimizer_history'];extra=checkpoint.get('extra_state',{})
    result={'num_updates':hist[-1]['num_updates'],'val_loss_bits':extra.get('val_loss'),
            'train_iterator':extra.get('train_iterator'), 'cfg':cfg}
    del checkpoint
    return result

def validate_pretraining(root,row):
    directory=pretrain_dir(root,row);last=checkpoint_summary(directory/'checkpoint_last.pt')
    cfg=last['cfg']
    expected={'common':{'seed':row['pt_seed'],'fp16':True,'fp16_init_scale':8,'fp16_scale_window':99954},
        'task':{'mask_prob':.15,'dup_factor':{'K1':1,'K10':10,'Dynamic':0}[row['condition']],
                'dynamic_masking':row['condition']=='Dynamic','mask_seed':row['pt_seed'],
                'leave_unmasked_prob':0.,'random_token_prob':0.,'tokens_per_sample':128,
                'base_sequence_limit':3756032,'sample_break_mode':'none'},
        'optimization':{'max_update':99953,'lr':[.002],'update_freq':[8]},
        'dataset':{'batch_size':128,'validate_interval_updates':9995},
        'distributed_training':{'distributed_world_size':4},
        'lr_scheduler':{'_name':'cosine','warmup_updates':1999},
        'optimizer':{'_name':'adam','adam_eps':1e-6,'weight_decay':.01}}
    for section,fields in expected.items():
        for key,value in fields.items():
            if cfg[section][key]!=value:raise ValueError(f'Pretraining mismatch: {section}.{key}')
    if last['num_updates']!=TOTAL_UPDATES:raise ValueError('Pretraining has not completed the full update budget')
    selected=checkpoint_summary(directory/checkpoint_name(row)) if row['condition']=='K1' else last
    return dict(condition=row['condition'],pt_seed=row['pt_seed'],final_num_updates=last['num_updates'],
                selected_num_updates=selected['num_updates'],checkpoint_used=checkpoint_name(row),
                val_loss_used_bits=selected['val_loss_bits'],val_loss_final_bits=last['val_loss_bits'])

def checked_entries(path,row,selected_step=None):
    entries=json.loads(path.read_text());task=row['task'];lr,batch,_,_=TASKS[task]
    expected=('valid','valid-mm') if task=='mnli' else ('valid',)
    if [e['subset'] for e in entries]!=list(expected):raise ValueError(f'Missing or duplicate evaluations: {path}')
    for e in entries:
        for key,value in {'seed':row['ft_seed'],'lr':lr,'effective_batch_size':batch,'max_epoch':10}.items():
            if e[key]!=value:raise ValueError(f'Final-recipe mismatch: {path}: {key}')
        if e['pretraining_step']!=(selected_step if selected_step is not None else (29985 if row['condition']=='K1' else 99953)):
            raise ValueError(f'Wrong pretraining checkpoint in {path}')
        if e['finetune_from_model_checkpoint']!=checkpoint_name(row):raise ValueError('Wrong pretraining selection policy')
        if e['finetune_steps']<=0:raise ValueError('Missing fine-tuning updates')
    return entries

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('pretrain','finetune'),required=True)
    p.add_argument('--list',action='store_true');p.add_argument('--task',type=int)
    p.add_argument('--source',type=Path,default=HERE/'vendor/dinkytrain')
    p.add_argument('--data-root',type=Path);p.add_argument('--output-root',type=Path)
    p.add_argument('--dry-run',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args();grid=tasks(a.stage)
    if a.list:print(json.dumps([dict(index=i,**r) for i,r in enumerate(grid)],indent=2));return
    index=a.task if a.task is not None else int(os.environ.get('SLURM_ARRAY_TASK_ID','-1'))
    if not 0<=index<len(grid):p.error(f'Choose task 0..{len(grid)-1}')
    if a.data_root is None or a.output_root is None:p.error('--data-root and --output-root are required')
    source=a.source.expanduser().resolve();root=a.output_root.expanduser().resolve();data=a.data_root.expanduser().resolve();row=grid[index]
    commands=[pretrain_command(source,data,root,row)] if a.stage=='pretrain' else finetune_commands(source,data,root,row)
    if a.dry_run:print(json.dumps(dict(task=row,commands=commands),indent=2));return
    verify_source(source)
    sys.path.insert(0,str(source))
    import torch
    required=4 if a.stage=='pretrain' else 1
    if torch.cuda.device_count()!=required:raise RuntimeError(f'Expose exactly {required} GPUs with CUDA_VISIBLE_DEVICES')
    if a.stage=='pretrain':
        from check_data import verify_pretraining_data
        verify_pretraining_data(data)
        out=pretrain_dir(root,row)
    else:
        if not (data/row['task']/'bin').is_dir():raise FileNotFoundError('Missing preprocessed GLUE task')
        selected=validate_pretraining(root,row)
        out=finetune_dir(root,row)
    record=dict(task=row,commands=commands);invocation=out/'invocation.json';done=out/'completed.json'
    if invocation.exists() and json.loads(invocation.read_text())!=record:raise ValueError('Output belongs to different parameters')
    if out.exists() and any(out.iterdir()) and not invocation.exists():raise ValueError('Output has no matching invocation receipt')
    if done.exists():
        if a.stage=='pretrain':validate_pretraining(root,row)
        else:checked_entries(out/f"{row['task']}.json",row,selected['selected_num_updates'])
        print(f'Already complete: {out}');return
    if out.exists() and any(out.iterdir()) and not a.resume:raise RuntimeError('Incomplete output exists; inspect before using --resume')
    out.mkdir(parents=True,exist_ok=True);invocation.write_text(json.dumps(record,indent=2)+'\n')
    env=dict(os.environ,PYTHONPATH=str(source),WANDB_MODE='disabled',WANDB_DISABLED='true',OMP_NUM_THREADS='8')
    for key in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','SLURM_PROCID','SLURM_NTASKS','SLURM_LOCALID'):
        env.pop(key,None)
    if a.stage=='finetune' and (out/f"{row['task']}.json").exists():
        # validate_glue appends, so never evaluate into an existing result file.
        checked_entries(out/f"{row['task']}.json",row,selected['selected_num_updates'])
    else:
        for command in commands:subprocess.run(command,check=True,cwd=source,env=env)
    if a.stage=='pretrain':summary=validate_pretraining(root,row)
    else:
        checked_entries(out/f"{row['task']}.json",row,selected['selected_num_updates']);summary=dict(row,selected_num_updates=selected['selected_num_updates'])
    done.write_text(json.dumps(summary,indent=2)+'\n');print(f'Completed {a.stage}: {row}')

if __name__=='__main__':main()
