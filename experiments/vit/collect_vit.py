"""Collect final-epoch ViT validation accuracies and reconstruction histories."""
from __future__ import annotations
import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from run_vit import PRESETS,tasks,model_directory,pretraining_settings,probe_profiles,probe_directory

def history(path,end):
    rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    if [r['epoch'] for r in rows]!=list(range(end+1)):
        raise ValueError(f'Incomplete epoch history: {path}')
    return rows

def write_csv(path,rows):
    if not rows:return
    columns=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)

def aggregate(rows):
    fields=('mode','rho','augmentation','recipe','pretraining_completed_epochs','probe_profile')
    groups={}
    for row in rows:groups.setdefault(tuple(row[k] for k in fields),[]).append(row)
    result=[]
    for key,group in groups.items():
        item=dict(zip(fields,key));item['n_initializations']=len(group)
        for metric in ('validation_top1','validation_top5'):
            values=[r[metric] for r in group];sd=statistics.stdev(values) if len(values)>1 else float('nan')
            item[metric+'_mean']=statistics.mean(values);item[metric+'_std']=sd;item[metric+'_sem']=sd/math.sqrt(len(values))
        result.append(item)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preset',choices=PRESETS,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--allow-partial',action='store_true')
    a=p.parse_args();root=a.output_root.expanduser().resolve();results=[];curves=[];missing=[]
    for i,row in enumerate(tasks(a.preset)):
        model=model_directory(root,row)
        try:
            cfg=pretraining_settings(row);pre=history(model/'pretraining/log.txt',cfg['end_epoch'])
            for h in pre:
                curves.append(dict(row,completed_pretraining_epochs=h['epoch']+1,
                                   train_reconstruction_mse=h['train_loss_mae'],
                                   online_classifier_train_ce=h.get('train_loss_ce'),learning_rate=h['train_lr']))
            for epoch,profile in probe_profiles(a.preset,row):
                end=89 if profile=='standard90' else 49
                probe=history(probe_directory(model,epoch,profile)/'log.txt',end)[-1]
                for key in ('test_acc1','test_acc5','test_loss','train_loss'):
                    if not math.isfinite(float(probe[key])):raise ValueError(f'Nonfinite probe metric: {key}')
                result=dict(row,pretraining_completed_epochs=epoch+1,probe_profile=profile,probe_completed_epochs=end+1,
                            validation_top1=float(probe['test_acc1'])/100.,validation_top5=float(probe['test_acc5'])/100.,
                            validation_cross_entropy=probe['test_loss'],probe_train_cross_entropy=probe['train_loss'])
                reconstruction=model/'reconstruction_validation'/f'epoch{epoch:04d}.json'
                if reconstruction.is_file():
                    saved=json.loads(reconstruction.read_text())
                    if saved['num_samples_evaluated']!=5000:raise ValueError('Incomplete reconstruction evaluation')
                    result['validation_reconstruction_mse']=saved['validation_reconstruction_mse']
                results.append(result)
        except (FileNotFoundError,ValueError,KeyError) as error:
            missing.append(dict(task=i,**row,reason=str(error)))
    if missing and not a.allow_partial:raise RuntimeError(f'{len(missing)} incomplete tasks; first: {missing[0]}')
    out=root/'summaries'/a.preset;out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'per_run_results.csv',results);write_csv(out/'aggregate_results.csv',aggregate(results))
    write_csv(out/'pretraining_history.csv',curves)
    (out/'coverage.json').write_text(json.dumps({'expected_tasks':len(tasks(a.preset)),'missing':missing,'probe_rows':len(results)},indent=2)+'\n')
    if a.preset=='unmasked' and not missing:
        chosen=max(results,key=lambda r:r['validation_top5'])
        (out/'validation_selection.json').write_text(json.dumps({'completed_pretraining_epochs':chosen['pretraining_completed_epochs'],
             'validation_top5':chosen['validation_top5'],'rule':'Highest final-probe validation top-5 among nine encoder checkpoints',
             'independent_test_evaluation':False},indent=2)+'\n')
    print(f'Collected {len(results)} probe rows; {len(missing)} incomplete tasks')

if __name__=='__main__':main()
