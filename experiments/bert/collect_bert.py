"""Collect only the final-recipe BERT runs and reproduce the nested table averages."""
import argparse,csv,json,math,statistics
from pathlib import Path
from run_bert import tasks,finetune_dir,checked_entries,TASKS,CONDITIONS,FT_SEEDS
COLUMNS={'mnli':(('MNLI-m','valid','accuracy'),('MNLI-mm','valid-mm','accuracy')),
         'qnli':(('QNLI','valid','accuracy'),),'qqp':(('QQP','valid','accuracy'),),
         'sst2':(('SST-2','valid','accuracy'),),'stsb':(('STS-B','valid','spearmanr'),),
         'mrpc':(('MRPC','valid','matthews_correlation'),),
         'cola':(('CoLA','valid','matthews_correlation'),),'rte':(('RTE','valid','matthews_correlation'),)}
COLUMNS_ORDER=('MNLI-m','MNLI-mm','QNLI','QQP','SST-2','STS-B','MRPC','CoLA','RTE')
AVG_COLUMNS=tuple(c for c in COLUMNS_ORDER if c!='MNLI-mm')

def write_csv(path,rows):
    if not rows:raise ValueError('No complete data to write')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def rows_from_runs(root):
    rows=[]
    for row in tasks('finetune'):
        directory=finetune_dir(root,row)
        complete=json.loads((directory/'completed.json').read_text())
        if any(complete.get(k)!=v for k,v in row.items()):raise ValueError(f'Wrong completed-run receipt: {directory}')
        step=int(complete['selected_num_updates'])
        entries=checked_entries(directory/f"{row['task']}.json",row,step)
        for column,subset,key in COLUMNS[row['task']]:
            e=next(e for e in entries if e['subset']==subset)
            rows.append(dict(condition=row['condition'],pt_seed=row['pt_seed'],ft_seed=row['ft_seed'],task=row['task'],
                column=column,dev_subset=subset,metric=key,value=e[key],num_samples=e['num_samples'],
                lr=e['lr'],batch_size=e['effective_batch_size'],epochs=e['max_epoch'],finetune_steps=e['finetune_steps'],
                pretraining_step=step,checkpoint_used=e['finetune_from_model_checkpoint']))
    return rows

def summarize(rows):
    expected={(r['condition'],r['pt_seed'],r['ft_seed'],c) for r in tasks('finetune') for c,_,_ in COLUMNS[r['task']]}
    records={}
    for r in rows:
        key=(r['condition'],int(r['pt_seed']),int(r['ft_seed']),r['column'])
        if key in records:raise ValueError('Duplicate table entry')
        value=float(r['value'])
        if not math.isfinite(value):raise ValueError('Nonfinite reported metric')
        task=r['task'];lr,batch,_,_=TASKS[task]
        if float(r['lr'])!=lr or int(r['batch_size'])!=batch or int(r['epochs'])!=10:
            raise ValueError('A run does not use the final fine-tuning recipe')
        if (r['column'],r['dev_subset'],r['metric']) not in COLUMNS[task]:raise ValueError('Wrong metric or split')
        if not -1<=value<=1:raise ValueError('Metrics must be on the original fraction scale')
        records[key]=value
    if set(records)!=expected:raise ValueError(f'Expected the exact 162 entries from 144 selected runs; got {len(records)}')
    per_seed=[];summary=[]
    for cond in CONDITIONS:
        values={}
        for seed in range(3):
            scores={c:statistics.mean(records[(cond,seed,f,c)] for f in FT_SEEDS[seed]) for c in COLUMNS_ORDER}
            scores['AVG']=statistics.mean(scores[c] for c in AVG_COLUMNS)
            per_seed.extend(dict(condition=cond,pt_seed=seed,column=c,score_100=100*v) for c,v in scores.items())
            values[seed]=scores
        for c in (*COLUMNS_ORDER,'AVG'):
            v=[100*values[s][c] for s in range(3)];sd=statistics.stdev(v)
            summary.append(dict(condition=cond,column=c,mean_100=statistics.mean(v),sd_100=sd,sem_100=sd/math.sqrt(3),n_pretraining_seeds=3,n_finetuning_seeds_per_pretraining=2))
    return per_seed,summary

def table_markdown(summary):
    values={(r['condition'],r['column']):r for r in summary}
    lines=['| Task/column | K=1 | K=10 | Dynamic |','|---|---:|---:|---:|']
    for c in (*COLUMNS_ORDER,'AVG'):
        cells=[f"{values[(condition,c)]['mean_100']:.2f} ± {values[(condition,c)]['sd_100']:.2f}" for condition in CONDITIONS]
        lines.append('| '+c+' | '+' | '.join(cells)+' |')
    return '\n'.join(lines)+'\n'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--runs-root',type=Path);source.add_argument('--input-csv',type=Path)
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    rows=rows_from_runs(a.runs_root) if a.runs_root else list(csv.DictReader(a.input_csv.open()))
    seeds,summary=summarize(rows);out=a.output_dir
    write_csv(out/'glue_runs.csv',rows);write_csv(out/'per_pretraining_seed.csv',seeds);write_csv(out/'summary.csv',summary)
    (out/'tables.md').write_text(table_markdown(summary))
    print('Collected exactly 144 runs; 162 scores; averages and sample SDs computed over three pretraining seeds')

if __name__=='__main__':main()
