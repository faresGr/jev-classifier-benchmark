"""Export saved public benchmark outputs, rescore Jev, and reproduce blog figures.

No API calls. Use --export-local once to package local runs; default reads archives.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from jev_benchmark.jev import parse_answer
from jev_benchmark.metrics import evaluate
from jev_benchmark.report import make_report

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {'ag_news': 'AG News', 'banking77': 'Banking77', 'emotion': 'Emotion'}
MODELS = {'logreg': 'Logistic regression', 'naive_bayes': 'Naive Bayes', 'linear_svm': 'Linear SVM',
          'xgboost_tfidf': 'XGBoost · TF-IDF', 'xgboost_svd': 'XGBoost · SVD'}

def f1(y, pred, k):
    matrix = np.bincount(y*k+pred, minlength=k*k).reshape(k,k)
    denom = matrix.sum(0)+matrix.sum(1)
    return np.divide(2*np.diag(matrix), denom, out=np.zeros(k), where=denom!=0).mean()

def paired_interval(reference, others, labels, samples=5000):
    """Stratified paired test bootstrap, conditional on already fitted models."""
    lookup = {label:i for i,label in enumerate(labels)}
    ids = [r['id'] for r in reference]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate test IDs')
    y = np.array([lookup[r['true_label']] for r in reference])
    p = np.array([lookup[r['predicted_label']] for r in reference])
    qs = []
    for records in others:
        by_id = {r['id']:r for r in records}
        if set(by_id) != set(ids) or any(by_id[r['id']]['true_label'] != r['true_label'] for r in reference):
            raise ValueError('Unpaired predictions')
        qs.append(np.array([lookup[by_id[i]['predicted_label']] for i in ids]))
    strata = [np.flatnonzero(y==i) for i in range(len(labels))]
    rng = np.random.default_rng(123)
    diffs = []
    for _ in range(samples):
        idx = np.concatenate([rng.choice(s,len(s),replace=True) for s in strata])
        diffs.append(f1(y[idx],p[idx],len(labels))-np.mean([f1(y[idx],q[idx],len(labels)) for q in qs]))
    return {'difference': float(f1(y,p,len(labels))-np.mean([f1(y,q,len(labels)) for q in qs])),
            'interval_95': np.quantile(diffs,[.025,.975]).tolist(), 'bootstrap_samples':samples,
            'bootstrap_seed':123,
            'scope':'Paired class-stratified test resampling; mean of three fixed fitted baseline seeds. Conditional on selected models; excludes model-selection and training uncertainty.'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-local',action='store_true')
    args=parser.parse_args()
    summary={}
    fig,axes=plt.subplots(3,1,figsize=(10,14))
    plt.rcParams.update({'font.size':11})
    for ax,(dataset,title) in zip(axes,DATASETS.items()):
        out=ROOT/'published-results'/dataset
        out.mkdir(parents=True,exist_ok=True)
        if args.export_local:
            source=ROOT/'runs'/f'{dataset}-jev'
            shutil.copyfile(source/'metadata.json',out/'metadata.json')
            shutil.copyfile(source/'results.json',out/'original-results.json')
            # Immutable gzip copy: full per-example outputs, no input text or credentials.
            with (source/'predictions.jsonl').open('rb') as src, (out/'predictions.jsonl.gz').open('wb') as dest:
                with gzip.GzipFile(fileobj=dest,mode='wb',mtime=0) as gz:
                    shutil.copyfileobj(src,gz)
        meta=json.loads((out/'metadata.json').read_text())
        results=json.loads((out/'original-results.json').read_text())
        labels=sorted(meta['labels'])
        groups=defaultdict(list)
        with gzip.open(out/'predictions.jsonl.gz','rt') as stream:
            for line in stream:
                record=json.loads(line)
                if record['run_id']=='jev' and record.get('raw_response'):
                    p,confidence,choice=parse_answer(record['raw_response'],labels)
                    record.update(probabilities=p,confidence=confidence,predicted_label=choice,error=None)
                groups[record['run_id']].append(record)
        jev=groups['jev']
        for run in results:
            if run['model']=='jev':
                run['metrics']=evaluate(jev,labels,1000)
        (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        make_report(out,results,meta)
        jr=next(r for r in results if r['model']=='jev')
        top=np.array([max(r['probabilities']) for r in jev])
        correct=np.array([r['true_label']==r['predicted_label'] for r in jev])
        classical=[r for r in results if r['model'] not in {'jev','dummy'}]
        max_n=max(r['n_train'] for r in classical)
        means={m:np.mean([r['metrics']['macro_f1_all'] for r in classical if r['model']==m and r['n_train']==max_n]) for m in MODELS}
        best=max(means,key=means.get)
        compared=[r for r in classical if r['model']==best and r['n_train']==max_n]
        summary[dataset]={'n':len(jev),'jev_macro_f1':jr['metrics']['macro_f1_all'],
            'ece_10_bins':jr['metrics']['probability_metrics_valid_only']['ece_10_bins'],
            'high_probability_count':int((top>=.9).sum()),'high_probability_correct':int(correct[top>=.9].sum()),
            'high_probability_errors':int((~correct[top>=.9]).sum()),
            'true_label_reported_zero':sum(r['raw_response']['answers']['category']['probabilities'][r['true_label']]==0 for r in jev),
            'training_labels_max_per_seed':max_n,'additional_validation_labels':meta['validation_label_count'],
            'best_observed_classical_at_max_budget':best,'classical_mean_macro_f1':float(means[best]),
            'paired_comparison':paired_interval(jev,[groups[r['run_id']] for r in compared],labels),
            'billing':jr['billing'], 'p50_ms':jr['metrics']['p50_ms']}
        for model,label in MODELS.items():
            by_n=defaultdict(list)
            for run in classical:
                if run['model']==model: by_n[run['train_per_class']].append(run['metrics']['macro_f1_all'])
            xs=sorted(by_n)
            ax.errorbar(xs,[np.mean(by_n[x]) for x in xs],yerr=[np.std(by_n[x]) for x in xs],marker='o',capsize=3,label=label)
        ax.axhline(jr['metrics']['macro_f1_all'],ls='--',color='black',label='Jev · original zero-shot prompt')
        xs=sorted(set(r['train_per_class'] for r in classical))
        ax.set(xscale='log',xticks=xs,ylim=(0,1),ylabel='Test macro-F1',xlabel='Training examples per category (log scale)',title=f'{title} · {len(labels)} categories · {len(jev)} test texts')
        ax.xaxis.set_major_formatter(ScalarFormatter()); ax.minorticks_off(); ax.grid(alpha=.2)
    fig.suptitle('How many labels is a description worth?',fontsize=21,y=.995)
    handles,labels_legend=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels_legend,loc='lower center',ncol=2,fontsize=11,bbox_to_anchor=(.5,.005))
    fig.tight_layout(rect=(0,.085,1,.985))
    fig.savefig(ROOT/'blog'/'jev-vs-classical-chart.png',dpi=170); plt.close(fig)
    (ROOT/'published-results'/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    # A separate 1200x630 preview remains legible in a social feed.
    fig,ax=plt.subplots(figsize=(12,6.3),dpi=100)
    fig.patch.set_facecolor('#f6f3ed'); ax.set_facecolor('#f6f3ed')
    x=np.arange(3); w=.32
    ax.bar(x-w/2,[summary[d]['jev_macro_f1'] for d in DATASETS],w,color='#157a78',label='Jev · zero-shot')
    ax.bar(x+w/2,[summary[d]['classical_mean_macro_f1'] for d in DATASETS],w,color='#304364',label='Best observed classical · max budget')
    for i,d in enumerate(DATASETS):
        for shift,key in [(-w/2,'jev_macro_f1'),(w/2,'classical_mean_macro_f1')]:
            v=summary[d][key]; ax.text(i+shift,v+.018,f'{v:.2f}',ha='center',fontsize=16)
    ax.set(xticks=x,xticklabels=['AG News\n4,000 train + 200 validation','Banking77\n1,540 train + 385 validation','Emotion\n3,000 train + 300 validation'],ylim=(0,1.08),ylabel='Test macro-F1')
    ax.tick_params(axis='x',labelsize=11); ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.14),ncol=2,frameon=False)
    fig.suptitle('Jev vs. classical ML',fontsize=27,fontweight='bold',y=.98)
    fig.text(.5,.885,'How many labels is a good description worth?',ha='center',fontsize=18)
    fig.text(.5,.025,'Fares Grina  ·  Three public datasets  ·  Classical scores: mean of 3 training seeds',ha='center',fontsize=11)
    fig.subplots_adjust(top=.74,bottom=.17,left=.09,right=.98)
    fig.savefig(ROOT/'blog'/'jev-linkedin-preview.png',dpi=100); plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
