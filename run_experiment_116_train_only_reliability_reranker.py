from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from run_experiment_60_62_rocket_imaging_selector_variants import CALIBRATION_PROFILES, evaluate_indices, load_candidate_predictions, results_path, summary_path, top_score_order
from run_experiment_89_74d_with_exp84_candidate import EXP87_CONFIG, as_float, format_indices, parse_indices
from run_experiment_93_nonpos_candidate_reranker import (
    EXP90_OPERATIONAL_SELECTOR, EXP90_PATH, MAX_TOP, candidate_pool, exp84_order,
    rank_map, read_dict_rows, score_candidates, sorted_candidates, weak_base_indices,
)
from train_only_reliability import adaptive_weights, source_reliabilities

DATA_DIR = Path('/Users/minho/Documents/Dataset')
EXPERIMENT_ID = 'experiment_116_train_only_reliability_reranker'
EXP93_PATH = DATA_DIR / 'experiment_93_nonpos_candidate_reranker_results.csv'
EXP93_SELECTOR = 'nonpos_weak_alert_replace'
EXP87_PATH = DATA_DIR / 'experiment_87_exp84_index_diagnostics_results.csv'
WORKERS = int(__import__("os").environ.get("RANK_EXPERIMENT_WORKERS", "6"))


def write_csv(path, rows):
    keys=[]
    for row in rows:
        for key in row:
            if key not in keys: keys.append(key)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore'); w.writeheader(); w.writerows(rows)


def load_maps():
    base={r['dataset_name']:r for r in read_dict_rows(EXP93_PATH) if r.get('selector_name')==EXP93_SELECTOR}
    exp90={r['dataset_name']:r for r in read_dict_rows(EXP90_PATH) if r.get('selector_name')==EXP90_OPERATIONAL_SELECTOR}
    exp87={}
    for r in read_dict_rows(EXP87_PATH):
        if r.get('config_name')==EXP87_CONFIG: exp87[(r['dataset_name'],r['threshold_method'])]=r
    if len(base)!=1117 or set(base)!=set(exp90): raise SystemExit('baseline coverage mismatch')
    return base,exp90,exp87


def ordered(bundle): return top_score_order(bundle,MAX_TOP)


def row_metrics(name, record, y, bundles, base, indices, weights, reliabilities, reason):
    m=evaluate_indices(y,bundles['rocket_exp40']['test_scores'],indices)
    return {**base,'experiment_id':EXPERIMENT_ID,'dataset_name':name,'family':record['family'],
      'config_name':'train_only_reliability_weak_replace','selector_name':'train_only_reliability_weak_replace',
      'selector_reason':reason,'threshold_method':'selector','score_family':'train_only_reliability_reranker',
      'selected_indices':format_indices(indices),'predicted_count':m['predicted_count'],'tp':m['tp'],'fp':m['fp'],'fn':m['fn'],
      'auc_roc':m['auc_roc'],'auc_pr':m['auc_pr'],'f1':m['f1'],'oracle_f1':m['oracle_f1'],
      'train_normal_count':len(record['train_series']),'tiny_train':int(len(record['train_series'])<=10),
      'rocket_weight':weights['rocket'],'exp55_weight':weights['exp55'],'exp56_weight':weights['exp56'],
      'rocket_reliability':reliabilities['rocket'],'exp55_reliability':reliabilities['exp55'],'exp56_reliability':reliabilities['exp56']}


def run_one(args):
    name, base, exp90, exp87=args
    record,y,bundles=load_candidate_predictions(name,threshold_rates=CALIBRATION_PROFILES['relaxed_15pct'])
    base_indices=parse_indices(exp90.get('selected_indices'))
    rel,_=source_reliabilities(bundles); weights=adaptive_weights(rel)
    train_n=len(record['train_series']); tiny=train_n<=10
    orders={'rocket':ordered(bundles['rocket_exp40']),'exp55':ordered(bundles['exp55_best']),'exp56':ordered(bundles['exp56_best'])}
    rank_maps={k:rank_map(v) for k,v in orders.items()}
    pool=candidate_pool(*(v[:8] for v in orders.values()))
    scored=score_candidates(pool|base_indices,rank_maps,weights)
    ranked=sorted_candidates(scored)
    top=next((idx for idx in ranked if idx not in base_indices),None)
    weak=weak_base_indices(base_indices,scored)
    replacement=set(base_indices); changed=0
    info=scored.get(top,{}) if top is not None else {}
    gain=float(info.get('score',0))-max([scored.get(i,{}).get('score',0) for i in base_indices] or [0])
    if not tiny and len(base_indices)<=1 and weak and top is not None:
        if int(info.get('support',0))>=2 and int(info.get('best_rank',99))<=3 and gain>=0.04:
            replacement=(replacement-weak)|{top}; changed=len(weak)
    baseline=dict(base); baseline.update({'experiment_id':EXPERIMENT_ID,'config_name':'baseline_exp93','selector_name':'baseline_exp93','selector_reason':'control'})
    adaptive=row_metrics(name,record,y,bundles,base,replacement,weights,rel,'replace weak sparse alert using train-only source reliability')
    adaptive['rerank_replaced_count']=changed; adaptive['rerank_added_count']=int(changed and top not in base_indices)
    adaptive['top_candidate']=top if top is not None else ''; adaptive['top_candidate_support']=info.get('support',0); adaptive['top_candidate_score_gain']=gain
    return [baseline,adaptive]


def summarize(rows):
    out=[]
    for cfg in sorted({r['config_name'] for r in rows}):
        x=[r for r in rows if r['config_name']==cfg]; f=[as_float(r.get('f1')) for r in x]
        out.append({'experiment_id':EXPERIMENT_ID,'config_name':cfg,'selector_name':cfg,'threshold_method':'selector','num_datasets':len(x),'mean_f1':float(np.mean(f)),'median_f1':float(np.median(f)),'zero_f1_count':sum(v==0 for v in f),'mean_fp':float(np.mean([as_float(r.get('fp')) for r in x])),'mean_tp':float(np.mean([as_float(r.get('tp')) for r in x])),'mean_fn':float(np.mean([as_float(r.get('fn')) for r in x])),'mean_auc_pr':float(np.mean([as_float(r.get('auc_pr')) for r in x])),'mean_oracle_f1':float(np.mean([as_float(r.get('oracle_f1')) for r in x])),'rerank_used_datasets':sum(as_float(r.get('rerank_replaced_count'))>0 for r in x)})
    return sorted(out,key=lambda r:(r['mean_f1'],-r['mean_fp']),reverse=True)


def main(limit=None):
    base,exp90,exp87=load_maps(); names=sorted(base)[:limit] if limit else sorted(base); rows=[]; errors=[]
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futures={ex.submit(run_one,(n,base[n],exp90[n],exp87)):n for n in names}
        for done,future in enumerate(as_completed(futures),1):
            try: rows.extend(future.result())
            except Exception as exc: errors.append((futures[future],repr(exc))); print(f'ERROR dataset={futures[future]} error={exc!r}',flush=True)
            if done%25==0 or done==len(names): print(f'Progress: [{done:4d}/{len(names):4d}] rows={len(rows)} last={futures[future]} errors={len(errors)}',flush=True)
    if errors or len(rows)!=len(names)*2: raise SystemExit(f'coverage failure {len(rows)}/{len(names)*2} {errors[:5]}')
    write_csv(results_path(EXPERIMENT_ID),rows); write_csv(summary_path(EXPERIMENT_ID),summarize(rows)); print(f'{EXPERIMENT_ID} finished rows={len(rows)} datasets={len(names)}')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset-limit',type=int);a=p.parse_args();main(a.dataset_limit)
