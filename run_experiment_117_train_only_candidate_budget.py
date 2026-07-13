from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from run_experiment_60_62_rocket_imaging_selector_variants import CALIBRATION_PROFILES, cap_indices_count, evaluate_indices, load_candidate_predictions, results_path, summary_path, top_score_order
from run_experiment_89_74d_with_exp84_candidate import as_float, format_indices, parse_indices
from run_experiment_93_nonpos_candidate_reranker import candidate_pool, rank_map, score_candidates, sorted_candidates
from train_only_reliability import adaptive_weights, source_reliabilities

DATA_DIR=Path('/Users/minho/Documents/Dataset')
EXPERIMENT_ID='experiment_117_train_only_candidate_budget'
EXP93_PATH=DATA_DIR/'experiment_93_nonpos_candidate_reranker_results.csv'
EXP93_SELECTOR='nonpos_weak_alert_replace'
WORKERS=int(__import__("os").environ.get("RANK_EXPERIMENT_WORKERS", "6"))

def write_csv(path,rows):
 keys=[]
 for r in rows:
  for k in r:
   if k not in keys: keys.append(k)
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)

def load_maps():
 exp93={r['dataset_name']:r for r in csv.DictReader(EXP93_PATH.open()) if r.get('selector_name')==EXP93_SELECTOR}
 if len(exp93)!=1117: raise SystemExit(f'Exp93 baseline coverage mismatch: {len(exp93)}')
 return exp93

def order(bundle,n=8): return top_score_order(bundle,n)

def make_row(name,record,y,bundles,base,cfg,indices,reason,rel,weights,budget):
 m=evaluate_indices(y,bundles['rocket_exp40']['test_scores'],indices)
 return {**base,'experiment_id':EXPERIMENT_ID,'dataset_name':name,'family':record['family'],'config_name':cfg,'selector_name':cfg,'selector_reason':reason,'threshold_method':'selector','score_family':'train_only_candidate_budget','selected_indices':format_indices(indices),'predicted_count':m['predicted_count'],'tp':m['tp'],'fp':m['fp'],'fn':m['fn'],'auc_roc':m['auc_roc'],'auc_pr':m['auc_pr'],'f1':m['f1'],'oracle_f1':m['oracle_f1'],'train_normal_count':len(record['train_series']),'tiny_train':int(len(record['train_series'])<=10),'alert_budget':budget,'rocket_reliability':rel['rocket'],'exp55_reliability':rel['exp55'],'exp56_reliability':rel['exp56'],'rocket_weight':weights['rocket'],'exp55_weight':weights['exp55'],'exp56_weight':weights['exp56'],'repair_added_count':len(indices-parse_indices(base.get('selected_indices')))}

def run_one(args):
 name,exp93=args
 record,y,bundles=load_candidate_predictions(name,threshold_rates=CALIBRATION_PROFILES['relaxed_15pct'])
 # Every variant starts from the current operating baseline.  The previous
 # version started candidate rows from Exp89 while displaying Exp93 as the
 # control, which made the reported deltas incomparable.
 base=parse_indices(exp93.get('selected_indices')); tiny=len(record['train_series'])<=10
 rel,_=source_reliabilities(bundles); weights=adaptive_weights(rel)
 orders={'rocket':order(bundles['rocket_exp40']),'exp55':order(bundles['exp55_best']),'exp56':order(bundles['exp56_best'])}
 ranks={k:rank_map(v) for k,v in orders.items()}; scored=score_candidates(candidate_pool(*orders.values()),ranks,weights); ranked=sorted_candidates(scored)
 top_by_reliability=orders[max(rel,key=rel.get)][0] if not tiny else None
 consensus=[i for i in ranked if scored[i]['support']>=2]
 top_consensus=consensus[0] if consensus else None
 baseline=dict(exp93);baseline.update({'experiment_id':EXPERIMENT_ID,'config_name':'baseline_exp93','selector_name':'baseline_exp93','selector_reason':'control','alert_budget':0,'repair_added_count':0})
 rows=[baseline]
 idx=set(base)
 if not idx and not tiny and top_by_reliability is not None: idx.add(top_by_reliability)
 rows.append(make_row(name,record,y,bundles,exp93,'reliability_top1_noalert_budget1',idx,'no-alert: top candidate from most train-stable source',rel,weights,1))
 idx=set(base)
 if not idx and not tiny and top_consensus is not None: idx.add(top_consensus)
 rows.append(make_row(name,record,y,bundles,exp93,'reliability_consensus_noalert_budget1',idx,'no-alert: train-weighted candidate with at least two source votes',rel,weights,1))
 idx=set(base); budget=1
 if not tiny and len(base)<=1 and len(record['train_series'])>=100 and np.mean(list(rel.values()))>=0.35:
  additions=[i for i in consensus if i not in idx][:2]; idx.update(additions); budget=2
 rows.append(make_row(name,record,y,bundles,exp93,'reliability_consensus_sparse_budget2',idx,'sparse alert: up to two train-stable consensus candidates only for >=100 normals',rel,weights,budget))
 return rows

def summarize(rows):
 out=[]
 for cfg in sorted({r['config_name'] for r in rows}):
  x=[r for r in rows if r['config_name']==cfg];f=[as_float(r.get('f1')) for r in x]
  out.append({'experiment_id':EXPERIMENT_ID,'config_name':cfg,'selector_name':cfg,'threshold_method':'selector','num_datasets':len(x),'mean_f1':float(np.mean(f)),'median_f1':float(np.median(f)),'zero_f1_count':sum(v==0 for v in f),'mean_fp':float(np.mean([as_float(r.get('fp')) for r in x])),'mean_tp':float(np.mean([as_float(r.get('tp')) for r in x])),'mean_fn':float(np.mean([as_float(r.get('fn')) for r in x])),'mean_auc_pr':float(np.mean([as_float(r.get('auc_pr')) for r in x])),'mean_oracle_f1':float(np.mean([as_float(r.get('oracle_f1')) for r in x])),'repair_used_datasets':sum(as_float(r.get('repair_added_count'))>0 for r in x)})
 return sorted(out,key=lambda r:(r['mean_f1'],-r['mean_fp']),reverse=True)

def main(limit=None):
 exp93=load_maps();names=sorted(exp93)[:limit] if limit else sorted(exp93);rows=[];errors=[]
 with ProcessPoolExecutor(max_workers=WORKERS) as ex:
  fs={ex.submit(run_one,(n,exp93[n])):n for n in names}
  for done,f in enumerate(as_completed(fs),1):
   try: rows.extend(f.result())
   except Exception as e: errors.append((fs[f],repr(e)));print(f'ERROR dataset={fs[f]} error={e!r}',flush=True)
   if done%25==0 or done==len(names): print(f'Progress: [{done:4d}/{len(names):4d}] rows={len(rows)} last={fs[f]} errors={len(errors)}',flush=True)
 if errors or len(rows)!=len(names)*4: raise SystemExit(f'coverage failure {len(rows)}/{len(names)*4} {errors[:5]}')
 write_csv(results_path(EXPERIMENT_ID),rows);write_csv(summary_path(EXPERIMENT_ID),summarize(rows));print(f'{EXPERIMENT_ID} finished rows={len(rows)} datasets={len(names)}')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset-limit',type=int);a=p.parse_args();main(a.dataset_limit)
