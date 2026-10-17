from common import *
import pandas as pd
r=pd.read_csv('/home/claude/ref_v1/eval/reference_v1/results.csv'); r=r[(r.subset=='all')&(r.metric=='balanced_acc')&(r.protocol=='allcv')]
print(M.task_group3.value_counts(dropna=False).head(), M.task_species.notna().sum())
for t in ['species','group3','type7']:
    m=task_mask(t); y=M[TASKS[t]].values[m].astype(str); f=M.fold_all.values[m]
    for feat in ['meta','physv0','ast_tile']:
        p=cv_predict(X[feat][m],y,f); ref=r[(r.task==('type_single' if t=='type7' else t))&(r.feature==feat)].value
        print(t,m.sum(),len(set(y)),feat,round(ba(y,p),3),'ref',ref.round(3).tolist())
