import json,numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
R=json.load(open('phys2.json'))
d=pd.read_csv('/home/claude/pkg_in/data/bench_manifest.csv',low_memory=False); d['cut_id']=d.cut_id.astype(str); D=d.set_index('cut_id')
ids=list(R); M=D.loc[ids]
df=pd.DataFrame({k:[R[c][k] for c in ids] for k in ['d0_f50','d1_f50','d0_f05','d1_f05','d0_f95','d1_f95','d0_peak_db','d1_peak_db','d0_duty','d1_duty']},index=ids)
df['type']=M.task_type_single.values; df['tx']=M.group3.values; df['dur']=M.dur_s.values
O={}
O['undef']={t:int(df[f'{t}_f50'].isna().sum()) for t in ['d0','d1']}
O['undef_by_type']={t:{k:int(v) for k,v in df[df[f'{t}_f50'].isna()].type.fillna('(복수·미기재)').value_counts().items()} for t in ['d0','d1']}
# duty bins
df['dbin']=pd.cut(df.d1_duty,[0,.25,.5,.75,.9,1.01],labels=['<25%','25–50','50–75','75–90','≥90%'])
g=df.groupby('dbin',observed=True).agg(n=('d0_peak_db','size'),peak0=('d0_peak_db','median'),peak1=('d1_peak_db','median'),und0=('d0_f50',lambda s:s.isna().mean()),und1=('d1_f50',lambda s:s.isna().mean()))
O['by_duty']=g.round(3).reset_index().astype({'dbin':str}).to_dict('records'); print(g)
both=df.d0_f50.notna()&df.d1_f50.notna(); dd=np.abs(df.d1_f50-df.d0_f50)[both]*np.log2(10)
O['f50_shift_oct']={'median':round(float(dd.median()),3),'p90':round(float(dd.quantile(.9)),3),'frac_gt_half_oct':round(float((dd>0.5).mean()),3)}
O['peak_by_type']={t:{'d0':round(float(df[df.type==t].d0_peak_db.median()),1),'d1':round(float(df[df.type==t].d1_peak_db.median()),1)} for t in ['whistle','moan','click','pulse_train','pulsed_call','growl']}
def au(a,b,col):
    A=df[df.type.isin(a)][col].dropna(); B=df[df.type.isin(b)][col].dropna(); y=np.r_[np.zeros(len(A)),np.ones(len(B))]; s=roc_auc_score(y,np.r_[A,B]); return round(max(s,1-s),3)
O['auroc_f50_whistle_moan']={t:au(['whistle'],['moan'],f'{t}_f50') for t in ['d0','d1']}
# taxon by band (single-label six types)
six=df[df.type.isin(['whistle','moan','click','pulse_train','pulsed_call','growl'])]
for t in ['d0','d1']:
    f=10**six[f'{t}_f50']; band=pd.cut(f,[0,1000,8000,1e6],labels=['≤1k','1–8k','>8k'])
    O[f'band_tx_{t}']=pd.crosstab(band,six.tx).to_dict()
    O[f'band_n_{t}']={str(k):int(v) for k,v in band.value_counts().items()}
print(json.dumps(O,ensure_ascii=False,indent=0)[:2500])
json.dump(O,open('snr.json','w'),ensure_ascii=False,default=str)
