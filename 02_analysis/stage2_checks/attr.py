import json,numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
R=json.load(open('phys2.json'))
d=pd.read_csv('/home/claude/pkg_in/data/bench_manifest.csv',low_memory=False); d['cut_id']=d.cut_id.astype(str); D=d.set_index('cut_id')
z=np.load('/home/claude/pkg_in/work/features/physv0.npz',allow_pickle=True); pn=[str(n) for n in z['names']]; P=pd.DataFrame(z['X'],index=z['cut_id'].astype(str),columns=pn)
ids=list(R); M=D.loc[ids]
A=pd.DataFrame([R[c]['attr'] for c in ids],index=ids)
T=pd.DataFrame(index=ids)
# ---- attribute schema (clip level, from frame-level measures)
T['imp_frac']=A.frac_impulsive                       # 충격성: 충격성 프레임 비율
T['imp_kurt']=P.loc[ids,'log_tr_kurt']              # 충격성: 1 ms 포락선 첨도 (log10)
T['rep_ici_ms']=10**P.loc[ids,'tr_log_ici_med']*1000 # 반복: 클릭 간격 (느린 반복)
T['rep_prr_hz']=A.prr_hz                             # 반복: 포락선 변조 피크 (빠른 반복 = 펄스 반복률)
T['rep_prr_prom']=A.prr_prom_db
T['harm']=A.harm                                     # 하모닉성: 스펙트럼 자기상관 피크
T['harm_spacing_hz']=A.harm_spacing                  # 하모닉 간격
T['fpk_over_spacing']=A.fpk_over_spacing             # 최대 주파수 / 하모닉 간격 (1≈기본음 존재, ≫1 = 반송파+측대역)
T['tonal_frac']=A.frac_tonal                         # 윤곽: 음조 프레임 비율
T['contour_cont']=A.contour_cont                     # 윤곽 연속성
T['fm_oct_s']=np.expm1(P.loc[ids,'log1p_fm_oct_per_s'])
T['f50_khz']=[10**R[c]['d1_f50']/1000 for c in ids]
T['band_obs_khz']=M.fs_hz.values/2000
T['env_crest']=A.env_crest
T['type']=M.task_type_single.fillna('').astype(str).values; T['multi']=M.task_type_multi_tags.fillna('').astype(str).values; T['tx']=M.group3.values; T['sp']=M.species_ko.values; T['site']=M.g_site.values; T['fs']=M.fs_hz.values
T.to_csv('attributes_3539.csv')
SIX=['whistle','moan','click','pulse_train','pulsed_call','growl']
cols=['imp_frac','imp_kurt','rep_ici_ms','rep_prr_hz','rep_prr_prom','harm','harm_spacing_hz','fpk_over_spacing','tonal_frac','contour_cont','fm_oct_s','f50_khz','env_crest']
O={}
O['median_by_type']={t:{c:(None if pd.isna(v) else round(float(v),3)) for c,v in T[T.type==t][cols].median().items()} for t in SIX}
def au(a,b,c):
    x=T[T.type.isin(a)][c].dropna(); y=T[T.type.isin(b)][c].dropna()
    if len(x)<10 or len(y)<10: return None
    s=roc_auc_score(np.r_[np.zeros(len(x)),np.ones(len(y))],np.r_[x,y]); return round(max(s,1-s),3)
pairs={'tonal vs pulsed':(['whistle','moan'],['pulsed_call','growl']),'whistle vs pulsed call':(['whistle'],['pulsed_call']),'moan vs growl':(['moan'],['growl']),
       'impulsive vs 나머지':(['whistle','moan','pulsed_call','growl'],['click','pulse_train']),'click vs pulse train':(['click'],['pulse_train']),'whistle vs moan':(['whistle'],['moan'])}
O['auroc']={p:{c:au(a,b,c) for c in cols} for p,(a,b) in pairs.items()}
# ---- multi-label validation
def has(s,keys): return isinstance(s,str) and any(k in s.split(';') for k in keys)
mt=T.multi
grp={'단일: tonal (whistle·moan)':T.type.isin(['whistle','moan']),'단일: impulsive (click·pulse train)':T.type.isin(['click','pulse_train']),
     '복수: tonal + impulsive 기재':(T.type=='')&mt.apply(lambda s:has(s,['whistle','moan']) and has(s,['click','pulse_train'])),
     '복수: tonal만 여러 개':(T.type=='')&mt.apply(lambda s:has(s,['whistle','moan']) and not has(s,['click','pulse_train','pulsed_call','growl']))}
V=[]
for g,mk in grp.items():
    s=T[mk]
    if len(s)==0: continue
    V.append(dict(group=g,n=int(len(s)),tonal_frac=round(float(s.tonal_frac.median()),3),imp_frac=round(float(s.imp_frac.median()),3),
                  both=round(float(((s.tonal_frac>=0.2)&(s.imp_frac>=0.2)).mean()),3),kurt=round(float(s.imp_kurt.median()),3)))
O['multi_validation']=V
print(pd.DataFrame(O['median_by_type']).T.to_string()); print(pd.DataFrame(O['auroc']).to_string()); print(pd.DataFrame(V).to_string())
print(T.type.isna().sum(), (T.type=='nan').sum())
json.dump(O,open('attr.json','w'),ensure_ascii=False)
