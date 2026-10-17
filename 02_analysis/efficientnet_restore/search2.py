from effcommon import *
import pandas as pd, json
d=pd.read_csv('/home/claude/pkg_in/data/bench_manifest.csv',low_memory=False); d['cut_id']=d.cut_id.astype(str)
z=np.load('/home/claude/pkg_in/work/features/effb3.npz',allow_pickle=True); ids=z['cut_id'].astype(str); ST=dict(zip(ids,z['X'])); m=d.set_index('cut_id').loc[ids]
mid=m[(m.dur_s>5.02)&(m.dur_s<5.6)].sample(8,random_state=1).index.tolist()
short=m[m.dur_s<4.5].sample(10,random_state=0).index.tolist(); long=m[m.dur_s>11].sample(8,random_state=0).index.tolist()
out={}
for lib in ['scipy','torch']:
    for r in [70,80,90,100]:
        E=embed([to_img(spec_db(load40(c)[:200000],lib=lib),f'top{r}','gray',True) for c in mid])
        out[f'range {lib} top{r}']=np.mean([cos(E[i],ST[c]) for i,c in enumerate(mid)])
print(out,flush=True)
for mode in ['zero','tile','none']:
    cs=[]
    for c in short:
        x=load40(c); w=windows(x,mode=mode)[0]; E=embed([to_img(spec_db(w,lib='torch'),'top80','gray',True)]); cs.append(cos(E[0],ST[c]))
    out[f'short {mode}']=(np.mean(cs),np.min(cs)); print('short',mode,np.mean(cs),np.min(cs),flush=True)
for agg in ['first','mean','max']:
    cs=[]
    for c in long:
        x=load40(c); W=windows(x,agg='first' if agg=='first' else 'all'); E=embed([to_img(spec_db(w,lib='torch'),'top80','gray',True) for w in W])
        v=E[0] if agg=='first' else (E.mean(0) if agg=='mean' else E.max(0)); cs.append(cos(v,ST[c]))
    out[f'long {agg}']=(np.mean(cs),np.min(cs)); print('long',agg,np.mean(cs),np.min(cs),flush=True)
json.dump({k:(list(v) if isinstance(v,tuple) else v) for k,v in out.items()},open('search2.json','w'))
