from effcommon import *
import pandas as pd, itertools, json
d=pd.read_csv('/home/claude/pkg_in/data/bench_manifest.csv',low_memory=False); d['cut_id']=d.cut_id.astype(str)
z=np.load('/home/claude/pkg_in/work/features/effb3.npz',allow_pickle=True); ids=z['cut_id'].astype(str); ST=dict(zip(ids,z['X']))
m=d.set_index('cut_id').loc[ids]
sel=m[(m.dur_s>5.02)&(m.dur_s<5.6)].sample(12,random_state=0).index.tolist()
X={c:load40(c)[:200000] for c in sel}
res=[]
for lib in ['scipy','torch']:
    S={c:spec_db(X[c],lib=lib) for c in sel}
    for scale,cmap,flip,norm in itertools.product(['minmax','top80','top60','pct'],['gray','viridis','magma','jet'],[True,False],['raw','imagenet']):
        E=embed([to_img(S[c],scale,cmap,flip) for c in sel],norm)
        cs=[cos(E[i],ST[c]) for i,c in enumerate(sel)]
        res.append(dict(lib=lib,scale=scale,cmap=cmap,flip=flip,norm=norm,mean=np.mean(cs),min=np.min(cs)))
    print(lib,'done',flush=True)
r=pd.DataFrame(res).sort_values('mean',ascending=False); print(r.head(15).to_string()); r.to_csv('search1.csv',index=False)
