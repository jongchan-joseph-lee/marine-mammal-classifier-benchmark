import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
d=pd.read_csv('/home/claude/pkg_in/data/bench_manifest.csv',low_memory=False).set_index('cut_id')
K=10
res=[]
for name in ['meta','mfcc16k','physv0','vggish','ast_tile','ast_zero']:
    z=np.load(f'/home/claude/pkg_in/full_cohort/features/{name}.npz',allow_pickle=True)
    ids=z['cut_id'].astype(str); X=np.nan_to_num(z['X'].astype(float))
    X=np.clip(StandardScaler().fit_transform(X),-8,8)
    m=d.loc[ids]; sp=np.asarray(m.species.values,dtype=object); site=np.asarray(m.g_site.values,dtype=object)
    nn=NearestNeighbors(n_neighbors=400).fit(X); dist,ix=nn.kneighbors(X)
    ix=ix[:,1:]
    allsp=(sp[ix[:,:K]]==sp[:,None]).mean()
    samesite=(site[ix[:,:K]]==site[:,None]).mean()
    oth=[]; 
    for i in range(len(ids)):
        row=ix[i][site[ix[i]]!=site[i]][:K]
        oth.append((sp[row]==sp[i]).mean() if len(row) else np.nan)
    oth=np.array(oth)
    # species with >=2 sources
    nsrc=m.groupby('species').g_site.nunique(); multi=m.species.map(nsrc).values>=2
    res.append(dict(rep=name,all=allsp,same_site=samesite,other=np.nanmean(oth),other_multi=np.nanmean(oth[multi]),other_single=np.nanmean(oth[~multi])))
r=pd.DataFrame(res); print(r.round(3).to_string())
m=d.loc[ids]; nsrc=m.groupby('species').g_site.nunique(); multi=m.species.map(nsrc)>=2
print('clips in single-source species', (~multi).sum(), (~multi).mean().round(3), 'species single', (nsrc==1).sum(), len(nsrc))
# ceiling: fraction of other-source pool that is same species, and chance
pool=[]; 
site=np.asarray(m.g_site.values,dtype=object); sp=np.asarray(m.species.values,dtype=object)
ss=pd.crosstab(m.species,m.g_site)
for i in range(len(m)):
    tot=(site!=site[i]).sum(); same=((sp==sp[i])&(site!=site[i])).sum()
    pool.append((same/tot, min(same,K)/K))
pool=np.array(pool); print('chance other', pool[:,0].mean().round(3), 'ceiling other (k=10)', pool[:,1].mean().round(3), 'ceiling multi', pool[multi.values,1].mean().round(3))
