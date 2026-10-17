"""Full-population unsupervised structure; labels are used only for diagnostics."""
import argparse
import json
import hashlib

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score
import umap

from analysis_common import OUT,SEED,FEATURES,FORMS,metadata,features,rr,nearest10,verify_neighbours,log,write_scope


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--features',default=','.join(FEATURES))
    ap.add_argument('--wait',action='store_true')
    args=ap.parse_args();names=args.features.split(',')
    m=metadata();scope=write_scope();verify_neighbours()
    out=OUT/'stage1';out.mkdir(exist_ok=True)
    rows=[]
    form_mask=m.type_single.isin(FORMS).to_numpy()
    for f in names:
        target=out/(f+'_structure.npz')
        if target.exists():
            log(f'Stage1 {f}: verified saved structure reused.')
            continue
        X,_,infos=features([f],wait=args.wait);raw=X[f]
        Z,_=rr.transform(raw)
        nn=nearest10(Z)
        labels=KMeans(n_clusters=8,n_init=10,random_state=SEED).fit_predict(Z)
        coords=umap.UMAP(n_components=3,n_neighbors=15,min_dist=.1,n_epochs=300,
                         random_state=SEED,transform_seed=SEED,n_jobs=1,metric='euclidean').fit_transform(Z)
        same_source=np.mean(m.g_site.to_numpy()[:,None]==m.g_site.to_numpy()[nn],axis=1)
        same_fs=np.mean(m.fs_family.to_numpy()[:,None]==m.fs_family.to_numpy()[nn],axis=1)
        source_macro=np.mean([same_source[m.g_site.eq(s)].mean() for s in sorted(m.g_site.unique())])
        result=dict(feature=f,n=len(m),sources=m.g_site.nunique(),shape_dimensions=raw.shape[1],
                    same_source_NN10=float(same_source.mean()),same_source_NN10_source_macro=float(source_macro),
                    same_fs_NN10=float(same_fs.mean()),cluster_source_AMI=float(adjusted_mutual_info_score(m.g_site,labels)),
                    cluster_fs_AMI=float(adjusted_mutual_info_score(m.fs_family,labels)),
                    cluster_taxon_AMI=float(adjusted_mutual_info_score(m.group3,labels)),
                    cluster_form_AMI=float(adjusted_mutual_info_score(m.type_single[form_mask],labels[form_mask])),
                    labeled_seven_form_n=int(form_mask.sum()),fully_missing_rows=int(np.isnan(raw).all(1).sum()))
        rows.append(result)
        np.savez_compressed(target,cut_id=m.cut_id.to_numpy(dtype=str),coords=coords,cluster8=labels,NN10=nn,Z=Z.astype(np.float32))
        (out/(f+'_summary.json')).write_text(json.dumps(result,indent=2))
        support=[dict(feature=f,cut_id=c,fully_missing=bool(b),missing_dimensions=int(v))
                 for c,b,v in zip(m.cut_id,np.isnan(raw).all(1),np.isnan(raw).sum(1))]
        pd.DataFrame(support).to_csv(out/(f+'_feature_support.csv'),index=False)
        log(f'Stage1 {f}: full n={len(m)}; source NN10={result["same_source_NN10"]:.3f}; form AMI={result["cluster_form_AMI"]:.3f}.')
    summaries=[json.loads(p.read_text()) for p in out.glob('*_summary.json')]
    pd.DataFrame(summaries).to_csv(out/'structure_summary.csv',index=False)
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
