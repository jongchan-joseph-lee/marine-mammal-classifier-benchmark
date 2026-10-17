from common import *
import json
from sklearn.metrics import roc_curve, roc_auc_score
OUT={}
meta=X['meta']
# ---------- C. meta / acoustic / both (original fold_all), D. no-PCA, E. phys ablation
pn=PHYS_NAMES; iB=[i for i,n in enumerate(pn) if n.startswith('rise3oct_')]
absf=set(['log_sig_f05','log_sig_f50','log_sig_f95'])|set(pn[i] for i in iB)
GROUPS={'phys51 (all)':list(range(51)),'phys − 1/3-oct (20)':[i for i in range(51) if i not in iB],
        'phys − abs. frequency (17)':[i for i,n in enumerate(pn) if n not in absf],
        'band only: A+B (38)':[i for i,n in enumerate(pn) if n in absf or n in ('sig_bw_oct','sig_bins_frac','rise_peak_db','tf_active_frac')]}
C=[]; Dn=[]; E=[]
for t in ['species','group3','type7']:
    m=task_mask(t); y=M[TASKS[t]].values[m].astype(str); f=M.fold_all.values[m]; site=M.g_site.values[m]
    pm=cv_predict(meta[m],y,f)
    for feat in FEATS[1:]:
        pa=cv_predict(X[feat][m],y,f); pb=cv_predict(np.c_[X[feat][m],meta[m]],y,f)
        ra=boot(y,pa,pm,site,500); rb=boot(y,pb,pm,site,500); rab=boot(y,pb,pa,site,500)
        C.append(dict(task=t,feat=feat,meta=round(ba(y,pm),3),acoustic=round(ra['ba'],3),a_d=round(ra['d'],3),a_lo=round(ra['dlo'],3),a_hi=round(ra['dhi'],3),
                      both=round(rb['ba'],3),b_d=round(rb['d'],3),b_lo=round(rb['dlo'],3),b_hi=round(rb['dhi'],3),both_vs_ac=round(rab['d'],3),bva_lo=round(rab['dlo'],3),bva_hi=round(rab['dhi'],3)))
        pn_=cv_predict(X[feat][m],y,f,pca=False)
        Dn.append(dict(task=t,feat=feat,pca=round(ba(y,pa),3),nopca=round(ba(y,pn_),3)))
        print(t,feat,'ac',round(ra['ba'],3),'both',round(rb['ba'],3),'meta',round(ba(y,pm),3),'nopca',round(ba(y,pn_),3),flush=True)
    for gname,cols in GROUPS.items():
        p=cv_predict(X['physv0'][m][:,cols],y,f); r=boot(y,p,pm,site,500)
        E.append(dict(task=t,set=gname,k=len(cols),**{k:round(float(v),3) for k,v in r.items()}))
        print(t,gname,round(r['ba'],3),flush=True)
OUT['C_meta_ac_both']=C; OUT['D_nopca']=Dn; OUT['E_phys_ablation']=E
# ---------- F. out-of-fold thresholds (source folds)
P=pd.DataFrame(X['physv0'],columns=pn,index=IDS)
t6=M.task_type_single.values; f_all=M.fold_all.values
def oof(feature, pos, neg, sign):
    sel=np.isin(t6,pos+neg)&~np.isnan(P[feature].values); v=P[feature].values[sel]*sign; y=np.isin(t6[sel],pos).astype(int); f=f_all[sel]
    pred=np.zeros(len(y),int); thr=[]
    for k in np.unique(f):
        tr,te=f!=k,f==k
        if len(np.unique(y[tr]))<2: pred[te]=0; continue
        fpr,tpr,th=roc_curve(y[tr],v[tr]); j=np.argmax(tpr-fpr); thr.append(th[j]); pred[te]=(v[te]>=th[j]).astype(int)
    fpr,tpr,th=roc_curve(y,v); j=np.argmax(tpr-fpr); pin=(v>=th[j]).astype(int)
    sens=lambda p:(p[y==1]==1).mean(); spec=lambda p:(p[y==0]==0).mean()
    return dict(feature=feature,pos='+'.join(pos),neg='+'.join(neg),n=int(len(y)),auroc=round(roc_auc_score(y,v),3),
                thr_in=round(float(th[j]*sign),4),thr_folds=[round(float(x*sign),4) for x in thr],
                ba_in=round(balanced_accuracy_score(y,pin),3),sens_in=round(sens(pin),3),spec_in=round(spec(pin),3),
                ba_oof=round(balanced_accuracy_score(y,pred),3),sens_oof=round(sens(pred),3),spec_oof=round(spec(pred),3))
F=[oof('tonal_conc',['click','pulse_train'],['whistle','moan','pulsed_call','growl'],-1),
   oof('tr_log_ici_med',['click'],['pulse_train'],1),
   oof('log_sig_f50',['whistle'],['moan'],1)]
for r in F: print(r)
OUT['F_thresholds']=F
json.dump(OUT,open('out_CDEF.json','w'),ensure_ascii=False)
