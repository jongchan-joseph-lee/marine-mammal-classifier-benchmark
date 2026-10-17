from emb import *
c=pd.read_csv(PKG+'data/core_3539_metadata.csv',low_memory=False); fn=dict(zip(c.cut_id,c.file_name))
ids=c.cut_id.iloc[np.unique(np.linspace(0,len(c)-1,12).astype(int))].tolist()
ref={f:np.load(PKG+f'work/features/{f}.npz') for f in ['ast_tile','vggish','mfcc16k']}
ref={f:dict(zip(z['cut_id'].astype(str),z['X'])) for f,z in ref.items()}
ast=AST(); vg=VGG(); t=time.time()
cos=lambda a,b: float(np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b))
for cid in ids:
    x,fs=read(cid,fn[cid]); x16=prepare(x,fs,False)
    xm=x-x.mean(); m=mfcc_summary(resample_to(xm,fs,16000),16000,64,50,8000)
    a=ast(x16); v=vg(x16)
    print(cid,fs,f"ast {cos(a,ref['ast_tile'][cid]):.6f} vgg {cos(v,ref['vggish'][cid]):.6f} mfcc maxabs {np.abs(m-ref['mfcc16k'][cid]).max():.2e}")
print('sec/clip',(time.time()-t)/len(ids))
