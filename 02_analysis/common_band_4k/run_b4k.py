from emb import *
c=pd.read_csv(PKG+'data/core_3539_metadata.csv',low_memory=False)
c=c[c.fs_hz>=8000].reset_index(drop=True); print('clips',len(c),flush=True)
out='b4k_cache.npz'; done={}
if os.path.exists(out):
    z=np.load(out,allow_pickle=True); done={k:z[k].item() for k in z.files}['d'] if 'd' in z.files else {}
ast=AST(); vg=VGG(); t0=time.time(); n=0
for cid,fn in zip(c.cut_id,c.file_name):
    if cid in done: continue
    x,fs=read(cid,fn); x16=prepare(x,fs,True)
    xm=x-x.mean(); m=mfcc_summary(resample_to(xm,fs,8000),8000,40,50,4000)
    done[cid]=dict(ast=ast(x16).astype(np.float32),vgg=vg(x16).astype(np.float32),mfcc=m)
    n+=1
    if n%100==0:
        np.savez(out,d=np.array(done,dtype=object)); print(len(done),f'{(time.time()-t0)/n:.2f}s/clip',flush=True)
np.savez(out,d=np.array(done,dtype=object)); 
ids=c.cut_id.tolist()
for k,name in [('ast','ast_tile_b4k'),('vgg','vggish_b4k'),('mfcc','mfcc_b4k')]:
    np.savez_compressed(f'{name}.npz',cut_id=np.array(ids),X=np.vstack([done[i][k] for i in ids]))
print('DONE',len(done),time.time()-t0,flush=True)
