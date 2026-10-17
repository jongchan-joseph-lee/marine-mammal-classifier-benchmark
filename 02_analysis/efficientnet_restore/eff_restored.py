"""EfficientNet-B3 clip embedding, restored preprocessing (cosine ≥0.99 vs stored vectors on test clips).
40 kHz resample → torch.stft(n_fft=1024, hop=256, hann, center) → 10·log10(power) → clip to [max−80 dB, max]
→ 0..1 → grayscale ×3 → low frequency at bottom → PNG-like uint8 → resize 800×300 (bilinear) → /255 (no ImageNet norm)
→ torchvision efficientnet_b3 (IMAGENET1K_V1) features → global average → 1,536.
Clips < 5 s are tiled to 5 s; longer clips are cut into 5 s windows (last window aligned to the end) and averaged."""
from effcommon import *
import json, os, sys
z=np.load('/home/claude/pkg_in/work/features/effb3.npz',allow_pickle=True); ids=z['cut_id'].astype(str); ST=z['X']
out='eff_restored.npy'; done=np.load(out) if os.path.exists(out) else np.full((len(ids),1536),np.nan,np.float32)
for i,c in enumerate(ids):
    if not np.isnan(done[i,0]): continue
    x=load40(c); W=windows(x,mode='tile',agg='all')
    E=embed([to_img(spec_db(w,lib='torch'),'top80','gray',True) for w in W]); done[i]=E.mean(0)
    if i%100==0: np.save(out,done); print(i,flush=True)
np.save(out,done)
cs=np.array([cos(done[i],ST[i]) for i in range(len(ids))]); np.save('eff_cos.npy',cs)
print('cos mean',cs.mean(),'median',np.median(cs),'p5',np.percentile(cs,5),'min',cs.min(),'n<0.95',(cs<0.95).sum())
