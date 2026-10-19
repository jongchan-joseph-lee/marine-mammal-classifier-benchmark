import time, torch, pandas as pd, numpy as np, os
from ast_core import *
m = load_model(); d = pd.read_csv('labels_3539.csv', dtype={'cut_id': str})
d = d.sort_values('dur')  # short first
Z = {}; t = time.time(); k = 0
if os.path.exists('z_frozen.npz'): Z = dict(np.load('z_frozen.npz'))
with torch.no_grad():
    for c in d.cut_id:
        if c in Z and os.path.exists(f'h10/{c}.npy'): continue
        f, tp = fbank(f'audio/{c}.wav'); h = embed(m, f, tp); h10 = run_layers(m, h, 0, 10)
        np.save(f'h10/{c}.npy', h10[0].numpy().astype(np.float16))
        Z[c] = pool(m, run_layers(m, h10, 10, 12))[0].numpy()
        k += 1
        if k % 200 == 0:
            np.savez('z_frozen.npz', **Z); print(k, len(Z), f'{time.time()-t:.0f}s', flush=True)
np.savez('z_frozen.npz', **Z); print('done', len(Z), f'{time.time()-t:.0f}s', flush=True)
