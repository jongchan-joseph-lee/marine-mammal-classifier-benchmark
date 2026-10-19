import copy, json, sys, time, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from ast_core import load_model
torch.set_num_threads(2); torch.manual_seed(0); np.random.seed(0)
d = pd.read_csv('labels_3539.csv', dtype={'cut_id': str})
CL = ['Mysticeti', 'Odontoceti', 'Pinnipedia']; y = d.tx.map(CL.index).to_numpy()
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(len(d)), y, d.site))
H = {c: None for c in d.cut_id}
def h10(c):
    return torch.from_numpy(np.load(f'h10/{c}.npy').astype(np.float32))
base = load_model()
class Head(nn.Module):
    def __init__(s):
        super().__init__()
        s.blocks = copy.deepcopy(base.audio_spectrogram_transformer.layers[10:12])
        s.ln = copy.deepcopy(base.audio_spectrogram_transformer.layernorm)
        s.fc = nn.Linear(768, 3)
    def forward(s, x, mask):
        for b in s.blocks:
            o = b(x, attention_mask=mask); x = o[0] if isinstance(o, tuple) else o
        x = s.ln(x); return s.fc((x[:, 0] + x[:, 1]) / 2)
def batches(idx, bs, shuffle):
    idx = np.array(idx); L = np.array([n_tok[i] for i in idx])
    order = idx[np.argsort(L)]; chunks = [order[i:i + bs] for i in range(0, len(order), bs)]
    if shuffle: np.random.shuffle(chunks)
    for ch in chunks:
        xs = [h10(d.cut_id[i]) for i in ch]; m = max(len(x) for x in xs)
        X = torch.zeros(len(xs), m, 768); M = torch.zeros(len(xs), 1, 1, m)
        for k, x in enumerate(xs): X[k, :len(x)] = x; M[k, ..., len(x):] = -1e4
        yield ch, X, M
n_tok = [np.load(f'h10/{c}.npy', mmap_mode='r').shape[0] for c in d.cut_id]
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
which = [int(a) for a in sys.argv[2:]] or range(5)
for f in which:
    tr, ev = folds[f]
    net = Head(); cw = torch.tensor(len(tr) / (3 * np.bincount(y[tr], minlength=3)), dtype=torch.float32)
    opt = torch.optim.AdamW([{'params': list(net.blocks.parameters()) + list(net.ln.parameters()), 'lr': 3e-5},
                             {'params': net.fc.parameters(), 'lr': 1e-3}], weight_decay=0.01)
    steps = EPOCHS * int(np.ceil(len(tr) / 8)); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[3e-5, 1e-3], total_steps=steps, pct_start=0.1)
    lossf = nn.CrossEntropyLoss(weight=cw); t = time.time()
    for ep in range(EPOCHS):
        net.train(); tot = 0
        for ch, X, M in batches(tr, 8, True):
            loss = lossf(net(X, M), torch.from_numpy(y[ch])); opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(ch)
        print(f'fold {f} ep {ep} loss {tot/len(tr):.3f} {time.time()-t:.0f}s', flush=True)
    net.eval(); P = np.zeros((len(ev), 3)); pos = {i: k for k, i in enumerate(ev)}
    with torch.no_grad():
        for ch, X, M in batches(ev, 16, False):
            p = torch.softmax(net(X, M), -1).numpy()
            for k, i in enumerate(ch): P[pos[i]] = p[k]
    np.save(f'oof_ft{"" if EPOCHS == 3 else EPOCHS}_fold{f}.npy', np.c_[ev, P]); print('saved fold', f, flush=True)
