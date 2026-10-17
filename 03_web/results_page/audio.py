import numpy as np, soundfile as sf, subprocess, json, os, io
from multiprocessing import Pool
from scipy.signal import stft
from PIL import Image
ids = json.load(open('data_core.json'))['clips']['id']
CAP = 15.0; TW, FH = 96, 64
def enc(cid):
    src = f'/home/claude/audio_core/{cid}.flac'; dst = f'aud/{cid}.webm'
    x, fs = sf.read(src)
    if x.ndim > 1: x = x[:, 0]
    x = x[:int(CAP * fs)]
    # spectrogram tile at original fs (max-pool in time keeps clicks)
    n = int(2 ** np.round(np.log2(max(fs * 0.016, 64))))
    f, t, Z = stft(x, fs, nperseg=n, noverlap=n // 2)
    P = np.abs(Z) ** 2  # (F, T)
    F = P.shape[0]; fe = np.linspace(0, F, FH + 1).astype(int)
    Pf = np.array([P[fe[i]:max(fe[i + 1], fe[i] + 1)].mean(0) for i in range(FH)])  # (FH, T)
    T = Pf.shape[1]; te = np.linspace(0, T, TW + 1).astype(int)
    Pt = np.array([Pf[:, te[i]:max(te[i + 1], te[i] + 1)].max(1) if te[i] < T else Pf[:, -1] for i in range(TW)]).T
    L = 10 * np.log10(Pt + 1e-20); hi = np.percentile(L, 99.5); L = np.clip((L - (hi - 60)) / 60, 0, 1)
    tile = (L[::-1] * 255).astype(np.uint8)  # high freq on top
    br = '16k' if fs <= 16000 else ('20k' if fs <= 24000 else '28k')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', src, '-t', str(CAP), '-ac', '1', '-c:a', 'libopus', '-b:a', br, '-vbr', 'on', '-f', 'webm', dst], check=True)
    return cid, tile, min(len(x) / fs, CAP)
if __name__ == '__main__':
    with Pool(2) as p: res = p.map(enc, ids, chunksize=16)
    # atlas: 4 sheets
    per = int(np.ceil(len(ids) / 4)); cols = 30
    index = {}; sheets = []
    for k in range(4):
        part = res[k * per:(k + 1) * per]; rows = int(np.ceil(len(part) / cols))
        A = np.zeros((rows * FH, cols * TW), np.uint8)
        for j, (cid, tile, dur) in enumerate(part):
            r, c = divmod(j, cols); A[r * FH:(r + 1) * FH, c * TW:(c + 1) * TW] = tile
            index[cid] = {'s': k, 'r': r, 'c': c, 'd': round(dur, 2)}
        Image.fromarray(A).save(f'spec_{k}.png', optimize=True)
    # audio bundles (~4)
    order = [cid for cid, _, _ in res]; sizes = [os.path.getsize(f'aud/{c}.webm') for c in order]
    total = sum(sizes); nb = int(np.ceil(total / 9e6)); target = total / nb
    b, acc, off, fh = 0, 0, 0, open('audio_0.bin', 'wb')
    for cid, sz in zip(order, sizes):
        if acc > target * (b + 1) and b < nb - 1:
            fh.close(); b += 1; off = 0; fh = open(f'audio_{b}.bin', 'wb')
        fh.write(open(f'aud/{cid}.webm', 'rb').read()); index[cid].update(b=b, o=off, l=sz); off += sz; acc += sz
    fh.close()
    json.dump({'tw': TW, 'fh': FH, 'cols': cols, 'nb': nb, 'cap': CAP, 'idx': index}, open('media_index.json', 'w'))
    print('audio total MB', total / 1e6, 'bundles', nb); os.system('ls -la spec_*.png audio_*.bin')
