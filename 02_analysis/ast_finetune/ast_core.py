import numpy as np, torch
from scipy.io import wavfile
from transformers import ASTForAudioClassification, ASTFeatureExtractor
torch.set_num_threads(2)
FE = ASTFeatureExtractor()
def load_model():
    m = ASTForAudioClassification.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593').eval()
    return m
def fbank(path, min_s=1.28, max_s=10.24):
    sr, w = wavfile.read(path); assert sr == 16000
    w = (w.astype(np.float32) / 32768.0) if w.dtype == np.int16 else w.astype(np.float32)
    if w.ndim > 1: w = w.mean(1)
    n_min = int(min_s * sr)
    if len(w) < n_min: w = np.tile(w, int(np.ceil(n_min / len(w))))[:n_min]
    w = w[:int(max_s * sr)]
    n = 1 + (len(w) - 400) // 160
    f = FE._extract_fbank_features(w, max_length=n)
    tp = (f.shape[0] - 16) // 10 + 1
    f = f[:16 + 10 * (tp - 1)]
    return FE.normalize(f).astype(np.float32), tp
def embed(m, f, tp):
    a = m.audio_spectrogram_transformer; e = a.embeddings
    x = torch.from_numpy(f)[None]                      # (1,T,128)
    p = e.patch_embeddings(x)                          # (1, 12*tp, 768)
    pos = e.position_embeddings[0]
    pp = pos[2:].reshape(12, 101, 768)[:, :tp].reshape(-1, 768)
    cls = e.cls_token + pos[0]; dist = e.distillation_token + pos[1]
    return torch.cat([cls.expand(1, -1, -1), dist.expand(1, -1, -1), p + pp[None]], 1)
def run_layers(m, h, lo, hi):
    for blk in m.audio_spectrogram_transformer.layers[lo:hi]:
        o = blk(h); h = o[0] if isinstance(o, tuple) else o
    return h
def pool(m, h):
    h = m.audio_spectrogram_transformer.layernorm(h)
    return (h[:, 0] + h[:, 1]) / 2
