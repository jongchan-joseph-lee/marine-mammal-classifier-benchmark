#!/usr/bin/env python3
"""Run the five representations on one clip and keep every intermediate needed for the figure.

Example clip: 8101403Z, North Atlantic right whale (Eubalaena glacialis) moan, 14.9 kHz, 2.16 s
(Watkins DB; audio taken from the Hugging Face copy ivangtorre/watkins-marine-mammal-full-cuts).

Preprocessing follows the benchmark code:
  AST, VGGish  02_analysis/common_band_4k/emb.py (full band, band4k=False)
  EfficientNet 02_analysis/efficientnet_restore/eff_restored.py (restored recipe)
  MFCC         emb.mfcc_summary(x16, 16 kHz, 64 mel, 50-8000 Hz)
  Physical     02_analysis/physical_features/phys.py at the native sampling rate

Usage: python 04_figures/extractor_example/compute_example.py path/to/8101403Z.flac
Output: results/extractor_example_8101403Z.npz
"""
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "02_analysis/common_band_4k"), str(ROOT / "02_analysis/efficientnet_restore"),
                str(ROOT / "02_analysis/physical_features")]
from emb import prepare, tile_to, mfcc_summary, mel_filterbank, AST, VGG   # noqa: E402
import effcommon as ec                                                      # noqa: E402
from phys import phys_features                                              # noqa: E402

path = Path(sys.argv[1])
x, fs = sf.read(path, dtype="float64", always_2d=True)
x, fs = x[:, 0].copy(), int(fs)
R = {"fs": fs, "x": x.astype(np.float32)}

# AST: 16 kHz, tiled to 10.24 s, Kaldi fbank 128 x 1024, Transformer 12 layers, pooled 768
x16 = prepare(x, fs, False)
ast = AST()
nw = int(10.24 * 16000)
R["ast_in"] = ast.fe([tile_to(x16, nw).astype(np.float32)], sampling_rate=16000, return_tensors="np")["input_values"][0]
R["ast_vec"] = ast(x16)

# VGGish: 16 kHz, 0.96 s log-mel 96 x 64 patches, VGG CNN, 128 per patch, averaged
vg = VGG()
R["vgg_in"] = vg.proc(torch.from_numpy(tile_to(x16, 16000).astype(np.float32)))[:, 0].numpy()
R["vgg_vec"] = vg(x16)

# EfficientNet-B3: 40 kHz STFT image 300 x 800, ImageNet CNN, global average 1,536
from math import gcd
xc = x - x.mean(); g = gcd(40000, fs)
from scipy.signal import resample_poly
x40 = resample_poly(xc, 40000 // g, fs // g)
W = ec.windows(x40, mode="tile", agg="all")
imgs = [ec.to_img(ec.spec_db(w, lib="torch"), "top80", "gray", True) for w in W]
R["eff_in"] = imgs[0][..., 0]
R["eff_vec"] = ec.embed(imgs).mean(0)

# MFCC: 16 kHz frames, 64 mel 50-8000 Hz, log, DCT c1-c20, four statistics = 80
from scipy.fft import dct
win, hop = 400, 160; nfft = 512
nf = 1 + (len(x16) - win) // hop
idx = np.arange(win)[None, :] + hop * np.arange(nf)[:, None]
spec = np.abs(np.fft.rfft(x16[idx] * np.hanning(win)[None, :], n=nfft, axis=1)) ** 2
logmel = 10 * np.log10(spec @ mel_filterbank(64, nfft, 16000, 50, 8000).T + 1e-10)
R["mfcc_c"] = dct(logmel, type=2, axis=1, norm="ortho")[:, 1:21]
R["mfcc_vec"] = mfcc_summary(x16, 16000, 64, 50, 8000)

# Physical descriptors: native sampling rate, no resampling, 51 values
R["phys_vec"] = phys_features(x, fs)

out = ROOT / "results" / "extractor_example_8101403Z.npz"
np.savez_compressed(out, **{k: np.asarray(v) for k, v in R.items()})
print("saved", out, {k: np.asarray(v).shape for k, v in R.items()})
