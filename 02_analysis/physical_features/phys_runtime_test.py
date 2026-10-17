import time, numpy as np, phys
rng=np.random.default_rng(0)
for fs in [16000, 48000, 96000, 192000]:
    for T in [10, 60]:
        x=rng.normal(0,0.05,fs*T); k=rng.integers(0,len(x)-200,T*5)
        for i in k: x[i:i+50]+=rng.normal(0,1,50)
        x+=0.3*np.sin(2*np.pi*(fs/8)*np.arange(len(x))/fs)*(np.arange(len(x))%fs<fs//3)
        t=time.perf_counter(); v=phys.phys_features(x.astype(np.float64),fs); dt=time.perf_counter()-t
        print(f'fs {fs/1000:.0f} kHz, {T:3d} s audio -> {dt:.2f} s  ({T/dt:.0f}x real time), {len(v)} values')
