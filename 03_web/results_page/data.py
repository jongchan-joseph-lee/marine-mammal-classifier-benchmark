import numpy as np, pandas as pd, json
REF = '/home/claude/ref_v1/eval/reference_v1'
d = pd.read_csv('/home/claude/pkg_in/data/bench_manifest.csv', low_memory=False)
d['cut_id'] = d.cut_id.astype(str); D = d.set_index('cut_id')
ids = np.load('/home/claude/pkg_in/work/features/physv0.npz', allow_pickle=True)['cut_id'].astype(str)
m = D.loc[ids]
SIG = {'whistle': 'tonal', 'moan': 'tonal', 'click': 'impulsive', 'pulse_train': 'impulsive', 'pulsed_call': 'pulsed', 'growl': 'pulsed'}
def s(x): return '' if pd.isna(x) else str(x)
clips = {'id': ids.tolist(), 'sp': [s(x) for x in m.species_ko], 'spe': [s(x) for x in m.species_en], 'tx': [s(x) for x in m.group3],
         'fs': [int(x) for x in m.fs_hz], 'dur': [round(float(x), 2) for x in m.dur_s], 'site': [s(x) for x in m.g_site],
         'split': [s(x) for x in m.split], 'type': [s(x) for x in m.task_type_single], 'sig': [SIG.get(s(x), '') for x in m.task_type_single],
         'multi': [s(x) for x in m.task_type_multi_tags], 'loc': [s(x)[:60] for x in m.location], 'year': [s(x).split('.')[0] for x in m.year],
         'note': [s(x)[:200] for x in m.note]}
out = {'clips': clips}
# ---- data composition (10,694)
main = d[(d.eligible == True) & (d.fs_family != 'LF(<5k)')]
sp = main.groupby(['species_ko', 'species_en', 'group3']).agg(n=('cut_id', 'size'), src=('g_site', 'nunique'), top=('g_site', lambda x: x.value_counts(normalize=True).iloc[0])).reset_index().sort_values(['group3', 'n'], ascending=[True, False])
out['species'] = sp.round(3).to_dict('records')
fams = ['5k', '10k', '20k', '40k', '80k', '160k']
out['fs_taxon'] = {g: {f: int(((main.group3 == g) & (main.fs_family == f)).sum()) for f in fams} for g in ['Odontoceti', 'Mysticeti', 'Pinnipedia']}
site = main.groupby('g_site').agg(n=('cut_id', 'size'), sp=('species', 'nunique'), tapes=('g_tape', 'nunique'), locs=('loc_key', 'nunique'),
                                   y0=('year', 'min'), y1=('year', 'max'), tx=('group3', lambda x: '/'.join(sorted(x.unique())))).reset_index().sort_values('n', ascending=False)
out['sites'] = [{k: (None if pd.isna(v) else (int(v) if isinstance(v, (np.integer, float)) and k in ('n', 'sp', 'tapes', 'locs', 'y0', 'y1') else v)) for k, v in r.items()} for r in site.to_dict('records')]
out['counts'] = {'all': int(len(d)), 'main': int(len(main)), 'core': int(len(ids)), 'species_main': int(main.species.nunique()), 'sites': int(main.g_site.nunique())}
# ---- linear probe
r = pd.read_csv(f'{REF}/results.csv'); r = r[(r.subset == 'all') & (r.metric == 'balanced_acc')]
FE = ['meta', 'physv0', 'mfcc16k', 'vggish', 'effb3', 'ast_tile']
p = pd.read_csv(f'{REF}/paired.csv'); p = p[p.A == 'meta']
probe = []
for _, x in r[r.feature.isin(FE) & r.protocol.isin(['filecv', 'allcv', 'regioncv', 'test'])].iterrows():
    row = dict(task=x.task, prot=x.protocol, feat=x.feature, ba=round(x.value, 3), lo=round(x.ci_lo, 3), hi=round(x.ci_hi, 3), chance=round(x.chance, 3), n=int(x.n_eval), sites=int(x.n_sites))
    q = p[(p.task == x.task) & (p.protocol == x.protocol) & (p.B == x.feature)]
    if len(q): row.update(d=round(-q.diff_balanced_acc.iloc[0], 3), dlo=round(-q.ci_hi.iloc[0], 3), dhi=round(-q.ci_lo.iloc[0], 3))
    probe.append(row)
out['probe'] = probe
pc = pd.read_csv(f'{REF}/per_class.csv'); pc = pc[(pc.subset == 'all') & (pc.metric == 'recall') & pc.feature.isin(FE) & pc.protocol.isin(['filecv', 'allcv'])]
out['per_class'] = [dict(task=a.task, prot=a.protocol, feat=a.feature, cls=a.cls, v=round(a.value, 3), n=int(a.n_pos)) for a in pc.itertuples()]
out['signal3'] = json.load(open('/home/claude/regroup/regroup3.json'))
out['ct6'] = json.load(open('/home/claude/v8/ct6.json'))
out['b4k'] = json.load(open('/home/claude/b4k/b4k_results.json'))
out['a7'] = {'cols': ['Baseline', 'MFCC', '물리 특징', 'VGGish', 'EffNet-B3', 'AST'], 'rows': [
    ['Random forest · 분류군 (Cominelli et al. 2024 분류기)', '랜덤', [0.94, 0.83, 0.88, 0.79, 0.90, 0.81]],
    ['', '출처 단위', [0.72, 0.56, 0.61, 0.49, 0.63, 0.56]],
    ['Random forest · 27종 (Cominelli et al. 2024 분류기)', '랜덤', [0.52, 0.80, 0.69, 0.67, 0.72, 0.69]],
    ['', '출처 단위', [0.11, 0.02, 0.04, 0.03, 0.06, 0.04]],
    ['Prototype matching · 29종 (Lipari et al. 2026)', '랜덤', [0.10, 0.41, 0.30, 0.40, 0.53, 0.58]],
    ['', '출처 단위', [0.10, 0.07, 0.08, 0.07, 0.14, 0.12]],
    ['k-means · 2–4종 (Liang et al. 2025)', '–', [0.70, 0.66, 0.56, 0.65, 0.78, 0.83]]]}
# ---- neighbours (recomputed today, 10,694 clips, k=10)
out['nn'] = [dict(feat='meta', all=0.452, same=0.538, other=0.106, multi=0.135), dict(feat='mfcc16k', all=0.708, same=0.747, other=0.124, multi=0.158),
             dict(feat='physv0', all=0.549, same=0.608, other=0.075, multi=0.096), dict(feat='vggish', all=0.731, same=0.767, other=0.073, multi=0.093),
             dict(feat='ast_tile', all=0.832, same=0.858, other=0.090, multi=0.115)]
# ---- tonal vs pulsed (today)
tp = pd.read_csv('/home/claude/v14/tp/tp_feats.csv', dtype={'cut_id': str}).set_index('cut_id')
tp = tp.join(D[['task_type_single', 'fs_family']])
out['tp'] = {'cols': ['mod_peak_db', 'mod_peak_hz', 'mod_depth', 'cpp_db', 'cpp_q', 'env_kurt_log'],
             'rows': {c: tp[tp.task_type_single == c][['mod_peak_db', 'mod_peak_hz', 'mod_depth', 'cpp_db', 'cpp_q', 'env_kurt_log']].round(4).values.tolist() for c in SIG},
             'probe': {'전체': {'meta baseline': 0.671, 'phys51': 0.557, 'new 6': 0.583, 'phys51 + new 6': 0.573},
                       '40 kHz 계열': {'meta baseline': 0.504, 'phys51': 0.570, 'new 6': 0.561, 'phys51 + new 6': 0.588}}}
out['summary'] = json.load(open('/home/claude/summary/summary.json'))
json.dump(out, open('data_core.json', 'w'), ensure_ascii=False, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
import os; print(os.path.getsize('data_core.json') / 1e6, 'MB'); print(out['counts']); print(len(probe))
