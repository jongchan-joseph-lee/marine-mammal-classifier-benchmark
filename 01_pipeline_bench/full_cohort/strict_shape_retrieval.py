"""Exploratory check of the 13 shape features with query-source-excluded fits.

This targeted follow-up is separate from the original 15 ablation conditions
and from the six main full-cohort representations. Only development cuts are
used. Candidate constraints and the source bootstrap are unchanged.
"""
import hashlib
import json

import numpy as np
import pandas as pd

from analysis_common import OUT, FullContext, configure_original, rr


def main():
    target = OUT / 'five_analyses/physical_strict'
    ctx = FullContext(['meta', 'physv0'])
    indices = np.concatenate([rr.GROUPS[k] for k in
                              ['timing', 'tonality', 'impulse', 'repeat']])
    assert len(indices) == 13 and len(set(indices)) == 13
    ctx.X['shape'] = ctx.X['physv0'][:, indices]
    configure_original(target, ['meta', 'shape'])
    rr.sourcefit_retrieval(ctx)
    table = target / 'tables'
    q = pd.read_csv(table / '03_source_excluded_fit_per_query.csv',
                    dtype={'cut_id': str})
    reference = pd.read_csv(
        OUT / 'five_analyses/tables/03_source_excluded_fit_per_query.csv',
        dtype={'cut_id': str})
    reference = reference[reference.feature.eq('meta')]
    key = ['condition', 'cut_id']
    baseline = q[q.feature.eq('meta')].set_index(key).sort_index()
    frozen = reference.set_index(key).sort_index()
    assert baseline.index.equals(frozen.index)
    compare = ['form', 'source', 'candidates', 'positive_candidates',
               'P5', 'null_P5'] + [f'neighbor_{i}' for i in range(1, 6)]
    pd.testing.assert_frame_equal(baseline[compare], frozen[compare],
                                  check_dtype=False, check_exact=True)
    byid = ctx.dev.set_index('cut_id')
    safe = True
    matched = True
    for col in [f'neighbor_{i}' for i in range(1, 6)]:
        keep = q[col].notna()
        query = byid.loc[q.loc[keep, 'cut_id']]
        neighbor = byid.loc[q.loc[keep, col].astype(str)]
        safe &= bool((query.g_site.to_numpy() != neighbor.g_site.to_numpy()).all())
        mask = q.loc[keep, 'condition'].eq('cross_source_matched').to_numpy()
        matched &= bool((query.fs_family.to_numpy()[mask] ==
                         neighbor.fs_family.to_numpy()[mask]).all())
        matched &= bool((np.abs(np.log2(query.dur_s.to_numpy(float)[mask]) -
                                np.log2(neighbor.dur_s.to_numpy(float)[mask])) <= 1).all())
    assert safe and matched
    assert not q.duplicated(['feature', 'condition', 'cut_id']).any()
    summary = pd.read_csv(table / '03_source_excluded_fit_summary.csv')
    error = 0.
    for (f, condition), d in q.groupby(['feature', 'condition']):
        saved = summary[summary.feature.eq(f) & summary.condition.eq(condition)]
        assert len(saved) == 1
        error = max(error, abs(d.groupby('form').P5.mean().mean() -
                               saved.iloc[0].P5_macro))
    assert error < 1e-12
    report = dict(complete=True,exploratory_followup=True,
                  feature_indices=indices.tolist(),dimensions=13,
                  baseline_identical_to_main_strict_retrieval=True,
                  all_neighbor_sources_differ=safe,
                  matched_neighbors_same_fs_and_within_factor_two_duration=matched,
                  query_counts=q.groupby(['feature','condition']).size().to_dict(),
                  max_summary_P5_error=error)
    report['query_counts'] = {' / '.join(k): int(v) for k,v in report['query_counts'].items()}
    report['output_sha256'] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in table.glob('03_source_excluded_fit_*.csv')}
    (target / 'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    summary.to_csv(OUT / 'summary/strict_shape_retrieval.csv',index=False)
    print(summary.to_string(index=False),flush=True)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
