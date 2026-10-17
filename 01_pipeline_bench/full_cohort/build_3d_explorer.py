"""Export all saved 3D UMAP coordinates to a portable, offline explorer."""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs

ROOT=Path(__file__).resolve().parents[1]
ORDER=['meta','mfcc16k','physv0','vggish','ast_zero','ast_tile']
LABEL=dict(meta='메타 대조군',mfcc16k='MFCC 16 kHz',physv0='물리 특징 51개',vggish='VGGish',ast_zero='AST 기본형',ast_tile='AST 반복 채움형')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--interim',action='store_true');args=ap.parse_args()
    out=ROOT/'output'
    audit=json.loads((out/'provenance/final_validation.json').read_text())
    if not args.interim and not audit['complete']:raise RuntimeError('Complete audit required for final explorer')
    m=pd.read_csv(out/'provenance/main_full_metadata.csv',dtype={'cut_id':str},low_memory=False).fillna('')
    columns=['cut_id','species','species_ko','group3','group3_ko','type_single','g_site','fs_hz','fs_family','dur_s','split']
    data=dict(columns=columns,rows=m[columns].to_numpy().tolist(),models=[],interim=args.interim)
    for f in ORDER:
        p=out/'stage1'/(f+'_structure.npz')
        if not p.exists():continue
        with np.load(p,allow_pickle=False) as z:
            assert np.array_equal(z['cut_id'].astype(str),m.cut_id.to_numpy(dtype=str))
            xyz=z['coords'];assert xyz.shape==(10694,3) and np.isfinite(xyz).all()
            data['models'].append(dict(id=f,label=LABEL[f],xyz=np.round(xyz.astype(float),5).tolist()))
    template=(ROOT/'scripts/3d_explorer_template.html').read_text()
    payload=json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    html=template.replace('__PLOTLY__',get_plotlyjs()).replace('__DATA__',payload)
    target=out/'Watkins_Full_Cohort_3D.html';target.write_text(html)
    print(json.dumps(dict(path=str(target),bytes=target.stat().st_size,models=len(data['models']),cuts=len(m),interim=args.interim),ensure_ascii=False))

if __name__=='__main__':main()
