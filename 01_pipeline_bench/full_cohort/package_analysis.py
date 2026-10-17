"""Package results, input vectors and runnable sources, without public audio.

Checkpoints are explicitly incomplete. The final archive reuses the same name
and is published only after scientific and rendering validation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',action='store_true');args=ap.parse_args()
    if not args.checkpoint:
        validation=ROOT/'output/provenance/final_validation.json'
        if not validation.exists() or not json.loads(validation.read_text()).get('complete'):
            raise RuntimeError('Final package requires a successful verify_completion.py audit.')
    output=ROOT/'deliverables';output.mkdir(exist_ok=True)
    target=output/'Watkins_Full_Cohort_Analysis.zip';temporary=target.with_suffix('.zip.pending')
    files=[]
    for folder in [ROOT/'scripts',ROOT/'input',ROOT/'output',BASE/'cohort_review/data',BASE/'cohort_review/scripts',
                   BASE/'deep_review/raw_new/src']:
        files.extend(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.pending'))
    for p in [ROOT/'requirements.txt',BASE/'deep_review/scripts/run_review.py',BASE/'deep_review/raw_new/data/bench_manifest.csv',
              BASE/'deep_review/output/tables/canonical_core_metadata.csv']:
        files.append(p)
    for f in ['mfcc16k','physv0']:
        files.append(BASE/f'deep_review/raw_new/work/features/{f}.npz')
    for f in ['vggish','ast_zero','ast_tile','effb3']:
        files.append(BASE/f'deep_review/raw/features/{f}.npz')
    # Reference comparisons are old-core-only, never substituted for missing
    # full-population analyses.
    files.extend((BASE/'deep_review/output/tables').glob('*.csv'))
    files.extend((BASE/'deep_review/output/methods/fonts').glob('*.ttf'))
    if args.checkpoint:
        files.extend(p for p in (ROOT/'work/embedding_cache').rglob('*.npz') if p.parent.name.startswith('ast_'))
    files=sorted(set(files))
    manifest=dict(complete=not args.checkpoint,n_eligible_main=10694,expected_full_features=['meta','mfcc16k','physv0','vggish','ast_zero','ast_tile'],
        available_full_features=[p.stem for p in (ROOT/'output/features').glob('*.npz')],
        audio_distribution='Public original FLACs are not bundled; download_full_audio.py obtains and verifies them.',
        gpu='compute_full_embeddings.py --device auto selects CUDA when available; FP32, TF32 disabled; completed embeddings reused.',
        checkpoint_message='Work in progress; incomplete AST representations must not be treated as full-cohort results.' if args.checkpoint else 'See final validation and review report.',
        files=[])
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for p in files:
            # Local report rendering PNGs are QA only.
            if 'qa_pages' in p.parts:continue
            name=str(p.relative_to(BASE));b=p.read_bytes()
            z.writestr(name,b);manifest['files'].append(dict(path=name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
        z.writestr('PACKAGE_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
        z.writestr('START_HERE_KO.txt',
            'Watkins 전체 사용 가능 자료 분석\n\n'+
            ('계산 중인 체크포인트입니다. 최종 결과가 아닙니다.\n\n' if args.checkpoint else '최종 결과 보고서와 그림은 full_cohort/output/에 있습니다.\n\n')+
            '분석 환경 및 재현 명령은 full_cohort/output/provenance/와 full_cohort/scripts/를 확인하세요.\n'+
            '기존 3539컷 결과는 deep_review/output/tables/에 별도로 있습니다.\n'+
            '사용 가능 주 분석은 10694컷이며 전체 재고 13693컷과 다릅니다.\n'+
            'EfficientNet-B3는 원 전처리 구현이 없어 원 표본 결과만 포함합니다.\n')
    temporary.replace(target)
    print(json.dumps(dict(path=str(target),bytes=target.stat().st_size,complete=manifest['complete'],files=len(manifest['files'])),indent=2))


if __name__=='__main__':main()
