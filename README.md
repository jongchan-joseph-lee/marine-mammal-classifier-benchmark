# Marine Mammal Classifier Benchmark

Watkins 해양포유류 음향 DB에서 사전학습 오디오 모델의 종 분류 성능을 녹음 출처 분리(source-disjoint) 조건으로 다시 평가한 프로젝트입니다.

## Overview

- **문제**: Watkins DB에서 높은 종 분류 성능을 보고한 선행연구들은 대부분 클립 단위 무작위 분할로 평가했습니다. 이 경우 같은 녹음에서 잘린 클립이 학습과 시험에 함께 들어갑니다.
- **방법**: 5개 표현(AST, VGGish, EfficientNet-B3, MFCC, 물리 특징 51개) + linear probe로 무작위 분할과 녹음 출처 분리 분할을 비교하고, 오디오 없이 메타데이터(샘플레이트 · 길이)만 쓴 기준선과 대조했습니다.
- **결과**: 녹음 출처를 분리하면 종 분류 균형정확도(BA)가 0.21–0.33으로 낮아져 메타데이터 기준선(0.24)과 비슷해집니다.
- **MVP**: 클립별 물리량 14개로 신호 유형을 먼저 판정하고 분류군 확률을 내는 분류기를 만들었습니다. 녹음 출처 분리 조건에서 분류군 BA 0.75입니다.

<p align="center"><img src="docs/figures/fig1_study_design.png" width="90%" alt="Data selection, feature extractors and evaluation"></p>

<details>
<summary>특징 추출기별 처리 예시</summary>

<p align="center"><img src="docs/figures/extractor_example.png" width="95%" alt="One right whale clip passed through each feature extractor"></p>
<p align="center"><sub>북대서양긴수염고래 moan 클립 1개의 입력, 처리 구조, 출력 벡터</sub></p>

</details>

## Results

| Balanced accuracy | 무작위 분할 | 녹음 출처 분리 |
|---|---|---|
| 종 (10) | 0.64–0.76 | 0.21–0.33 (기준선 0.24) |
| 콜 타입 (7) | 0.55–0.74 | 0.29–0.48 |
| 분류군 (3) | 0.82–0.93 | 0.63–0.80 |

<p align="center"><img src="docs/figures/fig3_balanced_accuracy.png" width="90%" alt="Balanced accuracy under random and source-disjoint splits"></p>

<details>
<summary>임베딩 분포 (UMAP · t-SNE)</summary>

<p align="center"><img src="docs/figures/fig4_umap.png" width="85%" alt="UMAP of VGGish vectors"></p>
<p align="center"><sub>VGGish 표현 UMAP (3,539 클립)</sub></p>

<p align="center"><img src="docs/figures/fig4b_tsne.png" width="75%" alt="t-SNE of VGGish vectors"></p>
<p align="center"><sub>VGGish 표현 t-SNE — 종별 색, 분류군별 색</sub></p>

</details>

- 녹음 출처가 달라져도 신호 유형(tonal · impulsive · pulsed)과 주파수 대역 기준의 구분은 유지되었습니다 (탐색적 분석).

<details>
<summary>신호 유형 · 물리량 · 주파수 대역</summary>

<p align="center"><img src="docs/figures/fig5_signal_type.png" width="85%" alt="Balanced accuracy for three signal types"></p>
<p align="center"><sub>신호 유형 3분류 균형정확도 (탐색적)</sub></p>

<p align="center"><img src="docs/figures/fig6_physical_quantities.png" width="85%" alt="Distributions of single physical descriptors"></p>
<p align="center"><sub>단일 물리량 분포 (탐색적)</sub></p>

<p align="center"><img src="docs/figures/fig7_taxon_by_band.png" width="85%" alt="Taxon composition by signal type and frequency band"></p>
<p align="center"><sub>신호 유형 × 주파수 대역별 분류군 구성 (탐색적)</sub></p>

</details>

- AST 부분 미세조정: 전체 BA 0.784 → 0.752. 수염고래 · 이빨고래 재현율은 오르고 기각류는 내려갔습니다 ([결과](results/ast_finetune_results.json)).

## MVP: Signal-type-first classifier

신호 유형(tonal · impulsive · pulsed)을 먼저 판정하고, 그 확률과 주파수 대역으로 분류군(수염고래 · 이빨고래 · 기각류) 확률을 내는 분류기입니다. 입력은 클립별 물리량 14개이고, 신뢰도가 낮은 클립은 검토 대상으로 넘깁니다.

| 분류군 BA, 녹음 출처 분리 | 전체 (3,539 클립) | fs ≥ 40 kHz, 이빨고래 vs 기각류 |
|---|---|---|
| 샘플레이트만 | 0.66 | 0.50 |
| 주파수 대역만 | 0.71 | 0.49 |
| 신호 유형 + 대역 | 0.72 | 0.51 |
| 신호 유형 + 대역 + 모양 물리량 | 0.75 | 0.71 |
| 물리량 전체, 단일 단계 | 0.76 | 0.73 |

출처 7-fold를 20회 다른 분할로 반복한 평균입니다. 참고로 같은 과제에서 AST는 0.75–0.80, 물리 특징 51개는 0.74였습니다(분할 방식은 같고 fold 배정은 다름).

![Taxon accuracy by model, sampling-rate control and abstention](docs/figures/fig8_mvp.png)

- 샘플레이트를 40 kHz 이상으로 맞추면 주파수 대역은 구분력이 없고, 모양 물리량(임펄스 비율, 반복 간격, 조화성 등)이 구분을 유지했습니다.
- 샘플레이트 기준선 대비 개선폭은 +0.07이지만, 출처 단위 부트스트랩 95% 구간(−0.04–0.18)이 0을 포함합니다.
- 신호 유형 단계를 거친 모델은 단일 단계 모델과 비슷했습니다. 신호 유형을 중간 결과로 함께 내므로 주석(annotation) 도구의 첫 단계로 쓸 수 있습니다.
- 신뢰도 하위 30%를 검토로 넘기면 나머지 70%의 BA는 0.80입니다.

## Dataset

- [Watkins Marine Mammal Sound Database](https://whoicf2.whoi.edu/science/B/whalesounds/index.cfm) (WHOI), Earth Species Project 공개본 v0.1.0
- 10,694 클립 (36종, 녹음 출처 49개) → 평가 표본 3,539 클립
- 원음은 저장소에 포함하지 않습니다.

<details>
<summary>자료 구조</summary>

<p align="center"><img src="docs/figures/fig2_data_structure.png" width="85%" alt="Species by number of recording sources and sampling-rate family by taxon"></p>
<p align="center"><sub>종별 녹음 출처 수, 분류군별 샘플레이트 구성</sub></p>

</details>

## Getting Started

```bash
pip install -r requirements.txt
python tools/set_root.py ~/watkins_work   # 작업 경로 설정
```

파이프라인 순서: `01_pipeline_bench/src`의 `b01` → `b02` → `b04` → `b05` → `b06` → `b07`

## Reproduce

| 결과 | 스크립트 | 출력 |
|---|---|---|
| 무작위 분할 vs 녹음 출처 분리 BA | `01_pipeline_bench/src/b07_evaluate.py` | `results/reference_v1_results.csv` (`filecv` = 무작위, `allcv` = 녹음 출처 분리) |
| 신호 유형 3분류 | `02_analysis/signal_type3/eval3.py` | `results/signal_type3_regroup3.json` |
| AST 부분 미세조정 | `02_analysis/ast_finetune/eval_ft.py` | `results/ast_finetune_results.json` |
| MVP | `05_mvp/signal_type_first.py`, `05_mvp/fig_mvp.py` | `results/mvp_signal_type_first.json`, `docs/figures/fig8_mvp.png` |
| 그림 2–7 | `04_figures/figs.py` | `docs/figures/` |

MVP는 저장소에 포함된 `results/attributes_3539.csv`만으로 실행되며 원음이 필요 없습니다.

## Project Structure

| 폴더 | 내용 |
|---|---|
| `01_pipeline_bench` | 자료 구축, 특징 추출, 평가 파이프라인 |
| `02_analysis` | 미세조정, 신호 유형 분석, 최근접 이웃, 점검 스크립트 |
| `03_web` · `04_figures` | 결과 열람 HTML, 그림 생성 |
| `05_mvp` | 신호 유형 우선 분류기 (MVP) |
| `results` | 결과 수치 (JSON · CSV) |
| `docs` | 그림 |

## Limitations

- Watkins DB 내부 평가입니다. 외부 자료, 연속 녹음, 실해역 운용은 다루지 않았습니다.
- 녹음 출처는 메타데이터(지역 + 연도)로 정한 근사 단위입니다.
- 신호 유형 분석은 첫 결과 이후 정의한 탐색적 분석입니다.
- MVP는 클립 단위 물리량을 씁니다. 연속 녹음에서의 신호 구간 검출은 아직 포함하지 않았습니다.

## License

코드는 MIT License를 따릅니다. Watkins 원음 · 메타데이터와 사전학습 가중치는 각 원 조건을 따릅니다.
