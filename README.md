# Marine Mammal Classifier Benchmark

Watkins 해양포유류 음향 DB에서 사전학습 오디오 모델의 종 분류 성능을 녹음 출처 분리(source-disjoint) 조건으로 다시 평가한 프로젝트입니다.

## Overview

- **문제**: Watkins DB에서 높은 종 분류 성능을 보고한 선행연구들은 대부분 클립 단위 무작위 분할로 평가했습니다. 이 경우 같은 녹음에서 잘린 클립이 학습과 시험에 함께 들어갑니다.
- **방법**: 5개 표현(AST, VGGish, EfficientNet-B3, MFCC, 물리 특징 51개) + linear probe로 무작위 분할과 녹음 출처 분리 분할을 비교하고, 오디오 없이 메타데이터(샘플레이트 · 길이)만 쓴 기준선과 대조했습니다.
- **결과**: 녹음 출처를 분리하면 종 분류 균형정확도(BA)가 0.21–0.33으로 낮아져 메타데이터 기준선(0.24)과 비슷해집니다.

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

## Project Structure

| 폴더 | 내용 |
|---|---|
| `01_pipeline_bench` | 자료 구축, 특징 추출, 평가 파이프라인 |
| `02_analysis` | 미세조정, 신호 유형 분석, 최근접 이웃, 점검 스크립트 |
| `03_web` · `04_figures` | 결과 열람 HTML, 그림 생성 |
| `results` | 결과 수치 (JSON · CSV) |
| `docs` | 그림 |

## Limitations

- Watkins DB 내부 평가입니다. 외부 자료, 연속 녹음, 실해역 운용은 다루지 않았습니다.
- 녹음 출처는 메타데이터(지역 + 연도)로 정한 근사 단위입니다.
- 신호 유형 분석은 첫 결과 이후 정의한 탐색적 분석입니다.

## License

코드는 MIT License를 따릅니다. Watkins 원음 · 메타데이터와 사전학습 가중치는 각 원 조건을 따릅니다.
