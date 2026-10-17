import sys
sys.path.insert(0, '/home/claude/v14')
from pptx import Presentation
from pptx.util import Inches
from pptx.oxml.ns import qn
from lib14 import *
from lxml import etree

prs = Presentation('/home/claude/v14/in.pptx')
S = list(prs.slides)
s3 = S[2]                      # existing background slide (becomes 배경 ④)
TOC = S[1]
import lib14
lib14.NOTES_TEMPLATE = [sh for sh in s3.notes_slide.shapes if sh.is_placeholder and sh.placeholder_format.type == 2][0]._element
CONTENT_IDX = range(8, 56)     # study cards + problem boxes in slide 3

def base_from_s3(keep_problem=False):
    """Duplicate slide 3, keep frame (tag, title, footer, progress, band, number, icon)."""
    n = dup_slide(prs, s3)
    shs = list(n.shapes)
    for i in CONTENT_IDX:
        shs[i]._element.getparent().remove(shs[i]._element)
    return n

def frame(n):
    shs = list(n.shapes)
    # order after removal: 0 tag, 1 title, 2 footer, 3 progress, 4 band rect, 5 band L, 6 band R, 7 slnum, 8 icon
    return dict(tag=shs[0], title=shs[1], foot=shs[2], prog=shs[3], band=shs[4], bl=shs[5], br=shs[6], icon=shs[8])

# =====================================================================================  배경 ①
b1 = base_from_s3(); f = frame(b1)
set_tag(f['tag'], '배경 ①', 'Watkins DB와 선행연구')
set_title(f['title'], 'Watkins DB는 [[대표 공개 자료]]이며, 이를 쓴 선행연구는 [[높은 종 분류 성능]]을 보고함')
set_plain(f['foot'], 'Watkins DB: Sayigh et al. 2016; WHOI 2016 보도자료. 공개본 = Earth Species Project v0.1.0 (본 연구 사용본). '
          '표의 수치는 각 논문이 보고한 값(본 연구의 재현값 아님). WhaleNet은 동료심사 전 preprint. '
          'Cominelli et al.의 random forest는 Placentia Bay 자료에 적용(Watkins 아님). 서지 전체는 참고문헌 쪽')
# left: DB facts
card(b1, 0.6, 1.5, 3.85, 3.45, 'Watkins Marine Mammal Sound DB', size=12.5, center=True)
facts = [('1949 –', '70여 년간 녹음\nSchevill · Watkins 등 여러 연구자'), ('≈ 1,800', '원본 테이프 (WHOI 소장)'),
         ('60여 종', '해양포유류 · 2016년 온라인 공개'), ('13,693', '공개본 클립 · 49종\n본 연구 사용본')]
for k, (big, lab) in enumerate(facts):
    y = 2.0 + k * 0.72
    tb(b1, 0.7, y, 1.45, 0.62, big, size=19, color=NAVY, bold=True, align='r')
    tb(b1, 2.3, y, 2.1, 0.62, lab.split('\n'), size=10.5, color=TXT)
    if k < 3: rect(b1, 0.8, y + 0.67, 3.45, 0.0, None, LINE, 6350)
# right: prior studies table
rows = [['연구', '방법', '대상', '보고 성능', '평가 분할'],
        ['Hagiwara et al. 2023\n~BEANS 벤치마크', 'SVM · VGGish 등', '31종', '정확도 0.87', '**클립 무작위 6:2:2**'],
        ['Licciardi & Carbone 2024\n~WhaleNet (preprint)', 'wavelet · mel → ResNet', '32종', '정확도 97.6%', '**클립 무작위 75:25**'],
        ['Liang et al. 2025', 'EMD + MFCC → k-means', '2–5종', '군집 정확도 84.8%', '분할 없음 (군집)'],
        ['Cominelli et al. 2024', 'VGGish → UMAP', '12종', '8종 뚜렷한 군집', '비지도 · **같은 파일 0.96 s 구간**'],
        ['Lipari et al. 2026', 'AST + metric learning', '25종', '정확도 92.4%', '**클립 단위 70/15/15**'],
        ['Kim et al. 2026', 'EfficientNet-B3 + YOLO', '4종 혼합음', 'F-score 0.82–1.00', '합성 자료 (종 혼합)']]
table(b1, 4.7, 1.5, [2.05, 1.85, 0.85, 1.45, 1.83], [0.4] + [0.505] * 6, rows,
      aligns=['l', 'l', 'c', 'c', 'l'], size=10.5)
set_band(f['bl'], ['보고 성능 대부분이 **클립 단위 무작위 분할** 또는 분할 없는 군집 평가에서 나옴',
                   'Lipari et al.도 "같은 세션의 클립은 약한 상관을 만들 수 있음", **종당 세션 2–7개**를 한계로 명시'])
set_band(f['br'], ['Watkins DB는 70여 년간 여러 연구자가 **서로 다른 장비 · 해역 · 샘플레이트**로 녹음한 자료의 모음',
                   '→ 높은 성능이 **소리**를 배운 결과인지, **녹음**을 알아본 결과인지 확인 필요 (배경 ② · ③)'])
notes(b1, 'Watkins DB는 우즈홀 해양연구소가 1949년부터 70여 년간 모은 해양포유류 녹음으로, 원본 테이프 약 1,800개, 60여 종이며 2016년에 온라인으로 공개되었습니다. '
      '오른쪽은 이 자료로 종 분류를 보고한 연구들입니다. 정확도 85–98% 수준이 보고되었지만, 평가 분할 열을 보면 대부분 클립을 무작위로 나누거나 분할 없이 군집을 평가했습니다. '
      'Lipari 등도 같은 세션에서 나온 클립이 상관을 만들 수 있고, 종당 세션이 2–7개뿐이라는 점을 한계로 적었습니다. '
      '그래서 이 성능이 소리를 배운 결과인지 녹음을 알아본 결과인지를 확인할 필요가 있습니다. 다음 두 쪽에서 그 근거를 다른 분야의 사례로 보겠습니다.')

# =====================================================================================  배경 ②
b2 = base_from_s3(); f = frame(b2)
set_tag(f['tag'], '배경 ②', '녹음 비독립성 (pseudoreplication)')
set_title(f['title'], '같은 녹음의 클립이 학습 · 평가에 함께 들어가면 [[성능이 과대추정됨]]')
set_plain(f['foot'], 'Hurlbert 1984; Roberts et al. 2017; Kapoor & Narayanan 2023; Flexer 2007 (artist filter: 한 가수의 곡을 학습 · 평가 중 한쪽에만 두는 분할); '
          'Ford 1991; Janik & Sayigh 2013; Stowell 2022. 왼쪽 그림은 개념도(테이프 3개 · 클립 5개는 예시)')
# left: concept diagram
TR, TE = NAVY, ORANGE
def split_panel(y0, title, assign, note):
    card(b2, 0.6, y0, 5.6, 1.65, title, size=12.5)
    xs = [0.8, 2.6, 4.4]
    for t, x in enumerate(xs):
        tb(b2, x, y0 + 0.47, 1.6, 0.22, f'테이프 {"ABC"[t]}', size=10, color=GRAY, align='c')
        rect(b2, x, y0 + 0.7, 1.6, 0.42, 'F7F9FB', LINE, geom=MSO_SHAPE.ROUNDED_RECTANGLE)
        for c in range(5):
            rect(b2, x + 0.12 + c * 0.28, y0 + 0.79, 0.24, 0.24, TR if assign[t][c] == 0 else TE)
    tb(b2, 0.8, y0 + 1.17, 5.25, 0.42, note, size=10.5, color=TXT)
split_panel(1.5, '클립 랜덤 분할 (선행연구 방식)', [[0, 1, 0, 0, 1], [0, 0, 1, 0, 0], [1, 0, 0, 1, 0]],
            '같은 테이프가 학습 · 평가 양쪽에 → **개체 · 장비 · 배경 잡음을 외워도 정답**')
split_panel(3.3, '출처 단위 분할 (본 연구)', [[0] * 5, [0] * 5, [1] * 5],
            '테이프(출처)째로 나눔 → **처음 보는 녹음**에서의 성능을 잼')
# legend
rect(b2, 4.45, 1.62, 0.16, 0.16, TR, WHITE, 12700); tb(b2, 4.65, 1.58, 0.5, 0.24, '학습', size=10, color=WHITE)
rect(b2, 5.25, 1.62, 0.16, 0.16, TE, WHITE, 12700); tb(b2, 5.45, 1.58, 0.5, 0.24, '평가', size=10, color=WHITE)
# right: evidence
card(b2, 6.45, 1.5, 6.28, 3.45, '다른 분야의 근거', size=12.5)
EV = [('생태학', ['**Hurlbert 1984**: 독립이 아닌 반복을 독립 표본처럼 다루면 효과가 과장됨 (pseudoreplication, 유사 반복)',
                 '**Roberts et al. 2017**: 시간 · 공간 · 계층 구조가 있는 자료는 무작위 교차검증이 오차를 과소추정 → block CV 권장']),
      ('기계학습', ['**Kapoor & Narayanan 2023**: 17개 분야 294편에서 data leakage(데이터 누수) 확인',
                   '학습 · 평가 표본의 비독립성(non-independence)이 누수 유형의 하나']),
      ('음악 분류', ['**Flexer 2007**: 같은 가수의 곡이 학습 · 평가에 함께 있으면 장르 정확도 75.7%',
                    '가수 단위로 분리하면 58.5% → 방법 간 우열도 사라짐']),
      ('생물음향', ['같은 개체 · 집단의 소리는 서로 닮음: 범고래 집단 방언(**Ford 1991**), 큰돌고래 signature whistle(**Janik & Sayigh 2013**)',
                   '→ 같은 녹음의 클립은 독립 표본이 아님'])]
for k, (lab, txt) in enumerate(EV):
    y = 2.0 + k * 0.73
    rect(b2, 6.57, y, 1.05, 0.64, SKY, SKY)
    tb(b2, 6.57, y, 1.05, 0.64, lab, size=11, color=NAVY, bold=True, align='c')
    tb(b2, 7.75, y - 0.02, 4.88, 0.68, txt, size=10.5, color=TXT, line_sp=95)
set_band(f['bl'], ['Watkins DB에서 한 종의 클립은 대부분 **소수의 테이프 · 원정**에서 나옴 (결과 ①에서 확인)',
                   'Stowell 2022: 생물음향에서도 학습 · 평가 자료를 **같은 원천**에서 뽑는 관행이 여전히 흔함'])
set_band(f['br'], ['→ **RQ1** 클립 랜덤 분할에서 선행연구 수준의 성능이 재현되는가',
                   '→ **RQ2** 녹음 출처 단위로 나눠도 성능이 유지되는가'])
notes(b2, '첫 번째 문제는 녹음 비독립성입니다. 왼쪽 위처럼 클립을 무작위로 나누면 같은 테이프의 클립이 학습과 평가에 함께 들어갑니다. '
      '그러면 모델은 종의 소리가 아니라 그 녹음의 개체, 장비, 배경 잡음을 외워도 정답을 맞힙니다. 아래처럼 테이프째로 나눠야 처음 보는 녹음에서의 성능을 잴 수 있습니다. '
      '이 문제는 여러 분야에서 이미 보고되었습니다. 생태학에서는 Hurlbert가 pseudoreplication으로 정의했고, Roberts 등은 구조가 있는 자료에 block 교차검증을 권했습니다. '
      '기계학습에서는 Kapoor와 Narayanan이 17개 분야 294편에서 데이터 누수를 확인했습니다. 음악 장르 분류에서는 같은 가수의 곡을 나눠 두면 75.7%였던 정확도가 가수 단위로 나누면 58.5%로 떨어졌습니다. '
      '해양포유류에서는 범고래의 집단 방언이나 큰돌고래의 signature whistle처럼 같은 개체와 집단의 소리가 서로 닮기 때문에, 같은 녹음의 클립을 독립 표본으로 볼 수 없습니다. '
      '그래서 RQ1과 RQ2를 세웠습니다.')

# =====================================================================================  배경 ③
b3 = base_from_s3(); f = frame(b3)
set_tag(f['tag'], '배경 ③', '녹음 조건 편향 (recording bias)')
set_title(f['title'], '녹음 조건이 라벨과 얽히면 모델은 [[소리 대신 녹음 조건]]으로 판별할 수 있음')
set_plain(f['foot'], 'Geirhos et al. 2020; Zech et al. 2018; Sturm 2014. (a)(b) 원 음원을 16 kHz로 리샘플링한 스펙트로그램 '
          '(8800601U 북극고래 moan, 5702601X 참돌고래 whistle). (c) 분석 대상 10,694클립의 분류군별 샘플레이트 계열 비율 (상세는 부록 A15)')
card(b3, 0.6, 1.5, 5.6, 3.45, '다른 분야의 근거: 지름길 학습 (shortcut learning)', size=12.5)
EV = [('개념', ['**Geirhos et al. 2020**: 모델이 의도한 특징 대신 **평가 자료에서만 통하는 손쉬운 단서**를 학습', '→ 조건이 바뀌면 실패']),
      ('의료영상', ['**Zech et al. 2018**: 폐렴 판독 CNN이 영상만으로 **촬영 병원을 99.95% 식별**',
                   '두 병원 자료로 학습: 내부 AUC 0.93 → 외부 병원 0.82']),
      ('음악 분류', ['**Sturm 2014**: 정답은 맞히지만 음악과 무관한 단서에 기대는 "horse" 시스템',
                    '→ **단서를 통제한 검증**이 필요'])]
for k, (lab, txt) in enumerate(EV):
    y = 2.05 + k * 0.97
    rect(b3, 0.72, y, 1.05, 0.82, SKY, SKY)
    tb(b3, 0.72, y, 1.05, 0.82, lab, size=11, color=NAVY, bold=True, align='c')
    tb(b3, 1.9, y, 4.2, 0.82, txt, size=10.5, color=TXT, line_sp=95)
picture(b3, '/home/claude/v14/fig/bg3_fs.png', 6.5, 1.45, w=6.23)
set_band(f['bl'], ['Watkins DB의 샘플레이트는 5–192 kHz로 다양하고 **분류군과 강하게 얽힘**: 수염고래류 99.9%가 20 kHz 이하, 이빨고래류 92%가 40 kHz 이상 (c)',
                   '고정 입력(16 kHz)으로 바꿔도 **빈 대역의 넓이**로 원 샘플레이트가 드러남 (a)'])
set_band(f['br'], ['→ **메타데이터 baseline**: 샘플레이트 · 클립 길이만 넣은 같은 분류기 → 이 값을 넘어야 소리에서 정보를 얻었다고 판단',
                   '→ **RQ3** 샘플레이트 계열 안에서 비교해, 녹음 조건을 통제해도 남는 정보를 확인'])
notes(b3, '두 번째 문제는 녹음 조건 편향입니다. 기계학습에서는 이것을 지름길 학습, shortcut learning이라고 부릅니다. '
      '대표적인 예로 흉부 X선 폐렴 판독 모델이 영상만으로 어느 병원에서 찍었는지를 99.95% 맞혔고, 외부 병원에서는 성능이 떨어졌습니다. '
      '음악 분류에서도 음악과 무관한 단서로 맞히는 시스템을 horse라고 부르고, 단서를 통제한 검증을 요구합니다. '
      'Watkins DB에서는 샘플레이트가 그런 단서가 될 수 있습니다. 오른쪽 아래처럼 수염고래류는 거의 모두 20 kHz 이하, 이빨고래류는 92%가 40 kHz 이상으로 녹음되었습니다. '
      '모든 클립을 16 kHz로 바꿔 넣어도, 10 kHz 녹음은 5–8 kHz가 비어 있어서 빈 대역의 넓이만 보면 원래 샘플레이트, 나아가 분류군을 짐작할 수 있습니다. '
      '그래서 음원 없이 샘플레이트와 클립 길이만 넣은 메타데이터 baseline을 기준으로 삼고, 샘플레이트 계열 안에서 비교하는 RQ3를 세웠습니다.')

# =====================================================================================  배경 ④ (existing slide 3)
f = frame_s3 = None
shs = list(s3.shapes)
tag, title, foot, prog, band, bl, br = shs[0], shs[1], shs[2], shs[3], shs[4], shs[5], shs[6]
for i in range(8, 32):            # remove the four study cards (moved to 배경 ①)
    shs[i]._element.getparent().remove(shs[i]._element)
for i in range(32, 56):           # move the two problem boxes to the left
    shs[i].left = shs[i].left - Inches(6.65)
set_tag(tag, '배경 ④', '문제 정의와 연구 질문')
set_title(title, '두 문제를 분리해 측정하기 위해 [[세 가지 연구 질문(RQ)]]을 설정함')
set_plain(foot, 'Hurlbert 1984 (pseudoreplication); Geirhos et al. 2020 (shortcut learning). 평가 분할과 분류기의 상세는 재료 및 방법 ②. RQ3는 탐색적 분석')
# update the 대응 rows of the problem boxes
fill_tf(shs[43], '출처 단위 분할 → **RQ1과 RQ2의 차이**로 크기 측정', size=11)
fill_tf(shs[55], '메타데이터 baseline (RQ2) · 샘플레이트 계열별 비교 (RQ3)', size=11)
RQ = [('RQ1', '재현성', 'reproducibility', '선행연구처럼 클립을 무작위로 나누면 보고된 수준의 성능이 재현되는가?', '클립 랜덤 5-fold'),
      ('RQ2', '일반화', 'generalization', '녹음 출처를 분리해도 성능이 유지되고, 메타데이터 baseline을 넘는가?', '출처 단위 7-fold · baseline 대비 차이'),
      ('RQ3', '강건성', 'robustness', '샘플레이트를 통제해도 남는 정보는 종 · 분류군 · 신호 유형 중 어느 수준인가?', '샘플레이트 계열별 비교 (탐색적)')]
for k, (rq, ko, en, q, ev) in enumerate(RQ):
    y = 1.5 + k * 1.2; x = 6.35; w = 6.38
    rect(s3, x, y, 1.3, 1.08, NAVY, NAVY)
    tb(s3, x, y + 0.12, 1.3, 0.42, rq, size=18, color=WHITE, bold=True, align='c')
    tb(s3, x, y + 0.52, 1.3, 0.24, ko, size=11.5, color=WHITE, bold=True, align='c')
    tb(s3, x, y + 0.76, 1.3, 0.22, en, size=9, color='C8E9F8', align='c')
    rect(s3, x + 1.3, y, w - 1.3, 1.08, WHITE, LINE)
    tb(s3, x + 1.48, y + 0.1, w - 1.65, 0.56, q, size=12, color=TXT, bold=True)
    rect(s3, x + 1.48, y + 0.71, 0.6, 0.26, SKY, SKY)
    tb(s3, x + 1.48, y + 0.71, 0.6, 0.26, '평가', size=10, color=NAVY, bold=True, align='c')
    tb(s3, x + 2.18, y + 0.71, w - 2.4, 0.26, ev, size=10.5, color=TXT)
set_band(bl, ['두 문제 모두 성능을 **과대평가하는 방향**으로 작용하고, Watkins DB는 종 · 출처 · 샘플레이트가 겹쳐 **특히 취약**',
              'RQ1 → RQ2의 성능 차이 = **녹음 비독립성**의 크기'])
set_band(br, ['RQ2에서 baseline과의 차이 = **녹음 조건 밖의 소리 정보**',
              'RQ3: 녹음 조건을 고정한 비교로 **어느 수준의 분류가 가능한지** 확인 → 주석 체계 제안의 근거'])
notes(s3, '앞의 두 문제를 정리하면 왼쪽과 같습니다. 녹음 비독립성은 출처 단위 분할로, 녹음 조건 편향은 메타데이터 baseline과 샘플레이트 계열별 비교로 대응합니다. '
      '이를 세 가지 연구 질문으로 나눴습니다. RQ1은 선행연구 방식인 클립 랜덤 분할에서 성능이 재현되는지, RQ2는 출처를 분리해도 유지되고 baseline을 넘는지, '
      'RQ3는 샘플레이트를 통제해도 남는 정보가 어느 수준인지입니다. RQ1과 RQ2의 차이가 녹음 비독립성의 크기이고, RQ2에서 baseline과의 차이가 녹음 조건 밖의 소리 정보입니다.')

# reorder: new background slides go before old slide 3 (index 2)
for k, n in enumerate([b1, b2, b3]):
    move_slide(prs, n, 2 + k)
section_add(prs, '배경', [b1, b2, b3], before_id=258)

# =====================================================================================  References
summ = [s for s in prs.slides if s.shapes[0].has_text_frame and s.shapes[0].text_frame.text.startswith('연구 요약')][0]
REFS1 = [('Watkins DB와 선행연구', [
    ('Sayigh, L., Daher, M. A., Allen, J., Gordon, H., Joyce, K., Stuhlmann, C., & Tyack, P. (2016). The Watkins Marine Mammal Sound Database: An online, freely accessible resource. Proceedings of Meetings on Acoustics, 27, 040013.', 'Watkins DB 소개 · 배경 ①'),
    ('Woods Hole Oceanographic Institution (2016). Historic marine mammal sound archive now available online. WHOI news release, 20 June 2016.', 'DB 규모(테이프 · 종) · 배경 ①'),
    ('Hagiwara, M., Hoffman, B., Liu, J.-Y., Cusimano, M., Effenberger, F., & Zacarian, K. (2023). BEANS: The Benchmark of Animal Sounds. Proc. IEEE ICASSP 2023.', 'Watkins 31종 클립 무작위 분할 · 배경 ①'),
    ('Licciardi, A., & Carbone, D. (2024). WhaleNet: A novel deep learning architecture for marine mammals vocalizations on Watkins Marine Mammal Sound Database. arXiv:2402.17775 (preprint).', '32종 97.6% · 배경 ①'),
    ('Liang, Y., Wang, Y., Chen, F., Yu, H., Ji, F., & Chen, Y. (2025). Automatic detection and unsupervised clustering-based classification of cetacean vocal signals. Applied Sciences, 15(7), 3585.', 'EMD · MFCC · k-means · 배경 ①, 부록 A6 · A7'),
    ('Cominelli, S., Bellin, N., Brown, C. D., Rossi, V., & Lawson, J. (2024). Acoustic features as a tool to visualize and explore marine soundscapes: Applications illustrated using marine mammal passive acoustic monitoring datasets. Ecology and Evolution, 14(2), e10951.', 'VGGish · UMAP · 배경 ①, 결과 ③'),
    ('Lipari, V., Brancatelli, G., & Forlin, E. (2026). Marine mammal acoustic species recognition via metric embedding learning with Audio Spectrogram Transformer. IEEE Access, 14. doi:10.1109/ACCESS.2026.3714099', 'AST · 클립 단위 분할 · 배경 ①, 부록 A6 · A7'),
    ('Kim, M. J., Lee, J., Cho, Y., Kim, W.-K., Park, J., Lee, D., & Bae, H. S. (2026). Automated detection and classification of marine species vocalizations using a YOLO-based deep learning framework. Ecology and Evolution, 16(4), e73466.', 'EfficientNet-B3 · 배경 ①, 부록 A3')]),
 ('평가 설계 · 데이터 누수 · 지름길 학습', [
    ('Hurlbert, S. H. (1984). Pseudoreplication and the design of ecological field experiments. Ecological Monographs, 54(2), 187–211.', '유사 반복 · 배경 ②'),
    ('Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. Ecography, 40(8), 913–929.', 'block CV · 배경 ②'),
    ('Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. Patterns, 4(9), 100804.', '데이터 누수 294편 · 배경 ②'),
    ('Flexer, A. (2007). A closer look on artist filters for musical genre classification. Proc. ISMIR 2007, Vienna.', 'artist filter · 배경 ②'),
    ('Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & Wichmann, F. A. (2020). Shortcut learning in deep neural networks. Nature Machine Intelligence, 2, 665–673.', '지름길 학습 · 배경 ③'),
    ('Zech, J. R., Badgeley, M. A., Liu, M., Costa, A. B., Titano, J. J., & Oermann, E. K. (2018). Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study. PLoS Medicine, 15(11), e1002683.', '병원 식별 · 배경 ③'),
    ('Sturm, B. L. (2014). A simple method to determine if a music information retrieval system is a "horse". IEEE Transactions on Multimedia, 16(6), 1636–1644.', '단서 통제 검증 · 배경 ③'),
    ('Stowell, D. (2022). Computational bioacoustics with deep learning: A review and roadmap. PeerJ, 10, e13152.', '생물음향 평가 관행 · 배경 ②')])]
REFS2 = [('생물음향 · 생태', [
    ('Ford, J. K. B. (1991). Vocal traditions among resident killer whales (Orcinus orca) in coastal waters of British Columbia. Canadian Journal of Zoology, 69(6), 1454–1483.', '범고래 집단 방언 · 배경 ②'),
    ('Janik, V. M., & Sayigh, L. S. (2013). Communication in bottlenose dolphins: 50 years of signature whistle research. Journal of Comparative Physiology A, 199(6), 479–489.', 'signature whistle · 배경 ②'),
    ('Laidre, K. L., Stirling, I., Lowry, L. F., Wiig, Ø., Heide-Jørgensen, M. P., & Ferguson, S. H. (2008). Quantifying the sensitivity of Arctic marine mammals to climate-induced habitat change. Ecological Applications, 18(2 Suppl.), S97–S125.', '북극권 서식 종 목록 · 결과 ①'),
    ('Gillespie, D., Mellinger, D. K., Gordon, J., McLaren, D., Redmond, P., McHugh, R., et al. (2008). PAMGUARD: Semiautomated, open source software for real-time acoustic detection and localisation of cetaceans. Proceedings of the Institute of Acoustics, 30(5).', 'PAM 검출 소프트웨어 · 고찰')]),
 ('데이터 · 도구', [
    ('Watkins Marine Mammal Sound Database. Woods Hole Oceanographic Institution. 본 연구 사용본: Earth Species Project 공개본 v0.1.0.', '분석 자료 · 재료 및 방법 ①'),
    ('game-icons.net (CC BY 3.0). 분류군 아이콘, 물범은 자체 제작.', '결과 ⑥')]),
 ('특징 추출 · 분석 방법', [
    ('Gong, Y., Chung, Y.-A., & Glass, J. (2021). AST: Audio Spectrogram Transformer. Proc. Interspeech 2021, 571–575.', 'AST 원 논문 · 부록 A1'),
    ('Hershey, S., Chaudhuri, S., Ellis, D. P. W., Gemmeke, J. F., Jansen, A., Moore, R. C., et al. (2017). CNN architectures for large-scale audio classification. Proc. IEEE ICASSP 2017, 131–135.', 'VGGish 원 논문 · 부록 A2'),
    ('Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. Proc. ICML 2019, PMLR 97, 6105–6114.', 'EfficientNet 원 논문 · 부록 A3'),
    ('Davis, S. B., & Mermelstein, P. (1980). Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences. IEEE Trans. Acoustics, Speech, and Signal Processing, 28(4), 357–366.', 'MFCC 원 논문 · 부록 A4'),
    ('Huang, N. E., Shen, Z., Long, S. R., Wu, M. C., Shih, H. H., Zheng, Q., et al. (1998). The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis. Proc. R. Soc. Lond. A, 454, 903–995.', 'EMD 원 논문 · 부록 A6'),
    ('McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform manifold approximation and projection for dimension reduction. arXiv:1802.03426.', 'UMAP · 결과 ③'),
    ('Youden, W. J. (1950). Index for rating diagnostic tests. Cancer, 3(1), 32–35.', '임계값 기준 · 고찰')])]

def ref_slide(groups_left, groups_right, part):
    n = dup_slide(prs, summ)
    shs = list(n.shapes)
    keep = {0, 1, 2, 3}
    for i, sh in enumerate(shs):
        if i in keep or sh.is_placeholder and 'Slide Number' in sh.name: continue
        if sh.shape_type == 13: continue
        sh._element.getparent().remove(sh._element)
    shs = list(n.shapes)
    set_tag(shs[0], '참고문헌', f'References ({part}/2)')
    set_title(shs[1], '본문 · 그림 · 각주에 인용한 문헌을 [[주제별]]로 정리함 (▸ 인용 위치)')
    set_plain(shs[2], '저자 연도 순이 아니라 주제별로 묶음. 각 항목 끝의 ▸는 슬라이드에서 인용한 위치. 선행연구 4편의 수치는 원문 보고값')
    set_progress(shs[3], None)
    pic = [sh for sh in shs if sh.shape_type == 13][0]
    # swap icon image
    from pptx.parts.image import ImagePart
    img_part, rId = n.part.get_or_add_image_part('/home/claude/v14/ic_ref.png')
    pic._element.blipFill.blip.set(qn('r:embed'), rId)
    from lib14 import _para
    for col, groups in enumerate([groups_left, groups_right]):
        x = 0.6 + col * 6.2; w = 5.93
        box = n.shapes.add_textbox(IN(x), IN(1.45), IN(w), IN(5.4)); box.text_frame.word_wrap = True
        fill_tf(box, [''], size=9, anchor='t')
        txb = box._element.find(qn('p:txBody'))
        for p in txb.findall(qn('a:p')): txb.remove(p)
        for gi, (gname, items) in enumerate(groups):
            hp = _para('▌ ' + gname, 12, NAVY, True, 'l', NAVY, False, 4)
            if gi > 0:
                ppr = hp.find(qn('a:pPr')); sb = etree.Element(qn('a:spcBef')); sp = etree.SubElement(sb, qn('a:spcPts')); sp.set('val', '1000')
                ppr.insert(0, sb)
            txb.append(hp)
            for ref, where in items:
                p = _para(ref + '  **▸ ' + where + '**', 9.5, TXT, False, 'l', NAVY, False, 6, line_sp=95)
                txb.append(p)
    return n
r1 = ref_slide(REFS1[:1], REFS1[1:], 1)
r2 = ref_slide(REFS2[:2], REFS2[2:], 2)
section_add(prs, '참고문헌', [r1, r2], new_after='요약')

# =====================================================================================  footers on existing slides
REPL = [('Gong et al. 2021, MIT/', 'Gong et al. 2021 (AST 원 논문). 가중치: MIT/'),
        ('Hershey et al. 2017, torchaudio 공개 가중치.', 'Hershey et al. 2017 (VGGish 원 논문). 가중치: torchaudio 공개본.'),
        ('Tan & Le 2019, torchvision ImageNet 가중치.', 'Tan & Le 2019 (EfficientNet 원 논문). 가중치: torchvision ImageNet.'),
        ('Davis & Mermelstein 1980.', 'Davis & Mermelstein 1980 (MFCC 원 논문).'),
        ('경험적 모드 분해):', '경험적 모드 분해, Huang et al. 1998):'),
        ('VGGish, 표본 3,539클립. 군집 분리도', 'VGGish, 표본 3,539클립. UMAP: McInnes et al. 2018. 군집 분리도'),
        ('Youden 기준으로', 'Youden 기준(Youden 1950)으로'),
        ('Gillespie et al. 2008 (PAMGuard)', 'Gillespie et al. 2008 (PAMGuard: 공개 PAM 검출 · 위치추정 소프트웨어)')]
hits = {k: 0 for k, _ in REPL}
for sl in prs.slides:
    for sh in sl.shapes:
        if not (sh.is_placeholder and sh.has_text_frame): continue
        if abs(sh.top / 914400 - 7.0) > 0.05: continue
        t = sh.text_frame.text
        nt = t
        for a, c in REPL:
            if a in nt: nt = nt.replace(a, c); hits[a] += 1
        if nt != t: set_plain(sh, nt)
print('footer hits', hits)

# =====================================================================================  TOC
from pptx.enum.text import PP_ALIGN
for sh in list(TOC.shapes):
    if sh.shape_type == 17 and sh.left / 914400 >= 5.0:
        sh._element.getparent().remove(sh._element)
DARK = '404040'; G2 = '8A949E'
def tbox(x, y, w, h, t, size, color, bold=False, align='l'):
    tb(TOC, x, y, w, h, t, size=size, color=color, bold=bold, align=align)
COLS = [[('01', '배경', [('Watkins DB와 선행연구', '03'), ('녹음 비독립성', '04'), ('녹음 조건 편향', '05'), ('문제 정의와 연구 질문', '06')]),
         ('02', '재료 및 방법', [('데이터 구성 · 녹음 출처', '07'), ('분석 흐름 · 평가 설계', '08'), ('특징 추출기', '09'), ('콜 타입 라벨', '10')])],
        [('03', '결과', [('데이터 구조', '11'), ('재현성과 일반화 (RQ1 · RQ2)', '12'), ('UMAP 일반화 (RQ2)', '13'), ('신호 유형 강건성 (RQ3)', '14'), ('콜 타입 6종', '15'), ('단일 특징과 대역', '16')]),
         ('04', '고찰', [('결론 및 제언', '17'), ('주석 절차 제안', '18')])]]
XN = [(5.15, 5.75, 5.45, 8.35), (9.21, 9.80, 9.51, 12.40)]
HH, IH, GAP = 0.5, 0.34, 0.2
for c, col in enumerate(COLS):
    y = 0.87; xn, xm, xi, xp = XN[c]
    for num, name, items in col:
        tbox(xn, y, 0.55, HH, num, 20, NAVY, True); tbox(xm, y, 3.0, HH, name, 20, NAVY, True); y += HH + 0.04
        for it, pg in items:
            tbox(xi, y, 2.95, IH, '•  ' + it, 12.5, DARK); tbox(xp, y, 0.6, IH, '-- ' + pg, 12.5, G2, align='r'); y += IH
        y += GAP
rect(TOC, 5.15, 5.12, 7.85, 0.0, None, LINE, 9525)
tbox(5.15, 5.27, 6.0, HH, '부록 · 요약 · 참고문헌', 20, NAVY, True)
APP = [[('특징 추출기 구조 (A1–5)', '19'), ('선행연구 재현 (A6–7)', '24'), ('콜 타입 6종 상세 (A8–13)', '26')],
       [('공통 대역 · 샘플레이트 (A14–15)', '32'), ('연구 요약', '34'), ('참고문헌', '35')]]
for c, items in enumerate(APP):
    y = 5.81; xn, xm, xi, xp = XN[c]
    for it, pg in items:
        tbox(xi, y, 2.95, IH, '•  ' + it, 12.5, DARK); tbox(xp, y, 0.6, IH, '-- ' + pg, 12.5, G2, align='r'); y += IH

prs.save('/home/claude/v14/stage1.pptx')
print('slides', len(prs.slides))
