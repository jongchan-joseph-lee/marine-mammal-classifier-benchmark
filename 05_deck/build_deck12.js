// build_deck12.js — 랩 세미나 v12. 공통 격자: 제목 한 줄(굵게, 자동 크기) · 내용 y1.5–5.0 · 설명 띠 y5.2–6.8 · 각주 y7.0
const pptxgen = require("pptxgenjs");
const React = require("react"), RDS = require("react-dom/server"), sharp = require("sharp");
const md = require("react-icons/md");
// Theme step used a private helper in the original environment; it is optional.
let applyTheme = async () => {}; try { ({ applyTheme } = require("./apply_theme.js")); } catch (e) {}
const FIG = "/home/claude/ms/fig_deck/", OUT = "/home/claude/ms/deck/Watkins_lab_seminar_v12.pptx";
const X = { navy: "0B4EA2", blue: "1A5FB4", cyan: "00B5E2", sky: "C8E9F8", pale: "EEF7FC", gray: "7F8C99", line: "CFD8E1", dark: "2B2B2B", white: "FFFFFF" };
const THEME = { name: "Watkins Seminar Blue", headFontFace: "Malgun Gothic", bodyFontFace: "Malgun Gothic",
  colors: { dk1: X.dark, lt1: X.white, dk2: X.navy, lt2: X.pale, accent1: X.blue, accent2: X.cyan, accent3: X.gray, accent4: X.sky, accent5: "7FCDEB", accent6: "1A7FC4", hlink: X.blue, folHlink: X.gray } };
// 공통 격자
const L = 0.6, W = 12.13, TOP = 1.5, BOT = 5.0, BAND_Y = 5.2, BAND_H = 1.6, GAP = 0.25, HEAD = 0.5;

const pres = new pptxgen(); pres.layout = "LAYOUT_WIDE"; pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Watkins DB 종 분류 성능 재평가 — 랩 세미나"; pres.author = "Watkins screening study"; const C = pres.SchemeColor;
pres.defineSlideMaster({ title: "TITLE", background: { color: X.navy }, objects: [
  { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 1.9, w: 11.7, h: 1.8, fontSize: 36, bold: true, color: C.background1, align: "left", valign: "bottom", margin: 0 }, text: "" } },
  { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.95, w: 11.7, h: 1.5, fontSize: 18, color: C.accent4, align: "left", valign: "top", margin: 0 }, text: "" } } ] });
pres.defineSlideMaster({ title: "CONTENT", background: { color: X.white }, slideNumber: { x: 12.2, y: 7.0, w: 0.53, h: 0.28, fontSize: 10, color: X.gray, align: "right" }, objects: [
  { placeholder: { options: { name: "tag", type: "body", x: L, y: 0.3, w: 7.3, h: 0.3, fontSize: 12, bold: true, color: C.text2, align: "left", margin: 0 }, text: "" } },
  { placeholder: { options: { name: "title", type: "title", x: L, y: 0.68, w: W, h: 0.55, fontSize: 22, bold: true, color: C.accent1, align: "left", valign: "middle", margin: 0 }, text: "" } },
  { placeholder: { options: { name: "src", type: "body", x: L, y: 7.0, w: 11.4, h: 0.28, fontSize: 9, color: C.accent3, align: "left", margin: 0 }, text: "" } } ] });

// 제목 한 줄 맞춤: 한글 1, 그 밖 0.55, 공백 0.3 폭으로 셈
function fitSize(t) { let u = 0; for (const ch of t) u += /[가-힣]/.test(ch) ? 1 : ch === " " ? 0.3 : 0.56; const s = Math.min(24, Math.floor(W * 72 * 0.85 / u));
  if (s < 18) console.warn("제목이 깁니다 (" + s + "pt): " + t); return s; }
// **굵게** 표시를 런으로
const runs = (t, base = {}) => t.split(/(\*\*[^*]+\*\*)/).filter(Boolean).map((p) => p.startsWith("**") ? { text: p.slice(2, -2), options: { ...base, bold: true, color: X.navy } } : { text: p, options: { ...base } });
// 글머리 기호는 텍스트 상자 단위로 주고(BUL), 문단은 마지막 런의 breakLine 으로 나눈다
const BUL = (space = 6) => ({ bullet: { indent: 13 }, paraSpaceAfter: space });
function para(items, space = 6) { const out = []; items.forEach((t, i) => { const r = runs(t); r[0].options = { ...r[0].options, bullet: { indent: 13 }, paraSpaceAfter: space }; if (i < items.length - 1) r[r.length - 1].options.breakLine = true; out.push(...r); }); return out; }

const SECTIONS = ["배경", "재료 및 방법", "결과", "고찰"]; let section = "";
function slide(tag, title, src, notes) { const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(tag, { placeholder: "tag" }); s.addText(title, { placeholder: "title", fontSize: fitSize(title) }); if (src) s.addText(src, { placeholder: "src" }); if (notes) s.addNotes(notes);
  s.addText(SECTIONS.map((t, i) => ({ text: t + (i < 3 ? "  ›  " : ""), options: { color: t === section ? X.navy : "B8C2CC", bold: t === section } })), { x: 7.9, y: 0.3, w: L + W - 7.9, h: 0.3, fontSize: 10, align: "right", margin: 0, isTextBox: true, objectName: "progress" });
  return s; }
async function fig(s, file) { const mt = await sharp(FIG + file).metadata(); const ratio = mt.width / mt.height; const maxH = BOT - TOP; let w = W, h = w / ratio; if (h > maxH) { h = maxH; w = h * ratio; } s.addImage({ path: FIG + file, x: L + (W - w) / 2, y: TOP + (maxH - h) / 2, w, h, objectName: file, altText: file }); }
function band(s, left, right) { s.addShape(pres.shapes.RECTANGLE, { x: L, y: BAND_Y, w: W, h: BAND_H, fill: { color: X.pale }, line: { color: X.pale }, objectName: "explain band" });
  const cw = (W - 0.5 - GAP) / 2; [left, right].forEach((items, k) => { if (items) s.addText(para(items), { x: L + 0.25 + k * (cw + GAP), y: BAND_Y + 0.15, w: cw, h: BAND_H - 0.3, fontSize: 13, color: X.dark, valign: "middle", margin: 0, isTextBox: true, objectName: "explain" }); }); }
const ic = {}; async function icon(n) { if (ic[n]) return ic[n]; const Cmp = md[n]; if (!Cmp) return null; const png = await sharp(Buffer.from(RDS.renderToStaticMarkup(React.createElement(Cmp, { size: 256, color: "#FFFFFF" })))).png().toBuffer(); return (ic[n] = "image/png;base64," + png.toString("base64")); }
function rect(s, x, y, w, h, fill, line) { s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: line || fill, width: 0.75 } }); }
function T(s, t, x, y, w, h, o = {}) { s.addText(typeof t === "string" ? runs(t) : t, { x, y, w, h, fontSize: 13, color: X.dark, margin: 0, valign: "top", isTextBox: true, ...o }); }
// 카드: 머리 띠(높이 HEAD) + 본문 상자. 아이콘은 선택
async function card(s, x, y, w, h, head, o = {}) { rect(s, x, y, w, HEAD, o.headFill || X.navy); let tx = x + 0.18;
  T(s, head, tx, y, w - (tx - x) - 0.12, HEAD, { fontSize: 14, bold: true, color: X.white, valign: "middle", align: o.center ? "center" : "left", ...(o.center ? { x, w } : {}) });
  rect(s, x, y + HEAD, w, h - HEAD, X.white, X.line); }
const cols = (n) => { const w = (W - GAP * (n - 1)) / n; return Array.from({ length: n }, (_, i) => ({ x: L + i * (w + GAP), w })); };
const arrow = (s, x, y) => s.addShape(pres.shapes.CHEVRON, { x: x - 0.09, y: y - 0.16, w: 0.18, h: 0.32, fill: { color: X.cyan }, line: { color: X.cyan } });
const HD = (t) => ({ text: t, options: { bold: true, color: X.white, fill: { color: X.navy }, align: "center", valign: "middle" } });
const HL = (t) => ({ text: t, options: { bold: true, color: X.navy, fill: { color: X.sky }, align: "center" } });
const N = (t, o = {}) => ({ text: t, options: { align: "center", ...o } });
const B = (t) => ({ text: t, options: { bold: true, color: X.navy } });
const tableOpt = (o) => ({ x: L, y: TOP, w: W, fontSize: 13, color: X.dark, border: { type: "solid", pt: 0.75, color: X.line }, valign: "middle", margin: 0.08, ...o });

(async () => {
const FX = "/home/claude/ms/fig_x/";
const pbar = (s, x, y, w, h, P, truthIdx) => { const lab = ["이빨", "수염", "기각"], col = [X.navy, X.cyan, "9AA5AE"], rh = h / 3;
  P.forEach((v, j) => { const yy = y + j * rh; T(s, lab[j], x, yy, 0.45, rh, { fontSize: 9, valign: "middle", color: X.dark });
    const bw = Math.max(0.02, (w - 1.0) * v); rect(s, x + 0.47, yy + rh * 0.18, bw, rh * 0.64, col[j]);
    T(s, `${Math.round(v * 100)}%`, x + 0.5 + bw, yy, 0.5, rh, { fontSize: 9, valign: "middle", bold: j === truthIdx, color: j === truthIdx ? X.navy : X.dark }); }); };
const hdr = (s, t, x, w, y = TOP) => T(s, t, x, y, w, 0.26, { fontSize: 11, bold: true, color: X.navy, align: "center", valign: "middle" });
const sbox = (s, x, y, w, h, t, fill = X.pale, o = {}) => { s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.05, fill: { color: fill }, line: { color: X.navy, width: 0.6 } });
  const col = o.color || X.dark, rr = t.split(/(\*\*[^*]+\*\*)/).filter(Boolean).map((q) => q.startsWith("**") ? { text: q.slice(2, -2), options: { bold: true, color: col === X.white ? X.white : X.navy } } : { text: q, options: {} });
  s.addText(rr, { x: x + 0.05, y, w: w - 0.1, h, fontSize: o.fs || 9.5, align: "center", valign: "middle", color: col, bold: o.bold, margin: 0, isTextBox: true }); };
const cap = (s, t, x, y, w, h, fs = 9.5) => T(s, t, x, y, w, h, { fontSize: fs, color: X.dark, valign: "top" });
const darr = (s, x, y) => s.addShape(pres.shapes.CHEVRON, { x: x - 0.08, y: y - 0.06, w: 0.16, h: 0.14, rotate: 90, fill: { color: X.cyan }, line: { color: X.cyan } });

/* 1 */ section = "표지"; pres.addSection({ title: section });
{ const s = pres.addSlide({ masterName: "TITLE", sectionTitle: section });
  s.addText("Watkins 해양포유류 음향 DB 기반\n종 분류 성능의 재평가", { placeholder: "title" });
  s.addText("녹음 비독립성(pseudoreplication)과 녹음 조건 편향(recording bias) 분석 및\n신호 유형 기반 분류 체계 제안\n\n랩 세미나 · 2026. 10 · [발표자]", { placeholder: "body" });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.8, y: 6.1, w: 1.2, h: 0.08, fill: { color: X.cyan }, line: { color: X.cyan } });
  s.addNotes("Watkins DB로 보고된 종 분류 성능을 녹음 비독립성과 녹음 조건 편향의 관점에서 다시 평가하고, 그 결과를 바탕으로 신호 유형 기반 주석 체계를 제안합니다."); }

/* 2 */ section = "요약"; pres.addSection({ title: section });
{ const s = slide("연구 요약", "종 분류 성능은 녹음 출처 의존적이며, 녹음 조건에 강건한 정보는 신호 유형·대역 수준임", "표본 3,539클립 · linear probe · balanced accuracy(BA) · 녹음 출처: 테이프·장소·원정 기준으로 묶은 녹음 단위 49개(재료 및 방법 ①) · 선행연구: Liang et al. 2025; Cominelli et al. 2024; Lipari et al. 2026 · RQ3는 탐색적 분석",
    "RQ1: 선행연구 방식에서는 재현됩니다. RQ2: 녹음 출처가 바뀌면 유지되지 않고 메타데이터 baseline과 차이가 없습니다. RQ3: 샘플레이트를 통제해도 신호 유형과 주파수 대역 수준의 정보는 유지됩니다. 그래서 신호 유형 기반 주석 체계를 제안합니다.");
  const F = [["RQ1  재현성", "선행연구 방식에서\n성능이 재현되는가?", "재현됨", "랜덤 분할 종 분류\nBA 0.64–0.76"],
    ["RQ2  일반화", "녹음 출처가 바뀌어도\n성능이 유지되는가?", "유지되지 않음", "출처 단위 분할 0.21–0.33\nbaseline 0.24와 차이 없음"],
    ["RQ3  강건성", "녹음 조건을 통제해도\n유지되는 정보는?", "신호 유형 · 대역", "40 kHz 계열 신호 유형 3분류\nbaseline 0.34 vs 0.48–0.69"],
    ["제안", "어떤 분류 체계가\n필요한가?", "신호 유형 기반 주석", "신호 유형 + 물리 특징값\n+ 대역 기반 분류군 확률"]];
  const G = cols(4);
  for (let i = 0; i < 4; i++) { const { x, w } = G[i], [h, q, a, d] = F[i], last = i === 3; await card(s, x, TOP, w, BOT - TOP, h, { headFill: last ? X.cyan : X.navy, center: true });
    T(s, q, x + 0.15, TOP + HEAD + 0.2, w - 0.3, 0.7, { color: X.gray, align: "center" });
    T(s, a, x + 0.1, TOP + HEAD + 1.0, w - 0.2, 0.65, { fontSize: 20, bold: true, color: last ? X.cyan : X.navy, align: "center", valign: "middle" });
    T(s, d, x + 0.15, TOP + HEAD + 1.85, w - 0.3, 0.9, { align: "center" });
    if (i < 3) arrow(s, x + w + GAP / 2, TOP + HEAD + 1.33); }
  band(s, ["랜덤 → 출처 단위 분할 시 종 분류 BA **0.71 → 0.28** (6개 표현 평균): 녹음 비독립성에 의한 과대추정", "음원을 쓰지 않는 baseline(**0.24**)과 유의한 차이 없음: 녹음 조건 편향"],
    ["샘플레이트를 통제하면 **신호 유형** 수준의 정보는 유지됨", "RQ3 결과는 재분류 기준을 결과 확인 후 정한 **탐색적 분석**으로, 독립 데이터 검증 필요"]); }

section = "배경"; pres.addSection({ title: section });
{ const s = slide("배경  선행연구와 문제 정의", "선행연구의 높은 종 분리 성능에는 녹음 비독립성과 녹음 조건 편향이 섞였을 수 있음", "Sayigh et al. 2016; Liang et al. 2025; Cominelli et al. 2024; Lipari et al. 2026; Kim et al. 2026; Hurlbert 1984; Kapoor & Narayanan 2023; Geirhos et al. 2020. Cominelli et al.의 random forest는 Placentia Bay 자료에 적용(Watkins 아님)",
    "왼쪽은 Watkins DB로 높은 성능을 보고한 선행연구 네 편입니다. 오른쪽은 이 성능 안에 섞여 있을 수 있는 두 가지 문제입니다. 첫째, 같은 녹음에서 잘린 클립들이 학습과 평가에 함께 들어가면 성능이 과대추정됩니다. 생태학의 pseudoreplication, 기계학습의 data leakage에 해당합니다. 둘째, 샘플레이트와 클립 길이가 종과 상관되어 있어 분류기가 소리 대신 녹음 조건으로 판별할 수 있습니다.");
  const lw = 6.3, st = [["Liang et al. 2025", "EMD + MFCC → k-means", "84.8%", "2–5종 군집 정확도"], ["Cominelli et al. 2024", "VGGish → UMAP (비지도)", "8 / 12종", "뚜렷한 군집 형성"],
    ["Lipari et al. 2026", "AST + metric learning", "92.4%", "종 정확도 (frozen 73.1%)"], ["Kim et al. 2026", "EfficientNet-B3 + YOLO", "0.82–1.00", "합성 혼합음 4종 F-score"]];
  const cw = (lw - GAP) / 2, ch = (BOT - TOP - GAP) / 2;
  for (let i = 0; i < 4; i++) { const x = L + (i % 2) * (cw + GAP), y = TOP + Math.floor(i / 2) * (ch + GAP), [n, m, big, lab] = st[i];
    await card(s, x, y, cw, ch, n, { center: true });
    T(s, m, x + 0.1, y + HEAD + 0.08, cw - 0.2, 0.35, { fontSize: 11, color: X.gray, align: "center", valign: "middle" });
    T(s, big, x, y + HEAD + 0.42, cw, 0.5, { fontSize: 24, bold: true, color: X.navy, align: "center", valign: "middle" });
    T(s, lab, x, y + HEAD + 0.92, cw, 0.3, { fontSize: 11, align: "center", valign: "middle" }); }
  const rx = L + lw + 0.35, rw = L + W - rx, ph = (BOT - TOP - GAP) / 2;
  const PR = [["녹음 비독립성 (pseudoreplication)", "종별 클립이 소수의 녹음에서 연속 추출됨", "랜덤 분할 시 같은 녹음이 학습·평가에 동시 포함 → **성능 과대추정**", "출처 단위 분할"],
    ["녹음 조건 편향 (recording bias)", "종의 발성 대역에 맞춰 샘플레이트를 정해 녹음", "샘플레이트·클립 길이가 종과 상관 → **녹음 조건으로 판별** (shortcut learning)", "메타데이터 baseline · 샘플레이트 통제"]];
  PR.forEach(([h, c, e, r], k) => { const y = TOP + k * (ph + GAP);
    rect(s, rx, y, rw, HEAD, X.navy); T(s, h, rx + 0.18, y, rw - 0.3, HEAD, { fontSize: 14, bold: true, color: X.white, valign: "middle" });
    rect(s, rx, y + HEAD, rw, ph - HEAD, X.white, X.line);
    [["원인", c], ["영향", e], ["대응", r]].forEach(([kk, vv], j) => { const yy = y + HEAD + 0.08 + j * ((ph - HEAD - 0.16) / 3), hh = (ph - HEAD - 0.16) / 3 - 0.04;
      rect(s, rx + 0.1, yy, 0.75, hh, X.sky); T(s, kk, rx + 0.1, yy, 0.75, hh, { fontSize: 11, bold: true, color: X.navy, align: "center", valign: "middle" });
      T(s, vv, rx + 0.95, yy, rw - 1.05, hh, { fontSize: 11, valign: "middle" }); }); });
  band(s, ["평가 단위: Lipari et al.은 **클립 단위 70/15/15 분할**, Cominelli et al.은 **같은 파일에서 0.96초 구간 다수 추출**", "Watkins DB(Sayigh et al. 2016)는 종과 녹음 출처가 거의 1:1로 중첩 → **두 문제에 특히 취약**"],
    ["두 문제 모두 성능을 **과대평가하는 방향**으로 작용", "→ 분할 방식과 메타데이터 baseline을 함께 비교해 두 요인을 분리해 측정함"]); }

section = "재료 및 방법"; pres.addSection({ title: section });
{ const s = slide("재료 및 방법 ①  데이터 구성", "분석 대상 10,694클립과 표본 3,539클립을 구성하고, 교차검증 단위인 녹음 출처를 정의함", "Earth Species Project 공개본 v0.1.0 · 사육·불량 표기 등 = 사육·다종·공중 녹음·불량 표기 · 샘플레이트 5–192 kHz, 클립 길이 중앙값 1.9초 · 녹음 출처는 메타데이터(테이프 번호, 음원 해시, 날짜, 장소명, 위경도, 사육 시설)로 구성",
    "왼쪽은 클립 구성, 오른쪽은 녹음 출처의 정의입니다. 녹음 출처는 같은 테이프, 같은 날짜와 장소, 같은 장소, 같은 원정 중 하나라도 공유하는 클립을 계속 연결해 만든 묶음입니다. 그래서 서로 다른 출처는 테이프도 장소명도 겹치지 않습니다. 표본은 종별로 같은 장소·같은 해의 현장 묶음마다 최대 40클립까지 뽑았습니다.");
  const lw = 5.75, R = [["전체", 13693, "49종"], ["분석 대상", 10694, "36종 · 이빨고래류 8,820 · 수염고래류 1,385 · 기각류 489"], ["표본", 3539, "종별 현장 묶음당 최대 40클립"]];
  const cut = ["제외 2,999: 참고래 567 · fs < 5 kHz 115 · 해달 2 · 사육·불량 표기 등 2,315", "상한 초과 7,155클립 제외"];
  R.forEach(([n, v, sub], i) => { const y = TOP + i * 1.2, bw = lw * v / 13693;
    T(s, [{ text: `${n}  `, options: { bold: true, color: X.navy } }, { text: `${v.toLocaleString("en-US")}클립`, options: { bold: true } }], L, y, lw, 0.3, { fontSize: 14, valign: "middle" });
    T(s, sub, L, y + 0.3, lw, 0.25, { fontSize: 11, color: X.gray, valign: "middle" });
    s.addShape(pres.shapes.RECTANGLE, { x: L, y: y + 0.58, w: bw, h: 0.3, fill: { color: i === 2 ? X.cyan : X.navy }, line: { color: i === 2 ? X.cyan : X.navy } });
    if (cut[i]) T(s, "▼  " + cut[i], L + 0.1, y + 0.9, lw - 0.1, 0.28, { fontSize: 10.5, color: X.gray, valign: "middle" }); });
  const tx = L + lw + 0.35, tw = L + W - tx;
  T(s, [{ text: "녹음 출처 만드는 과정  ", options: { bold: true, color: X.navy } }, { text: "테이프 213 → 녹음 세션 171 → 현장 묶음 114 → ", options: { color: X.dark } }, { text: "녹음 출처 49", options: { bold: true, color: X.navy } }], tx, TOP, tw, 0.3, { fontSize: 11.5, valign: "middle" });
  T(s, "테이프 번호 · 날짜 · 장소명 · 1° 격자 · 원정 중 하나라도 같으면 같은 출처로 연결", tx, TOP + 0.3, tw, 0.25, { fontSize: 10, color: X.gray, valign: "middle" });
  const Sh = (t) => ({ text: t, options: { bold: true, color: X.navy } }), Nc = (t, o = {}) => ({ text: t, options: { align: "center", ...o } });
  s.addTable([[HD("녹음 출처 예시"), HD("테이프"), HD("장소"), HD("연도"), HD("종"), HD("클립"), HD("하나로 묶인 이유")],
    [Sh("A060 지중해"), Nc("1"), Nc("5"), Nc("1988–92"), Nc("5"), Nc("338"), "한 테이프에 여러 장소·연도"],
    [Sh("A002 버뮤다"), Nc("5"), Nc("8"), Nc("1951–63"), Nc("1"), Nc("167"), "같은 지역명 · 1° 격자"],
    [HL("A007 북서대서양"), Nc("64", { bold: true, color: X.navy, fill: { color: X.sky } }), Nc("22", { bold: true, color: X.navy, fill: { color: X.sky } }), Nc("1956–79\n(13개)", { bold: true, color: X.navy, fill: { color: X.sky } }), Nc("11", { bold: true, color: X.navy, fill: { color: X.sky } }), Nc("3,830\n(36%)", { bold: true, color: X.navy, fill: { color: X.sky } }), { text: "장소·격자·원정이 연쇄 연결", options: { fill: { color: X.sky } } }]],
    tableOpt({ x: tx, y: TOP + 0.65, w: tw, colW: [1.35, 0.55, 0.5, 0.88, 0.4, 0.7, tw - 4.38], rowH: [0.42, 0.75, 0.75, 0.85], fontSize: 10.5, margin: [0.03, 0.06, 0.03, 0.06] }));
  band(s, ["녹음 출처는 **테이프·장소명·1° 격자**로 구성됨", "최대 녹음 출처는 연도(13)·장소(22)·테이프(64)·클립(3,830)이 나타난 **북서대서양** 지역으로 나타남"],
    ["평가는 출처 49개에 대해 **7-fold Cross-Validation** 방식으로 수행 (같은 출처는 학습·평가에 함께 들어가지 않음)", "출처는 녹음 기록(테이프·날짜·장소)으로 정의 → **같은 개체가 다른 출처에 녹음됐는지는 확인할 수 없음**"]); }
{ const s = slide("재료 및 방법 ②  분석 흐름 · 평가 설계", "다섯 표현과 메타데이터 baseline을 같은 linear probe로, 세 가지 분할에서 비교함", "제외 2,999클립: 참고래 567(신호 품질) · fs < 5 kHz 115 · 해달 2 · 사육 2,018 · 다종 233 · 공중 녹음 114 · 불량 9 (중복 가능). 상한 초과 7,155클립(개발 5,823 · 검증 1,332). 사전학습 가중치 고정(frozen), fine-tuning 미실시",
    "왼쪽부터 데이터, 입력 조건, 특징 추출기, 클립당 벡터, 평가 순서입니다. 하늘색은 사전학습 신경망, 연한 색은 고정 계산, 점선은 음원을 쓰지 않는 메타데이터 baseline입니다. 모든 표현은 클립당 벡터 하나가 되고, 같은 linear probe로 평가합니다.");
  const y0 = TOP + 0.3, rows = 6, rg = 0.08, rh = (BOT - y0 - (rows - 1) * rg) / rows;
  const C = { d: [L, 1.75], i: [L + 1.95, 1.3], f: [L + 3.45, 3.55], v: [L + 7.2, 0.8], e: [L + 8.2, W - 8.2] };
  [["데이터", "d"], ["입력", "i"], ["특징 추출기", "f"], ["벡터", "v"], ["평가", "e"]].forEach(([t, k]) => { T(s, t, C[k][0], TOP, C[k][1], 0.24, { fontSize: 11.5, bold: true, color: X.navy, align: "center", valign: "middle" });
    s.addShape(pres.shapes.LINE, { x: C[k][0], y: TOP + 0.26, w: C[k][1], h: 0, line: { color: X.navy, width: 1 } }); });
  const box = (k, y, h, t, o = {}) => { s.addShape(pres.shapes.RECTANGLE, { x: C[k][0], y, w: C[k][1], h, fill: { color: o.fill || X.white }, line: { color: o.line || X.navy, width: 0.75, dashType: o.dash ? "dash" : "solid" } });
    T(s, t, C[k][0] + 0.06, y, C[k][1] - 0.12, h, { fontSize: o.fs || 10, align: "center", valign: "middle", color: X.dark }); };
  const arr = (x1, y1, x2, y2) => s.addShape(pres.shapes.LINE, { x: x1, y: y1, w: x2 - x1, h: y2 - y1, line: { color: X.navy, width: 0.9, endArrowType: "triangle" } });
  // data column
  const dh = (BOT - y0 - 2 * 0.45) / 3;
  [["**Watkins DB**\n13,693클립 · 49종"], ["**분석 대상**\n10,694클립 · 36종\n녹음 출처 49개"], ["**표본**\n3,539클립\n종별 현장 묶음당 ≤ 40"]].forEach(([t], i) => { const y = y0 + i * (dh + 0.45); box("d", y, dh, t, { fill: i === 2 ? X.sky : X.white, fs: 10 });
    if (i < 2) { arr(C.d[0] + C.d[1] / 2, y + dh, C.d[0] + C.d[1] / 2, y + dh + 0.45); T(s, i === 0 ? "−2,999" : "−7,155", C.d[0] + C.d[1] / 2 + 0.08, y + dh, 0.9, 0.45, { fontSize: 9.5, color: X.gray, valign: "middle" }); } });
  const R = [["16 kHz\n10.24 s 창", "**AST** · Lipari et al. 2026\nlog-mel 128 → 16×16 patch → Transformer", "768", X.sky],
    ["16 kHz\n0.96 s 단위", "**VGGish** · Cominelli et al. 2024\nlog-mel 64 → Conv 4단 → FC 3층", "128", X.sky],
    ["40 kHz\n5 s 창", "**EfficientNet-B3** · Kim et al. 2026\nlinear STFT 이미지 → MBConv", "1,536", X.sky],
    ["원 샘플레이트\n리샘플링 없음", "**물리 특징** · 본 연구\n초·Hz 단위 STFT → 6개 그룹", "51", X.pale],
    ["16 kHz\n25 ms 프레임", "**MFCC** · Liang et al. 2025\nmel 64 → DCT c1–c20 + Δ", "80", X.pale],
    ["음원 미사용", "**메타데이터 baseline**\n샘플레이트 · 클립 길이", "2", X.white]];
  const busX = C.i[0] - 0.12, dy = y0 + 2 * (dh + 0.45) + dh / 2;
  s.addShape(pres.shapes.LINE, { x: C.d[0] + C.d[1], y: dy, w: busX - C.d[0] - C.d[1], h: 0, line: { color: X.navy, width: 0.9 } });
  s.addShape(pres.shapes.LINE, { x: busX, y: y0 + rh / 2, w: 0, h: (rows - 1) * (rh + rg), line: { color: X.navy, width: 0.9 } });
  R.forEach(([inp, fx, v, fill], i) => { const y = y0 + i * (rh + rg), dash = i === 5;
    arr(busX, y + rh / 2, C.i[0], y + rh / 2);
    box("i", y, rh, inp, { dash, fs: 9.5 }); arr(C.i[0] + C.i[1], y + rh / 2, C.f[0], y + rh / 2);
    box("f", y, rh, fx, { fill, dash, fs: 9.5 }); arr(C.f[0] + C.f[1], y + rh / 2, C.v[0], y + rh / 2);
    box("v", y, rh, `**${v}**`, { fs: 12 });
    s.addShape(pres.shapes.LINE, { x: C.v[0] + C.v[1], y: y + rh / 2, w: 0.1, h: 0, line: { color: X.navy, width: 0.9 } }); });
  const bx = C.v[0] + C.v[1] + 0.1; s.addShape(pres.shapes.LINE, { x: bx, y: y0 + rh / 2, w: 0, h: (rows - 1) * (rh + rg), line: { color: X.navy, width: 0.9 } });
  const E2 = [[["**분류기: Linear probe**", "표준화 → PCA ≤ 64 → 로지스틱 회귀 (조정 없음)"], ["**과제별 평가 클립** (balanced accuracy)", "분류군(3) 3,539 · 종(10) 1,664\n콜 타입(7) 1,542 · 신호 유형(3) 1,456"]],
    [["**평가 분할 (RQ별)**", "RQ1 클립 랜덤 분할 · RQ2 출처 단위 분할\nRQ3 샘플레이트 계열별"], ["**baseline 비교**", "같은 클립 paired 차이 · 출처 bootstrap 95% CI"]]];
  const g2 = 0.3, eh = (BOT - y0 - g2) / 2;
  arr(bx, y0 + eh / 2, C.e[0], y0 + eh / 2);
  E2.forEach((parts, i) => { const y = y0 + i * (eh + g2), x = C.e[0], w = C.e[1];
    s.addShape(pres.shapes.RECTANGLE, { x, y, w, h: eh, fill: { color: X.white }, line: { color: X.navy, width: 0.75 } });
    s.addShape(pres.shapes.LINE, { x: x + 0.25, y: y + eh / 2, w: w - 0.5, h: 0, line: { color: X.line, width: 0.75, dashType: "dash" } });
    parts.forEach(([h, b], k) => T(s, h + "\n" + b, x + 0.06, y + k * eh / 2, w - 0.12, eh / 2, { fontSize: 10, align: "center", valign: "middle" }));
    if (i < 1) arr(x + w / 2, y + eh, x + w / 2, y + eh + g2); });
  band(s, ["표본 **3,539클립 전부**를 다섯 표현으로 변환 → 클립당 벡터 1개 → **동일 linear probe** (조정 없음, fold 내부에서만 학습)", "과제마다 **라벨이 있는 클립만** 평가 (종: 출처가 충분한 10종, 콜 타입: 메모에 하나만 기재) · 같은 과제 안에서는 **모든 표현이 같은 클립**"],
    ["**메타데이터 baseline**: 음원 없이 샘플레이트·클립 길이 2개 값만 쓴 같은 분류기 → 이 값을 넘어야 **소리에서 정보를 얻었다**고 판단", "선행연구 분류 절차(RF · prototype · k-means)도 재현해 비교 → 부록 A6–A7"]); }
{ const s = slide("재료 및 방법 ③  특징 추출기", "같은 클립을 다섯 표현에 넣으면, 리샘플링과 처리 방식에 따라 서로 다른 벡터가 나옴", "예시: 8101403Z 북대서양긴수염고래 moan (14.9 kHz, 2.16 s, 하모닉 0.4–6 kHz). 출력 벡터 = 이 클립의 실제 값(표준화해 색으로 표시). 구조 상세는 부록 A1–A5",
    "북대서양긴수염고래 moan 하나를 다섯 표현에 똑같이 넣은 결과입니다. 원본은 14.9 kHz 녹음이라 7.45 kHz까지만 기록되어 있습니다. AST, VGGish, MFCC는 16 kHz로 아주 조금 올려 넣기 때문에 moan이 그대로 들어가고, 7.45–8 kHz만 비어 있습니다. EfficientNet-B3는 40 kHz로 올려 넣기 때문에 7.45–20 kHz가 통째로 비어 있습니다. 이 빈 대역의 넓이가 곧 샘플레이트 정보라서, 녹음 조건을 알려 주는 단서가 될 수 있습니다. 물리 특징만 리샘플링 없이 원래 샘플레이트에서 계산합니다. 각 표현은 클립당 벡터 하나를 내고, 이후 모두 같은 linear probe로 평가합니다. 각 모델의 내부 구조는 부록에 자세히 두었습니다.");
  const C = { o: [L, 1.8], r: [L + 2.0, 1.55], i: [L + 3.75, 1.85], p: [L + 5.8, 1.75], v: [L + 7.75, 1.6], d: [L + 9.45, 0.75], a: [L + 10.3, W - 10.3] };
  [["원본 클립", "o"], ["리샘플링", "r"], ["모델 입력", "i"], ["처리", "p"], ["출력 벡터", "v"], ["차원", "d"], ["상세", "a"]].forEach(([t, k]) => hdr(s, t, C[k][0], C[k][1]));
  s.addImage({ path: FX + "m_s_orig.png", x: L, y: TOP + 0.35, w: 1.8, h: 1.8 * 1.5 / 1.9 });
  cap(s, "**북대서양긴수염고래 moan**\n14.9 kHz · 2.16 s\n하모닉 0.4–6 kHz\n(7.45 kHz까지만 기록)", L, TOP + 0.35 + 1.5, 1.8, 1.1, 10);
  const R5 = [["AST", "14.9 → 16 kHz ↑\n7.45–8 kHz 빈 대역", "m_in_ast.png", "Transformer 12층\n전역 attention", "m_out_ast.png", "768", "A1", X.sky],
    ["VGGish", "14.9 → 16 kHz ↑\n7.45–8 kHz 빈 대역", "m_in_vgg.png", "Conv 4단 + FC 3층\n0.96 s 국소 패턴", "m_out_vgg.png", "128", "A2", X.sky],
    ["EfficientNet-B3", "14.9 → 40 kHz ↑\n**7.45–20 kHz 빈 대역**", "m_in_eff.png", "MBConv CNN\n사진용 구조", "m_out_eff.png", "1,536", "A3", X.sky],
    ["MFCC", "14.9 → 16 kHz ↑\n7.45–8 kHz 빈 대역", "m_in_mfcc.png", "mel → log → DCT\n→ 평균·표준편차", "m_out_mfcc.png", "80", "A4", X.pale],
    ["물리 특징", "리샘플링 없음\n원 샘플레이트", "m_in_phys.png", "초·Hz 단위 STFT\n6개 그룹 51개", "m_out_phys.png", "51", "A5", X.pale]];
  const y0 = TOP + 0.35, lh = (BOT - y0 - 4 * 0.05) / 5;
  R5.forEach(([n, rs, ii, pr, oo, d, ap, fill], k) => { const y = y0 + k * (lh + 0.05), cy = y + lh / 2;
    sbox(s, C.r[0], y + 0.03, C.r[1], lh - 0.06, `**${n}**\n${rs}`, X.white, { fs: 9 });
    s.addImage({ path: FX + ii, x: C.i[0], y: y + 0.01, w: C.i[1], h: lh - 0.02 });
    sbox(s, C.p[0], y + 0.03, C.p[1], lh - 0.06, pr, fill, { fs: 9 });
    s.addImage({ path: FX + oo, x: C.v[0], y: y + 0.07, w: C.v[1], h: lh - 0.14 });
    T(s, d, C.d[0], y, C.d[1], lh, { fontSize: 12, bold: true, color: X.navy, align: "center", valign: "middle" });
    T(s, "부록 " + ap, C.a[0], y, C.a[1], lh, { fontSize: 10, color: X.gray, align: "center", valign: "middle" });
    [C.r[0], C.i[0], C.p[0], C.v[0]].forEach((xx) => arrow(s, xx - 0.1, cy)); });
  band(s, ["16 kHz 입력(AST·VGGish·MFCC)은 0–8 kHz만, EfficientNet-B3는 0–20 kHz를 봄 → 녹음마다 **빈 대역의 넓이가 다름**", "빈 대역의 넓이 = **샘플레이트 정보** → 소리가 아니라 녹음 조건을 알려 주는 단서가 될 수 있음"],
    ["물리 특징만 리샘플링 없이 **원 샘플레이트**에서 계산", "학습 데이터: AST·VGGish = 일반 오디오, EfficientNet-B3 = 사진, MFCC·물리 특징 = 학습 없음 → **해양포유류로 학습된 표현은 없음**"]); }
{ const s = slide("재료 및 방법 ④  콜 타입 라벨", "녹음 메모에 기재된 8개 콜 타입을 신호 특성에 따라 3개 신호 유형으로 재분류함", "예시 스펙트로그램은 원 샘플레이트 음원(Watkins 공개본)에서 산출. 클립: 5702601X, 8800601U, 9420108A, 6002602W, 9420108L, 6700900B",
    "콜 타입은 Watkins 녹음 메모의 원문 표현을 대응표로 8개 범주에 매핑한 것으로, 연구자가 일부를 들어 대체로 맞는 것을 확인했지만, 일치도를 수치로 낸 체계적 청취 검증은 아직 하지 않았습니다. 시간–주파수 구조를 기준으로 whistle과 moan은 tonal, click과 pulse train은 impulsive, pulsed call과 growl은 pulsed로 묶었습니다. song은 단위 신호들이 이어진 상위 구조라서, bark는 15클립뿐이라서 제외했습니다. 참고로 pulsed call로 적힌 클립 중에는 하모닉 tonal처럼 보이는 것도 있어, 라벨 자체의 한계도 함께 봐야 합니다.");
  const TY = [["Tonal", 852, [["whistle", 679], ["moan", 173]], [["whistle", "ex_whistle.png", "참돌고래 · 40 kHz"], ["moan", "ex_moan.png", "북극고래 · 10 kHz"]], "협대역 주파수 윤곽, 주파수 변조\n→ tonal concentration 높음"],
    ["Impulsive", 407, [["click", 252], ["pulse train", 155]], [["click", "ex_click.png", "참거두고래 · 47.6 kHz"], ["pulse train", "ex_pulse_train.png", "범고래 · 45.4 kHz"]], "광대역 펄스, 반복 간격으로 구분\n→ tonal concentration 낮음"],
    ["Pulsed", 197, [["pulsed call", 135], ["growl", 62]], [["pulsed call", "ex_pulsed_call.png", "참거두고래 · 47.6 kHz"], ["growl", "ex_growl.png", "하프물범 · 10.2 kHz"]], "빠른 펄스 반복 → 하모닉형 또는 거친 소리\n→ 물리 특징만으로는 tonal과 구분 어려움"]];
  const G = cols(3);
  for (let i = 0; i < 3; i++) { const { x, w } = G[i], [nm, n, inc, ex, desc] = TY[i];
    await card(s, x, TOP, w, BOT - TOP, `${nm}   ${n.toLocaleString("en-US")}클립`, { center: true });
    T(s, inc.map(([c, k]) => `**${c}** ${k}`).join("   ·   "), x, TOP + HEAD + 0.07, w, 0.3, { align: "center", valign: "middle" });
    const iw = (w - 0.3 - 0.12) / 2, ih = iw * 1.3 / 1.75, iy = TOP + HEAD + 0.42;
    ex.forEach(([c, f, cap], k) => { const ix = x + 0.15 + k * (iw + 0.12); s.addImage({ path: FIG + f, x: ix, y: iy, w: iw, h: ih, objectName: f, altText: c });
      T(s, [{ text: c, options: { bold: true, color: X.navy } }, { text: "\n" + cap, options: { color: X.gray } }], ix, iy + ih + 0.03, iw, 0.42, { fontSize: 10, align: "center" }); });
    rect(s, x + 0.15, BOT - 0.72, w - 0.3, 0.6, X.pale); T(s, desc, x + 0.25, BOT - 0.72, w - 0.5, 0.6, { fontSize: 11.5, align: "center", valign: "middle" }); }
  band(s, ["녹음 메모 원문 → 대응표(83개 표현) → 8개 콜 타입 (범주 체계는 본 연구 정의)", "**메모 기반 라벨**: 39%는 2개 이상 기재되어 단일 라벨 1,557클립만 사용 · 연구자 청취로 대체로 일치 확인 (일치도는 미산출)"],
    ["제외: **song** 86클립 (단위 신호가 이어진 상위 구조), **bark** 15클립 (표본 부족)", "재분류 기준은 결과 확인 후 결정 → 이에 기반한 결과(④–⑥)는 **탐색적 분석**"]); }

section = "결과"; pres.addSection({ title: section });
{ const s = slide("결과 ①  데이터 구조", "녹음 출처는 종 단위로, 샘플레이트는 분류군 단위로 얽혀 녹음 조건을 대신함", "분석 대상 10,694클립 · 36종 · 녹음 출처 49개. 회색 막대 = 1. ○ 북극권 서식 종 (Laidre et al. 2008): 북극고래·흰고래·일각고래·바다코끼리·고리무늬물범·턱수염물범·띠무늬물범·하프물범",
    "(a)는 36종과 49개 녹음 출처의 클립 수입니다. 위 막대는 출처마다 몇 종이 있는지, 오른쪽 막대는 종마다 몇 개 출처에서 나왔는지입니다. 출처 49개 중 38개가 한 종만 담고 있고, 36종 중 18종은 출처가 하나뿐입니다. 이런 구조에서는 종의 소리와 그 녹음의 특성을 떼어 낼 수 없습니다. (b)의 샘플레이트는 종 단위로는 여러 계열에 걸쳐 있지만 분류군 단위로 갈립니다. 수염고래류는 5–20 kHz, 이빨고래류는 40 kHz 이상에 몰려 있습니다. 동그라미는 북극권 종으로, 출처가 1–3개뿐입니다.");
  await fig(s, "../fig_x/fig_struct.png");
  band(s, ["(a) 녹음 출처는 **종 단위로 중첩(nested)**: 38/49 출처가 1종만, 18/36종이 출처 1개", "가장 큰 출처(북서대서양) 하나가 분석 대상의 36% · 11종"],
    ["(b) 샘플레이트는 종 단위로는 섞여 있으나(28/36종이 2개 계열 이상) **분류군 단위로 갈림**", "수염고래류 5–20 kHz 99.9% · 이빨고래류 40 kHz 이상 92% → **샘플레이트만으로도 분류군 구분 가능**"]); }
{ const s = slide("결과 ②  RQ1·RQ2 재현성과 일반화", "랜덤 분할에서는 선행연구 수준이 재현되나, 출처를 분리하면 baseline 수준으로 떨어짐", "표본 3,539클립. 출처 단위 교차검증에서 baseline을 유의하게 상회한 추출기 없음 (최대: AST 분류군 +0.12, 95% CI −0.01 ~ +0.26)",
    "빈 원은 랜덤 분할, 채운 원은 출처 단위 분할이고, 둘 사이 거리가 녹음 비독립성에 의한 과대추정분입니다. 세분화된 과제일수록 하락 폭이 큽니다. 점선은 메타데이터 baseline이며 추출기들이 그 근처에 있습니다.");
  await fig(s, "fig3.png");
  band(s, ["빈 원(랜덤) → 채운 원(출처 단위)의 차이 = 녹음 비독립성에 의한 **과대추정분**", "종 **0.64–0.76 → 0.21–0.33** (chance 0.10)", "콜 타입 0.55–0.74 → 0.29–0.48, 분류군 0.82–0.93 → 0.63–0.80"],
    ["점선(메타데이터 baseline): 종 **0.24**, 콜 타입 **0.41**, 분류군 **0.68**", "세분화된 과제일수록 하락 폭이 큼 (종 약 0.4 > 콜 타입 0.3 > 분류군 0.2)"]); }
{ const s = slide("결과 ③  RQ2 일반화", "종 라벨 기반 UMAP 군집은 학습 출처에 종속되어 미학습 출처로 일반화되지 않음", "VGGish, 표본 3,539클립. 군집 분리도 = 2D 지도에서 최근접 15개 이웃의 다수결로 종을 맞힌 BA. 단일 seed. Cominelli et al. 2024는 0.96초 단위 비지도 UMAP (본 분석은 종 라벨 사용 조건의 별도 검증)",
    "(c)는 29개 출처만으로 종 라벨을 써서 만든 지도이고, (d)는 같은 지도에 학습에 쓰지 않은 20개 출처의 클립을 라벨 없이 올린 것입니다. 연한 회색은 (c)의 클립입니다. 종 라벨을 쓴 supervised UMAP은 선행연구처럼 깔끔한 군집을 만들지만, 미학습 출처의 클립을 투영하면 군집 분리도가 0.16으로 비지도 UMAP과 비슷합니다. 라벨을 섞으면 군집이 생기지 않으므로 군집 자체는 실재하지만 학습 출처에 종속됩니다.");
  await fig(s, "fig4.png");
  band(s, ["(a)→(b) 종 라벨을 주면(supervised UMAP) 군집 분리도 **0.44 → 0.83**", "(c)→(d) 29개 출처로 만든 지도에 미학습 출처 20개를 투영하면 **0.77 → 0.16** (chance 0.06, 비지도 투영 0.12)"],
    ["라벨 무작위화 시 군집 미형성(0.06–0.10) → 군집은 실재하나 **학습 출처에 종속**", "최근접 이웃(10,694클립): 동일 종 비율 0.55–0.82 → 타 출처만 고려 시 **0.07–0.13**"]); }
{ const s = slide("결과 ④  RQ3 강건성 (탐색적 분석)", "샘플레이트를 통제하면 신호 유형 3분류는 baseline을 유의하게 상회함", "단일 라벨 1,456클립. 채운 원 = baseline 대비 차이의 95% CI가 0 미포함. 80 kHz 계열은 pulsed 12클립 (사실상 tonal vs impulsive)",
    "신호 유형 3분류는 출처 단위 분할에서도 하락 폭이 0.12–0.16으로 종 분류보다 훨씬 작습니다. 전체로는 baseline이 0.56으로 높아 유의하지 않지만, 40 kHz 계열만 보면 baseline이 chance 수준인 0.34로 떨어지고 6개 표현이 모두 유의하게 상회합니다.");
  await fig(s, "fig5.png");
  band(s, ["랜덤 → 출처 단위 분할 시 하락 폭 **0.12–0.16** (종 분류는 약 0.4)", "전체 데이터: baseline 0.56으로 높아 유의하게 상회한 추출기 없음"],
    ["40 kHz 계열: baseline **0.34**(chance 수준), **6개 표현 모두 유의** (EfficientNet-B3 +0.35, 물리 특징 +0.21)", "80 kHz 계열: 3개 유의 / pulsed 재현율이 가장 낮음 (0.42–0.56)"]); }
{ const s = slide("결과 ⑤  콜 타입 6종 (탐색적 분석)", "6종 콜 타입은 EfficientNet-B3·AST만 baseline을 넘고, growl·pulsed call은 늘 섞임", "단일 라벨 6종 1,456클립 · 출처 40개. 출처 단위 7-fold, 6클래스 linear probe. 3유형 = 6클래스 예측을 묶은 값(결과 ④의 3클래스 분류기와 다름). 채운 표시 = baseline 대비 차이의 95% CI(출처 bootstrap)가 0 미포함. 표현별 상세 혼동 행렬과 랜덤 분할 결과는 부록 A8–A13",
    "여섯 표현 각각으로 whistle, moan, click, pulse train, pulsed call, growl을 분류한 결과입니다. 왼쪽 작은 행렬은 출처를 분리했을 때 각 콜이 맞게 분류된 비율이고, 주황색 상자는 같은 신호 유형입니다. 메타데이터 baseline도 0.38로 chance의 두 배가 넘는데, 샘플레이트가 고주파 콜과 저주파 콜을 가르기 때문입니다. baseline을 유의하게 넘는 것은 EfficientNet-B3와 AST뿐입니다. EfficientNet-B3는 whistle을 0.81로 가장 잘 맞히는데, 0–20 kHz를 보는 표현이기 때문으로 보입니다. 반면 growl과 pulsed call은 어떤 표현에서도 잘 구분되지 않습니다. growl은 moan과, pulsed call은 whistle과 섞이는데, 메모 기반 라벨의 한계도 함께 작용했을 수 있습니다.");
  await fig(s, "../fig_x/fig_ct6sum.png");
  band(s, ["baseline만으로 6종 **0.38** (chance 0.17): 샘플레이트가 고주파 콜(whistle·click)과 저주파 콜(moan·growl)을 가름", "baseline을 넘는 표현: **EfficientNet-B3 0.55 (+0.16)** · **AST 0.48 (+0.09)** / 물리 특징·MFCC·VGGish는 차이 없음"],
    ["EfficientNet-B3 whistle 재현율 **0.81**: 0–20 kHz를 보는 유일한 신경망 표현과 일관", "**growl은 moan과, pulsed call은 whistle과** 모든 표현에서 섞임 → 특징의 한계 또는 메모 라벨의 한계"]); }
{ const s = slide("결과 ⑥  단일 특징과 대역 (탐색적 분석)", "유형·세부 콜은 단일 물리 특징으로 갈리고, 분류군은 대역을 따르되 1–8 kHz에서 중첩됨", "(a)–(c) 표본 단일 라벨, 학습 과정 없음. AUROC: 단일 값의 판별력 (0.5 = 무작위, 1 = 완전 구분). (d) 신호 대역 측정 가능 1,430클립, 중심 주파수는 원 샘플레이트에서 산출. 아이콘: game-icons.net (CC BY 3.0), 물범은 자체 제작",
    "왼쪽은 분류기 없이 물리 특징 하나만으로 얼마나 갈리는지입니다. tonal concentration 하나로 impulsive와 나머지가 AUROC 0.95로 갈리고, 중심 주파수로 whistle과 moan이, click 간격으로 click과 pulse train이 어느 정도 갈립니다. 오른쪽은 신호 유형과 중심 주파수 대역에 따른 분류군 비율입니다. 8 kHz 이상은 거의 이빨고래류, 1 kHz 이하의 tonal과 pulsed는 대부분 수염고래류이고, 1–8 kHz에서는 세 분류군이 섞입니다. 기각류는 대부분 이 중첩 대역에 있습니다.");
  await fig(s, "../fig_x/fig_m13.png");
  band(s, ["(a) tonal concentration: impulsive vs 나머지 **AUROC 0.95** (샘플레이트 통제 시 0.93–0.95)", "(b) whistle **7.8 kHz** vs moan **0.9 kHz** (0.90) · (c) click **11 ms** vs pulse train **3.5 ms** (0.76)"],
    ["(d) 8 kHz 초과: 이빨고래류 **98–99%** / 1 kHz 이하 tonal·pulsed: 수염고래류 **85–87%**", "1–8 kHz는 세 분류군 중첩 · 기각류 133클립 중 **112클립(84%)**이 이 대역 → 특징 개발의 우선 대상"]); }
section = "고찰"; pres.addSection({ title: section });
{ const s = slide("고찰  RQ3 기반 주석 절차 제안", "물리 특징값을 단계적으로 적용해 신호 유형과 분류군 확률을 출력하는 주석 절차를 제안함", "임계값: 표본 단일 라벨 1,456클립에서 Youden 기준으로 산출한 예시값(미고정). 분류군 확률: 결과 ⑥의 표본 비율(사전확률 미보정). 전문가 검토 기준 80%는 예시. Gillespie et al. 2008 (PAMGuard)",
    "왼쪽은 절차, 오른쪽은 실제 클립 세 개를 그 절차에 넣은 결과입니다. A 참돌고래 whistle은 8 kHz 이상이라 tonal인지 pulsed인지 몰라도 이빨고래류로 나옵니다. B 참거두고래 click은 impulsive, click형, 이빨고래류입니다. C 하프물범 growl은 1–8 kHz 중첩 대역이라 분류군을 정하지 못하고 전문가 검토로 넘어갑니다. 실패를 숨기지 않고 표시하는 것이 이 절차의 요점입니다. 참고로 B의 click 간격 89 ms는 약한 펄스까지 포함한 중앙값이라, 눈에 보이는 강한 click 간격(약 0.28 s)보다 짧게 잡힙니다. 판정에는 영향이 없지만 특징 정의는 다시 점검해야 합니다.");
  const fw = 4.45, nb = 0.42, bh = (BOT - TOP - 3 * 0.17) / 4;
  const STEP = [["1", "이벤트 검출", "원 샘플레이트에서 배경 소음 대비 에너지가 큰 구간 (Watkins 클립은 이미 잘린 상태)"],
    ["2", "신호 유형", "tonal concentration(TC) < 0.08 → **Impulsive**\nTC ≥ 0.08 → **Tonal·Pulsed** (둘의 구분은 미결정)"],
    ["3", "세부 측정", "중심 주파수(f50) 측정 · Impulsive는 click 간격\n≥ 9.5 ms → click형, 미만 → pulse train형"],
    ["4", "분류군 확률", "유형 × f50 대역 표(결과 ⑥) 조회\n최대 확률 < 80%이면 **전문가 검토**"]];
  STEP.forEach(([n, h, d], i) => { const y = TOP + i * (bh + 0.17);
    rect(s, L, y, nb, bh, X.navy); T(s, n, L, y, nb, bh, { fontSize: 18, bold: true, color: X.white, align: "center", valign: "middle" });
    rect(s, L + nb, y, fw - nb, bh, X.white, X.line);
    T(s, [{ text: h, options: { bold: true, color: X.navy, fontSize: 12, breakLine: true } }, ...runs(d, { fontSize: 10.5 })], L + nb + 0.15, y, fw - nb - 0.25, bh, { valign: "middle" });
    if (i < 3) s.addShape(pres.shapes.CHEVRON, { x: L + nb / 2 - 0.08, y: y + bh + 0.015, w: 0.16, h: 0.14, rotate: 90, fill: { color: X.cyan }, line: { color: X.cyan } }); });
  const x0 = L + fw + 0.3, ew = (L + W - x0 - 0.4) / 3, ih = ew / (2.2 / 1.35);
  const EXS = [["A", "참돌고래 · whistle", "ex18_A.png", ["② TC **0.77** → Tonal·Pulsed", "③ f50 **11.3 kHz** (8 kHz 초과)", "④ 이빨고래류 **98% 이상**"], "Tonal·Pulsed · 이빨고래류", X.cyan],
    ["B", "참거두고래 · click", "ex18_B.png", ["② TC **0.02** → Impulsive", "③ click 간격 89 ms → click형", "    f50 **13.0 kHz** (8 kHz 초과)", "④ 이빨고래류 **99%**"], "Impulsive(click형) · 이빨고래류", X.cyan],
    ["C", "하프물범 · growl", "ex18_C.png", ["② TC **0.28** → Tonal·Pulsed", "③ f50 **2.3 kHz** (1–8 kHz)", "④ 이빨 51–53%", "    수염 26–40% · 기각 7–24%"], "분류군 미결정 → 전문가 검토", X.navy]];
  EXS.forEach(([k, truth, f, steps, out, oc], i) => { const x = x0 + i * (ew + 0.2);
    rect(s, x, TOP, ew, HEAD, X.navy);
    T(s, [{ text: `예시 ${k}`, options: { bold: true, color: X.white, fontSize: 12, breakLine: true } }, { text: `실제: ${truth}`, options: { color: X.sky, fontSize: 9.5 } }], x + 0.12, TOP, ew - 0.2, HEAD, { valign: "middle" });
    s.addImage({ path: FIG + f, x, y: TOP + HEAD + 0.05, w: ew, h: ih, objectName: f, altText: `예시 ${k}` });
    const sy = TOP + HEAD + 0.1 + ih;
    s.addText(steps.flatMap((t, j) => { const r = runs(t); if (j < steps.length - 1) r[r.length - 1].options.breakLine = true; return r; }), { x: x + 0.05, y: sy, w: ew - 0.05, h: BOT - 0.5 - sy - 0.04, fontSize: 10, color: X.dark, valign: "middle", margin: 0, isTextBox: true, paraSpaceAfter: 2 });
    rect(s, x, BOT - 0.5, ew, 0.5, oc); T(s, out, x, BOT - 0.5, ew, 0.5, { fontSize: 11, bold: true, color: X.white, align: "center", valign: "middle" }); });
  band(s, ["② 기준 0.08: Impulsive 88% · Tonal 87% · Pulsed 98%가 올바른 쪽으로 분류 (AUROC 0.95)", "**Tonal vs Pulsed는 현재 특징으로 구분 불가**: 단일 특징 최대 AUROC 0.71, 물리 특징 51개 로지스틱 회귀(출처 단위 분할) BA 0.57"],
    ["③ click 간격 9.5 ms: BA 0.71 / Tonal 확정 시 whistle형·moan형 f50 2.7 kHz: BA 0.85", "1–8 kHz는 세 분류군이 중첩 → C처럼 **미결정으로 표시**하고 추가 특징(하모닉 간격, 펄스 반복률) 개발 대상"]); }
{ const s = slide("결론 및 제언", "랜덤 분할 성능은 녹음 식별을 반영하며, 종 수준 검증에는 다출처 자료가 필요함", "출처 3개 이상·각 20클립 이상(예시 기준)을 만족하는 6종: 범고래·혹등고래·참거두고래·참돌고래·큰머리돌고래·줄박이돌고래",
    "결론입니다. 클립을 랜덤으로 나누면 종 분류가 0.64–0.76으로 선행연구 수준이 나오지만, 출처를 분리하면 0.21–0.33으로 떨어지고 이는 음원을 쓰지 않는 메타데이터 baseline 0.24와 비슷합니다. Watkins의 종과 출처가 거의 1:1로 묶여 있기 때문입니다. 오른쪽처럼 종마다 여러 해역과 장비의 출처가 있어야 이 문제를 측정하고 통제할 수 있습니다. 다만 녹음 출처에는 개체군이나 행동 맥락 같은 생물학적 차이도 섞여 있으므로, 출처 의존성 전체를 기술적 편향으로 볼 수는 없습니다. 그전까지는 종 대신 신호 유형, 대역, 분류군 확률 수준으로 주석하고, 종 수준 주장은 다출처 6종으로 좁히는 것을 제안합니다.");
  const S3 = [["클립 랜덤 분할", "0.64–0.76", "종 분류 BA · 선행연구 방식", X.navy], ["출처 단위 분할", "0.21–0.33", "종 분류 BA · 신규 녹음에 가까움", X.navy], ["메타데이터 baseline", "0.24", "종 분류 BA · 음원 미사용", X.cyan]];
  const cw = 2.15, ch = 1.05;
  for (let i = 0; i < 3; i++) { const y = TOP + i * (ch + 0.12), [h, big, d, col] = S3[i];
    rect(s, L, y, cw, ch, col); T(s, h, L, y + 0.04, cw, 0.3, { fontSize: 11, bold: true, color: X.white, align: "center", valign: "middle" });
    T(s, big, L, y + 0.32, cw, 0.48, { fontSize: 24, bold: true, color: X.white, align: "center", valign: "middle" });
    T(s, d, L, y + 0.76, cw, 0.25, { fontSize: 9.5, color: X.white, align: "center", valign: "middle" }); }
  const fx = L + cw + 0.4, fw = 5.0, fh = fw * 773 / 1298; s.addImage({ path: FX + "fig_concept.png", x: fx, y: TOP + (BOT - TOP - fh) / 2, w: fw, h: fh });
  const nx = fx + fw + 0.35, nw = L + W - nx;
  await card(s, nx, TOP, nw, BOT - TOP, "필요한 자료 조건");
  s.addText(para(["종마다 **서로 다른 해역·장비**의 출처 여러 개", "같은 샘플레이트 안에 **여러 분류군**", "**검증된 이벤트 라벨** (call 단위 시작·끝)", "연속 녹음 (이벤트 검출·소음 포함)"], 8), { x: nx + 0.2, y: TOP + HEAD + 0.15, w: nw - 0.35, h: BOT - TOP - HEAD - 0.3, fontSize: 12, color: X.dark, valign: "top", margin: 0, isTextBox: true });
  band(s, ["**결론**: 랜덤 분할 기반 종 분류 성능은 음향 특성보다 **녹음 식별 능력**을 반영함 (출처 분리 시 baseline 수준)", "[대안 해석] 출처에는 개체군·행동 맥락도 섞여 있어 **출처 의존성 전체를 기술적 편향으로 볼 수는 없음**"],
    ["**제언**: 그전까지는 **신호 유형·대역·분류군 확률** 수준으로 주석, 종 수준 주장은 다출처 6종으로 한정", "다음 단계: 기준 고정 → 청취 검증(390클립) → Watkins 외 연속 녹음에서 검증"]); }
section = "부록"; pres.addSection({ title: section });
  { const s = slide("부록 A1  AST (Transformer)", "AST는 스펙트로그램을 1,212개 패치로 나눠, 모든 패치 사이의 관계를 12층에 걸쳐 계산함", "Gong et al. 2021, MIT/ast-finetuned-audioset-10-10-0.4593. attention rollout: 12층 attention(head 평균)을 곱해 요약 토큰이 각 패치에 둔 비중을 근사한 해석 보조 지표. 예시: 8101403Z 북대서양긴수염고래 moan (14.9 kHz, 2.16 s).",
      "AST는 원래 사진용으로 만든 Vision Transformer를 소리에 적용한 모델입니다. 먼저 log-mel 스펙트로그램을 16×16 패치로 10칸씩 겹쳐 자르면 주파수 12 × 시간 101 = 1,212개 조각이 됩니다. 각 조각을 768차원 토큰으로 바꾸고 위치 정보를 더한 뒤, 맨 앞에 요약용 토큰 두 개를 붙입니다. self-attention은 모든 토큰이 다른 모든 토큰과의 관련도를 계산해 정보를 섞는 과정이고, 이것을 12층 반복합니다. 그래서 멀리 떨어진 시간이나 주파수의 조각도 한 번에 연결할 수 있습니다. 마지막에 요약 토큰 두 개를 평균한 768차원 벡터를 씁니다. 이 moan 클립에서 요약 토큰은 기본 주파수가 있는 저주파 대역의 call 구간에 주로 주목했습니다.");
    const c1 = [L, 2.75], c2 = [L + 2.95, 2.2], c3 = [L + 5.35, 3.4], c4 = [L + 8.95, W - 8.95];
    hdr(s, "① 패치 나누기", c1[0], c1[1]); hdr(s, "② 패치 → 토큰", c2[0], c2[1]); hdr(s, "③ self-attention × 12층", c3[0], c3[1]); hdr(s, "④ 출력", c4[0], c4[1]);
    const yA = TOP + 0.32;
    s.addImage({ path: FX + "m_ast_patch.png", x: c1[0], y: yA, w: c1[1], h: c1[1] * 2.0 / 2.6 });
    cap(s, "log-mel 128 × 1024 (10.24 s)\n16×16 패치 · 10칸 간격(겹침)\n→ 주파수 12 × 시간 101 = **1,212개**", c1[0], yA + 2.15, c1[1], 0.85, 9.5);
    [["패치 1개 = 16×16 = 256개 값", X.white], ["선형 변환 → 768차원 토큰", X.sky], ["+ 위치 정보 (시간·주파수 위치)", X.sky], ["+ 요약용 토큰 2개 → 총 1,214개", X.pale]].forEach(([t, f], k) => { const y = yA + 0.05 + k * 0.62; sbox(s, c2[0], y, c2[1], 0.48, t, f, { fs: 9.5 }); if (k < 3) darr(s, c2[0] + c2[1] / 2, y + 0.55); });
    cap(s, "요약용 토큰: CLS · distillation\n(DeiT 구조에서 물려받음)", c2[0], yA + 2.55, c2[1], 0.5, 9);
    s.addImage({ path: FX + "m_ast_full.png", x: c3[0], y: yA, w: c3[1], h: c3[1] * 0.95 / 3.0 });
    s.addImage({ path: FX + "m_ast_roll.png", x: c3[0], y: yA + 1.12, w: c3[1], h: c3[1] * 0.95 / 3.0 });
    T(s, "입력 (2.16 s 클립을 10.24 s로 반복)", c3[0] + 0.45, yA + 0.02, 2.9, 0.22, { fontSize: 8.5, color: X.navy, bold: true });
    T(s, "요약 토큰이 주목한 패치 (attention rollout)", c3[0] + 0.45, yA + 1.14, 2.9, 0.22, { fontSize: 8.5, color: X.white, bold: true });
    cap(s, "각 토큰이 **1,214개 토큰 모두**와의 관련도를 계산해 정보를 섞음\n(12개 head 병렬) → MLP → **12층 반복**\n→ 멀리 떨어진 시간·주파수 조각도 한 번에 연결", c3[0], yA + 2.25, c3[1], 0.85, 9.5);
    sbox(s, c4[0], yA + 0.05, c4[1], 0.5, "요약 토큰 2개 평균 → **768차원**", X.navy, { color: X.white, fs: 10 });
    s.addImage({ path: FX + "m_out_ast.png", x: c4[0], y: yA + 0.65, w: c4[1], h: 0.6 });
    sbox(s, c4[0], yA + 1.35, c4[1], 0.55, "PCA ≤ 64 → linear probe\n→ 수염고래류 **100%** (이 클립)", X.pale, { fs: 9.5 });
    cap(s, "사전학습: ImageNet(사진) 초기화 → AudioSet 약 200만 클립 · 527개 소리 범주\n본 연구: 가중치 고정(frozen)", c4[0], yA + 2.05, c4[1], 0.9, 9);
    [[c2[0], yA + 1.0], [c3[0], yA + 1.0], [c4[0], yA + 1.0]].forEach(([xx, yy]) => arrow(s, xx - 0.1, yy));
    band(s, ["강점: 하모닉(주파수 방향)과 call 윤곽(시간 방향)처럼 **멀리 떨어진 조각의 관계**를 한 번에 연결", "본 연구에서 출처를 분리해도 분류군 성능이 가장 높음 (0.79, baseline 0.68)"],
      ["한계: 입력이 **0–8 kHz · 10.24 s로 고정**, 짧은 클립은 반복해 채움", "이 클립의 주목 지점: **moan 기본 주파수 대역의 call 구간** (attention 지도는 **해석 보조**이며 판단 근거로 단정할 수 없음)"]); }

  /* ③-2 VGGish */
  { const s = slide("부록 A2  VGGish (CNN)", "VGGish는 0.96초 패치마다 합성곱으로 국소 패턴을 찾고, 패치별 128차원 벡터를 평균함", "Hershey et al. 2017, torchaudio 공개 가중치. 특징맵 = 각 pooling 뒤 채널 평균(이 클립의 실제 값). 본 연구는 마지막 ReLU 이전 128차원 사용. 그림은 moan이 든 두 번째 패치. 예시: 8101403Z 북대서양긴수염고래 moan (14.9 kHz, 2.16 s).",
      "VGGish는 사진 분류용 VGG 구조를 소리에 맞춘 합성곱 신경망입니다. 16 kHz 소리를 0.96초씩 잘라 64개 mel 대역의 스펙트로그램 패치로 만들고, 3×3 합성곱으로 작은 시간–주파수 패턴을 찾습니다. pooling 때마다 크기는 절반으로 줄고 패턴 종류는 늘어나 마지막에는 6×4 크기에 512종이 됩니다. 이것을 펼쳐 완전연결층 세 개를 지나면 패치마다 128차원 벡터가 나오고, 클립 안의 패치들을 평균해 씁니다. 0.96초 안의 국소 패턴만 보고, 평균을 내기 때문에 시간 순서 정보는 남지 않습니다.");
    const yA = TOP + 0.32, c1 = [L, 1.95], c2 = [L + 2.15, 4.95], c3 = [L + 7.3, 1.85], c4 = [L + 9.35, W - 9.35];
    hdr(s, "① 입력 패치", c1[0], c1[1]); hdr(s, "② 합성곱 4단 (Conv + pooling)", c2[0], c2[1]); hdr(s, "③ 완전연결(FC) 3층", c3[0], c3[1]); hdr(s, "④ 출력", c4[0], c4[1]);
    s.addImage({ path: FX + "m_vgg_in96.png", x: c1[0], y: yA, w: c1[1], h: c1[1] * 1.25 / 1.7 });
    cap(s, "16 kHz → log-mel 64 대역\n(125–7,500 Hz)\n25 ms 창 · 10 ms 간격\n**0.96 s = 96 프레임**", c1[0], yA + 1.5, c1[1], 1.1, 9.5);
    const B = [["Conv 64\n+ pool", "48 × 32 × 64", 1.0], ["Conv 128\n+ pool", "24 × 16 × 128", 0.82], ["Conv 256 ×2\n+ pool", "12 × 8 × 256", 0.66], ["Conv 512 ×2\n+ pool", "6 × 4 × 512", 0.5]];
    const bw = (c2[1] - 3 * 0.2) / 4;
    B.forEach(([t, sz, sc], k) => { const x = c2[0] + k * (bw + 0.2); sbox(s, x, yA + 0.02, bw, 0.55, t, k % 2 ? X.sky : X.pale, { fs: 9 });
      const iw = bw * sc, ih = iw * 0.75; s.addImage({ path: FX + `vgg_m${k}.png`, x: x + (bw - iw) / 2, y: yA + 0.68 + (bw * 0.75 - ih) / 2, w: iw, h: ih });
      T(s, sz, x, yA + 0.72 + bw * 0.75, bw, 0.25, { fontSize: 9, align: "center", color: X.gray });
      if (k > 0) arrow(s, x - 0.1, yA + 0.3); });
    cap(s, "3×3 합성곱: 작은 시간–주파수 패턴 검출 → pooling으로 크기 절반\n단계마다 **해상도는 줄고 패턴 종류(채널)는 늘어남**\n(그림: 각 단계 특징맵의 채널 평균)", c2[0], yA + 2.05, c2[1], 0.9, 9.5);
    [["펼치기 6×4×512 = 12,288", X.white], ["FC 4,096", X.sky], ["FC 4,096", X.sky], ["FC 128", X.navy]].forEach(([t, f], k) => { const y = yA + 0.02 + k * 0.6; sbox(s, c3[0], y, c3[1], 0.45, t, f, { fs: 9.5, color: f === X.navy ? X.white : X.dark, bold: f === X.navy }); if (k < 3) darr(s, c3[0] + c3[1] / 2, y + 0.52); });
    cap(s, "FC = 앞 단계 값 **전부**에 가중치를 곱해 더한 값\n4,096개 = 서로 다른 가중합 4,096개\n→ 국소 패턴들을 **조합해 패치 전체를 요약**", c3[0] - 0.05, yA + 2.35, c3[1] + 0.25, 0.75, 8.5);
    sbox(s, c4[0], yA + 0.02, c4[1], 0.5, "패치별 128 → 클립 평균 → **128차원**", X.navy, { color: X.white, fs: 9.5 });
    s.addImage({ path: FX + "m_out_vgg.png", x: c4[0], y: yA + 0.62, w: c4[1], h: 0.6 });
    sbox(s, c4[0], yA + 1.32, c4[1], 0.55, "PCA ≤ 64 → linear probe\n→ 수염고래류 **100%** (이 클립)", X.pale, { fs: 9.5 });
    cap(s, "사전학습: YouTube 영상 오디오의\n소리 종류 분류 (Hershey et al. 2017)\n본 연구: 가중치 고정(frozen)", c4[0], yA + 2.0, c4[1], 0.9, 9);
    [[c2[0], yA + 0.3], [c3[0], yA + 0.3], [c4[0], yA + 0.3]].forEach(([xx, yy]) => arrow(s, xx - 0.1, yy));
    band(s, ["보는 범위: **0.96 s 패치 하나 안의 국소 패턴** → 클립 평균이라 **시간 순서 정보는 남지 않음**", "강점: click·짧은 call 조각 같은 국소 질감에 민감"],
      ["한계: **125–7,500 Hz만** 봄 → 8 kHz 위 신호(whistle·click)는 입력에 없음", "출처 분리 시 분류군 0.63으로 메타데이터 baseline(0.68)과 차이 없음"]); }

  /* ③-3 EfficientNet-B3 */
  { const s = slide("부록 A3  EfficientNet-B3 (CNN)", "EfficientNet-B3는 0–20 kHz 스펙트로그램을 사진처럼 받아 26개 블록으로 특징을 뽑음", "Tan & Le 2019, torchvision ImageNet 가중치. 원 이미지 전처리 코드는 확보하지 못해 ①·②의 그림은 구조 설명용 재계산(저장 벡터와 cosine 0.76). ④의 출력·예측은 저장된 벡터 기준. 예시: 8101403Z 북대서양긴수염고래 moan (14.9 kHz, 2.16 s).",
      "EfficientNet-B3는 사진 분류용 합성곱 신경망입니다. 소리를 40 kHz로 바꾼 뒤 5초 창의 선형 주파수 스펙트로그램을 300×800 크기 이미지로 만들어 넣습니다. 그래서 0–20 kHz를 보지만, 이 moan은 14.9 kHz 녹음이라 7.45 kHz 위는 비어 있습니다. 내부는 MBConv라는 블록 26개가 7단계로 쌓여 있습니다. 블록 하나는 1×1로 채널을 늘리고, 채널별로 따로 합성곱한 뒤, 채널마다 중요도를 다시 매기고, 1×1로 줄이는 구조입니다. 단계를 지날수록 해상도는 1/32까지 줄고, 마지막에 1,536개 채널을 전체 평균해 벡터를 만듭니다. 마지막 특징맵에서도 moan 위치가 남아 있습니다.");
    const yA = TOP + 0.32, c1 = [L, 2.7], c2 = [L + 2.9, 5.3], c3 = [L + 8.4, W - 8.4];
    hdr(s, "① 입력 이미지", c1[0], c1[1]); hdr(s, "② MBConv 블록 26개 (7단계)", c2[0], c2[1]); hdr(s, "③ 출력", c3[0], c3[1]);
    s.addImage({ path: FX + "m_eff_img.png", x: c1[0], y: yA, w: c1[1], h: c1[1] * 1.2 / 2.6 });
    cap(s, "40 kHz STFT (1,024점 · 256 간격)\n5 s 창 → **300 × 800 이미지 (0–20 kHz)**\n짧은 클립은 반복, 3채널 복제 · ImageNet 정규화", c1[0], yA + 1.35, c1[1], 0.95, 9.5);
    const M = [["1×1 확장", X.white], ["depthwise k×k\n(채널별 합성곱)", X.sky], ["SE\n(채널 중요도)", X.sky], ["1×1 축소\n+ 건너뛰기", X.white]];
    const mw = (c2[1] - 3 * 0.2) / 4;
    M.forEach(([t, f], k) => { const x = c2[0] + k * (mw + 0.2); sbox(s, x, yA + 0.02, mw, 0.55, t, f, { fs: 9 }); if (k > 0) arrow(s, x - 0.1, yA + 0.3); });
    T(s, "MBConv 블록 1개의 구조", c2[0], yA + 0.6, c2[1], 0.22, { fontSize: 9, color: X.gray, align: "center" });
    const E = [["eff_m0.png", "stem\n150 × 400 × 40"], ["eff_m3.png", "3단계\n38 × 100 × 48"], ["eff_m4.png", "4단계\n19 × 50 × 96"], ["eff_m8.png", "head\n10 × 25 × 1,536"]];
    E.forEach(([f, t], k) => { const x = c2[0] + k * (mw + 0.2); s.addImage({ path: FX + f, x, y: yA + 0.9, w: mw, h: mw * 0.55 });
      T(s, t, x, yA + 0.95 + mw * 0.55, mw, 0.42, { fontSize: 9, align: "center", color: X.gray }); if (k > 0) arrow(s, x - 0.1, yA + 0.9 + mw * 0.275); });
    cap(s, "단계를 지날수록 **해상도는 1/32로 줄고 채널은 1,536개로 늘어남**\n(그림: 각 단계 특징맵의 채널 평균 · head에서도 moan 위치가 남아 있음)", c2[0], yA + 2.35, c2[1], 0.7, 9.5);
    sbox(s, c3[0], yA + 0.02, c3[1], 0.5, "전체 평균(GAP) → **1,536차원**", X.navy, { color: X.white, fs: 9.5 });
    s.addImage({ path: FX + "m_out_eff.png", x: c3[0], y: yA + 0.62, w: c3[1], h: 0.6 });
    sbox(s, c3[0], yA + 1.32, c3[1], 0.55, "PCA ≤ 64 → linear probe\n→ 수염고래류 **100%** (이 클립)", X.pale, { fs: 9.5 });
    cap(s, "사전학습: ImageNet 사진 128만 장 · 1,000 범주\n본 연구: backbone만 사용, 가중치 고정 (Kim et al. 2026의 검출 head는 미사용)", c3[0], yA + 2.0, c3[1], 1.0, 9);
    [[c2[0], yA + 0.3], [c3[0], yA + 0.3]].forEach(([xx, yy]) => arrow(s, xx - 0.1, yy));
    band(s, ["보는 범위: **0–20 kHz · 5 s** → 넓지만, 14.9 kHz 녹음은 **7.45–20 kHz가 빈 채로** 입력됨", "학습: 소리가 아니라 **사진의 질감·형태**로 학습한 특징"],
      ["한계: 주파수 축을 사진의 세로축처럼 취급 → **주파수 위치 변화에 둔감**할 수 있음", "빈 윗대역의 넓이 = **샘플레이트 정보** → 녹음 조건 단서가 될 수 있음"]); }

  /* ③-4 MFCC */
  { const s = slide("부록 A4  MFCC (고정 계산)", "MFCC는 학습 없이 스펙트럼 모양을 20개 계수로 요약하고, 시간 통계 80개로 클립을 표현함", "Davis & Mermelstein 1980. 본 연구 설정: 16 kHz, mel 64 대역(50–8,000 Hz), c1–c20 + Δ, 평균·표준편차 = 80차원. 그림은 이 클립의 실제 중간값. 예시: 8101403Z 북대서양긴수염고래 moan (14.9 kHz, 2.16 s).",
      "MFCC는 학습이 필요 없는 고정 계산으로, 음성 인식에서 오래 써 온 특징입니다. 16 kHz 소리를 25 ms 창으로 10 ms씩 옮기며 파워 스펙트럼을 구하고, 이를 mel 간격의 삼각 필터 64개로 묶습니다. 저주파는 촘촘하게, 고주파는 넓게 묶는 방식입니다. log로 음량 차이를 줄인 뒤 DCT로 스펙트럼 모양을 계수로 바꾸고, 전체 레벨인 c0는 빼고 c1–c20만 남깁니다. 마지막으로 계수와 그 변화량의 시간 평균과 표준편차를 내면 80개 값이 됩니다. 스펙트럼의 전체 모양, 즉 음색은 잘 요약하지만, 시간 평균 때문에 주파수 윤곽 정보는 대부분 사라집니다.");
    const yA = TOP + 0.32, n = 5, g = 0.2, pw = (W - (n - 1) * g) / n;
    const S5 = [["① 프레임 스펙트럼", "mf_pow.png", 1.0 / 1.6, "16 kHz · 25 ms 창 · 10 ms 간격\n프레임별 파워 스펙트럼 (0–8 kHz)"],
      ["② mel 필터 64개", "mf_fb.png", 1.0 / 1.6, "50–8,000 Hz를 mel 간격 삼각 필터로 묶음\n(저주파 촘촘, 고주파 넓게)"],
      ["③ log-mel", "mf_logmel.png", 1.0 / 1.6, "log 변환으로 음량 차이 압축\n64 대역 × 프레임"],
      ["④ DCT → c1–c20", "mf_c.png", 1.0 / 1.6, "스펙트럼 모양을 20개 계수로 요약\n(c0 = 전체 레벨은 제외)"],
      ["⑤ 시간 통계 → 80", "mf_out.png", 0.75 / 1.8, "c, Δc의 시간 평균 · 표준편차\n4 × 20 = **80차원**"]];
    S5.forEach(([h, f, r, c], k) => { const x = L + k * (pw + g); sbox(s, x, yA, pw, 0.42, `**${h}**`, k === 4 ? X.navy : X.sky, { fs: 10, color: k === 4 ? X.white : X.dark });
      const ih = Math.min(pw * r, 1.45), iw = ih / r; s.addImage({ path: FX + f, x: x + (pw - iw) / 2, y: yA + 0.52, w: iw, h: ih });
      cap(s, c, x, yA + 2.05, pw, 0.75, 9); if (k > 0) arrow(s, x - g / 2, yA + 0.21); });
    sbox(s, L + 4 * (pw + g), yA + 2.75, pw, 0.42, "linear probe → 수염고래류 **90%**", X.pale, { fs: 9 });
    band(s, ["학습 없음 (고정 계산) · 음성 인식에서 온 표준 특징", "보는 것: **스펙트럼 포락선(음색)**의 평균적 모양"],
      ["한계: **0–8 kHz만**, 시간 평균이라 **call 윤곽·피치 변화는 대부분 사라짐**", "이 moan처럼 **하모닉이 뚜렷한 저주파 call**은 포락선 모양에 잘 드러남"]); }
{ const s = slide("부록 A5  물리 특징", "물리 특징 51개는 원 샘플레이트에서 초·Hz 단위로 정의한 6개 그룹으로 구성됨", "정의: b05_features_baseline.py phys_features() (본 연구의 초안, 문헌 방법 아님). 예시 값은 같은 코드를 다시 구현해 저장값과 51개 모두 일치함을 확인. 예시: A·B·D 5702601X 참돌고래 whistle, C·F 6700900B 하프물범 growl, E 9420108A 참거두고래 click",
    "여섯 그룹마다 실제 클립 하나로 무엇을 재는지 보여 줍니다. A는 주파수별 band SNR이 9.4 dB를 넘는 부분으로 신호 대역을 정하고, B는 같은 값을 1/3 옥타브 대역별로 요약합니다. C는 시간에 따라 활동 구간을 묶고, D는 활동 프레임의 최대 주파수 궤적으로 음조성을 잽니다. E는 1 ms 포락선에서 click을 찾고, F는 활동 패턴의 반복 주기를 잽니다. E 예시를 보면 강한 click 사이에 약한 이중 펄스가 함께 잡혀서 click 간격 중앙값이 89 ms로 나옵니다. 정의를 다시 점검해야 할 부분입니다.");
  const GP = [["A  Signal band", 7, "pf_A.png", ["f05 · f50 · f95", "bandwidth (oct)", "band occupancy", "peak band SNR (dB)", "active T–F fraction"]],
    ["B  1/3-octave levels", 31, "pf_B.png", ["band SNR excess per 1/3-octave band", "63 Hz – 63 kHz", "bands above Nyquist = 0"]],
    ["C  Temporal", 4, "pf_C.png", ["duty cycle", "event rate (/s)", "event duration (median)", "event duration (max)"]],
    ["D  Tonality", 3, "pf_D.png", ["tonal concentration", "FM rate (oct/s)", "peak-frequency range (IQR, oct)"]],
    ["E  Impulsiveness", 4, "pf_E.png", ["envelope kurtosis", "click rate (/s)", "inter-click interval (median)", "ICI coefficient of variation"]],
    ["F  Repetition", 2, "pf_F.png", ["autocorrelation lag (s)", "autocorrelation strength"]]];
  const g = 0.14, cw = (W - 5 * g) / 6, ih = cw / (2.0 / 1.8);
  for (let i = 0; i < 6; i++) { const x = L + i * (cw + g), [h, n, f, items] = GP[i];
    rect(s, x, TOP, cw, HEAD, X.navy);
    T(s, [{ text: h, options: { bold: true, color: X.white, fontSize: 11.5, breakLine: true } }, { text: `${n} features`, options: { color: X.sky, fontSize: 9.5 } }], x + 0.1, TOP, cw - 0.15, HEAD, { valign: "middle" });
    rect(s, x, TOP + HEAD, cw, BOT - TOP - HEAD, X.white, X.line);
    s.addImage({ path: FIG + f, x: x + 0.02, y: TOP + HEAD + 0.05, w: cw - 0.04, h: ih - 0.03, objectName: f });
    s.addText(items.map((t, j) => ({ text: t, options: { bullet: { indent: 9 }, breakLine: j < items.length - 1, paraSpaceAfter: 2 } })), { x: x + 0.08, y: TOP + HEAD + ih + 0.1, w: cw - 0.14, h: BOT - (TOP + HEAD + ih + 0.1) - 0.08, fontSize: 10, color: X.dark, valign: "top", margin: 0, isTextBox: true }); }
  band(s, ["**band SNR** = 10·log₁₀(P95/P50): 주파수 칸마다 시간 방향 파워의 95번째와 50번째 백분위 비 (본 연구의 정의)", "기준 9.4 dB = 잡음만 있을 때의 기대값 6.4 dB + 3 dB. 분석 창은 32 ms(초 단위)로 고정하고 샘플 수를 샘플레이트에 맞춤"],
    ["개별로 쓰는 값: **f50 · tonal concentration · inter-click interval**(결과 ⑥, 고찰)", "한계: B는 나이퀴스트 이상을 0으로 둬 **샘플레이트 정보가 섞임** / E는 약한 이중 펄스까지 세어 **ICI가 짧게 측정**될 수 있음"]); }
{ const s = slide("부록 A6  선행연구 방법 재현", "선행연구 4편의 방법을 단계별로 대응시켜, 재현한 부분과 재현하지 않은 부분을 구분함", "○ 재현, △ 바꿔서 적용, × 미재현. 대상: 개발용 출처 29개, 2,769클립 (부록 A7). EMD(empirical mode decomposition, 경험적 모드 분해): 신호를 진동 성분(IMF)으로 나누는 적응형 분해로, Liang et al.은 잡음 제거에 사용. Liang et al. 4종 선택은 원문 대조 필요. RF 등 세부 설정은 이전 실행 코드를 따름. 모든 절차에 여섯 표현(메타데이터 baseline 포함)을 모두 적용",
    "선행연구 네 편을 단계별로 나눠, 왼쪽 원 논문의 항목과 오른쪽 본 연구의 항목이 한 줄씩 대응되게 했습니다. 마지막 열은 재현 정도입니다. MFCC는 Liang et al.의 특징 그대로가 아니라 일반 설정의 MFCC를 썼고, 그 앞단의 EMD 잡음 제거와 endpoint detection은 하지 않았습니다. EfficientNet-B3는 Kim et al.의 backbone을 그대로 특징 추출기로 쓰고, 검출 head는 쓰지 않았습니다.");
  const St = (t, y) => ({ text: [{ text: t, options: { bold: true, color: X.navy, breakLine: true } }, { text: y, options: { color: X.gray, fontSize: 9.5 } }], options: { rowspan: 3, valign: "middle" } });
  const St2 = (t, y) => ({ ...St(t, y), options: { rowspan: 2, valign: "middle" } });
  const Sg = (t) => ({ text: t, options: { color: X.gray, align: "center" } }), R = (t) => ({ text: t, options: { align: "center", bold: true, color: t === "×" ? X.gray : X.navy } });
  s.addTable([[HD("선행연구"), HD("단계"), HD("원 논문"), HD("본 연구"), HD("재현")],
    [St("Cominelli et al. 2024", "VGGish · UMAP · RF"), Sg("특징"), "VGGish, 0.96초 단위 임베딩 (12종)", "VGGish, 클립 평균 벡터 128차원 (36종)", R("△")],
    [Sg("시각화"), "비지도 UMAP, 8/12종 군집 형성 (정성 평가)", "UMAP (결과 ④), 군집 분리도를 수치로 산출", R("△")],
    [Sg("분류"), "balanced RF, Placentia Bay 자료에 적용", "balanced RF, Watkins 분류군·27종, 랜덤 5-fold", R("△")],
    [St("Lipari et al. 2026", "AST · metric learning"), Sg("특징"), "AST + metric learning fine-tuning", "AST frozen (fine-tuning 없음), 768차원", R("△")],
    [Sg("분류"), "클립 70/15/15 분할, 종 정확도 92.4% (frozen 73.1%)", "prototype matching (cosine), 클립 85/15 분할, 29종", R("△")],
    [Sg("추가 평가"), "open-set · few-shot", "–", R("×")],
    [St("Liang et al. 2025", "EMD · MFCC · k-means"), Sg("전처리"), "EMD 잡음 제거 + endpoint detection", "없음 (클립 전체 사용)", R("×")],
    [Sg("특징"), "MFCC (논문 설정)", "MFCC 일반 설정 (16 kHz, 20계수 + delta, 평균·표준편차)", R("△")],
    [Sg("군집"), "k-means · GMM, 2–5종, 평균 84.8%", "k-means (k = 종 수), 4종 중 2–4종 조합 전부", R("○")],
    [St2("Kim et al. 2026", "EfficientNet-B3 · YOLO"), Sg("특징"), "EfficientNet-B3 backbone (frozen), 40 kHz linear spectrogram", "같은 backbone, 5초 창 global average 1,536차원 (클립 벡터화는 본 연구 구성)", R("○")],
    [Sg("검출"), "YOLO 검출 head, 합성 혼합음 4종 F-score 0.82–1.00", "–", R("×")]],
    tableOpt({ y: TOP, colW: [1.95, 0.9, 4.15, 4.35, 0.78], rowH: [0.36, ...Array(11).fill((6.8 - TOP - 0.36) / 11)], fontSize: 10.5, margin: [0.02, 0.07, 0.02, 0.07] })); }

{ const s = slide("부록 A7  선행연구 절차 결과", "랜덤 분할에서는 선행연구 성능이 재현되나, 출처 단위 분할에서는 chance 수준으로 하락함", "개발용 출처 29개, 2,769클립. RF: BA, prototype matching: accuracy, k-means: 군집 정확도. 신뢰구간 미산출. 하늘색 셀: baseline이 추출기와 동등 이상. 원 논문 값: Lipari et al. 2026 frozen 73.1%, Liang et al. 2025 84.8%",
    "선행연구 방식대로 랜덤 분할로 평가하면 선행연구와 비슷한 값이 나옵니다. 출처 단위 분할에서는 27종 RF 분류가 모든 추출기에서 0.02–0.06으로 chance 수준까지 떨어지고 baseline 0.11보다 낮습니다.");
  const row = (v, hl) => v.map((x, i) => i === 0 && hl ? HL(x) : N(x)); const M = (t) => ({ text: t, options: { rowspan: 2, bold: true, color: X.navy, valign: "middle" } });
  s.addTable([[HD("평가 방식"), HD("분할"), HD("Baseline"), HD("MFCC"), HD("물리 특징"), HD("VGGish"), HD("EffNet-B3"), HD("AST")],
    [M("Random forest · 분류군\n(Cominelli et al. 2024 분류기)"), N("랜덤"), ...row(["0.94", "0.83", "0.88", "0.79", "0.90", "0.81"], true)], [N("출처 단위"), ...row(["0.72", "0.56", "0.61", "0.49", "0.63", "0.56"], true)],
    [M("Random forest · 27종\n(Cominelli et al. 2024 분류기)"), N("랜덤"), ...row(["0.52", "0.80", "0.69", "0.67", "0.72", "0.69"])], [N("출처 단위"), ...row(["0.11", "0.02", "0.04", "0.03", "0.06", "0.04"], true)],
    [M("Prototype matching · 29종\n(Lipari et al. 2026)"), N("랜덤"), ...row(["0.10", "0.41", "0.30", "0.40", "0.53", "0.58"])], [N("출처 단위"), ...row(["0.10", "0.07", "0.08", "0.07", "0.14", "0.12"])],
    [B("k-means · 2–4종 (Liang et al. 2025)"), N("–"), ...row(["0.70", "0.66", "0.56", "0.65", "0.78", "0.83"], true)]],
    tableOpt({ colW: [3.43, 1.3, 1.2, 1.2, 1.25, 1.2, 1.3, 1.25], rowH: (BOT - TOP) / 8, margin: 0.05, fontSize: 12 }));
  band(s, ["랜덤 분할에서는 선행연구와 유사한 수준이 재현됨 (구현상 문제 아님)", "출처 단위 분할 시 27종 분류는 **0.02–0.06**으로 baseline(0.11)보다 낮음"],
    ["랜덤 분할에서 분류군은 **baseline(0.94)이 가장 높음**", "Cominelli et al. 2024는 RF를 Placentia Bay 자료에 적용 → 본 분석은 같은 분류기를 Watkins에 적용한 정량 비교"]); }
{
  const SRC = "단일 라벨 6종 1,456클립 · 출처 40개. 출처 단위 7-fold, 6클래스 linear probe. 3유형 = 6클래스 예측을 묶은 값(결과 ④의 3클래스 분류기와 다름). within type = 유형을 맞힌 클립 중 세부 콜 정답률. 유의 = baseline 대비 차이의 95% CI(출처 bootstrap)가 0 미포함. 라벨은 메모 기반";
  const P6 = [["meta", "부록 A8  콜 타입 6종 · 메타데이터 baseline", "녹음 조건만으로도 6종 콜 타입을 BA 0.38로 맞힘: 샘플레이트가 고·저주파 콜을 가름",
      ["출처 분리 6종 BA **0.38** (chance 0.17), 3유형 0.45", "whistle의 65%가 click·pulse train으로 → **고샘플레이트 콜끼리 묶임** (tonal·impulsive 구분 못함)"],
      ["moan·growl·pulsed call은 **저샘플레이트 콜끼리 서로 섞임**", "유형을 맞히면 세부 콜 정답률 0.77 → whistle vs moan처럼 **대역이 다른 쌍은 샘플레이트로 갈림** (다음 쪽들의 기준선)"]],
    ["physv0", "부록 A9  콜 타입 6종 · 물리 특징", "물리 특징은 3유형에서 baseline을 넘지만(0.61), 6종에서는 유의한 차이가 없음(0.46)",
      ["출처 분리 6종 **0.46** (+0.08, CI −0.00 ~ +0.19) · 3유형 **0.61 (+0.16, 유의)**", "whistle 재현율 0.19 → **0.54**: tonal concentration 등으로 click과 구분하기 시작"],
      ["moan → growl·pulsed call, growl → moan(47%): **저주파 콜끼리는 여전히 섞임**", "랜덤 분할 6종 0.66 → 출처 분리 0.46 (0.20 하락)"]],
    ["mfcc16k", "부록 A10  콜 타입 6종 · MFCC", "MFCC는 6종 BA 0.34로 baseline보다 낮음: 포락선만으로는 콜 타입을 구분하지 못함",
      ["출처 분리 6종 **0.34** (−0.04) · 3유형 0.51 (+0.05) → **모두 유의한 차이 없음**", "pulse train 재현율 **0.30**: click·whistle과 혼동"],
      ["pulsed call의 39%가 whistle로 → 하모닉 구조를 포락선으로는 구분 못함", "랜덤 분할 0.53 → 출처 분리 0.34: 남은 정보 대부분이 **같은 출처를 알아보는 정보**"]],
    ["vggish", "부록 A11  콜 타입 6종 · VGGish", "VGGish도 6종 BA 0.37로 baseline 수준이며, pulse train·pulsed call 재현율이 낮음",
      ["출처 분리 6종 **0.37** (−0.01) · 3유형 0.50 (+0.04) → **유의한 차이 없음**", "pulse train 0.38 · pulsed call 0.19 (39%가 whistle로)"],
      ["moan의 31%가 growl로: 저주파 콜 혼동은 다른 표현과 같음", "랜덤 분할 0.59 → 출처 분리 0.37 (0.22 하락)"]],
    ["effb3", "부록 A12  콜 타입 6종 · EfficientNet-B3", "EfficientNet-B3가 가장 높음: 6종 BA 0.55 (baseline +0.16), whistle 재현율 0.81",
      ["출처 분리 6종 **0.55 (+0.16, 유의)** · 3유형 **0.69 (+0.24, 유의)**", "whistle **0.81** · moan 0.62: **0–20 kHz를 입력으로 보는 유일한 신경망 표현**"],
      ["growl은 0.26으로 여전히 낮음 (moan 34% · pulsed call 27%로)", "단, 샘플레이트와 얽힌 효과일 수 있음 → **샘플레이트 통제 분석(결과 ④)과 함께 해석**"]],
    ["ast_tile", "부록 A13  콜 타입 6종 · AST", "AST는 6종 BA 0.48로 baseline을 넘고, pulsed call·growl 재현율이 가장 높음",
      ["출처 분리 6종 **0.48 (+0.09, 유의)** · 3유형 **0.62 (+0.17, 유의)**", "pulsed call **0.39** · growl **0.32**: 여섯 표현 중 가장 높음"],
      ["whistle은 0.60으로 EfficientNet-B3(0.81)보다 낮음: **8 kHz 위 whistle 정보 손실**과 일관", "공통 대역(0–4 kHz) 검증에서도 분류군 성능 유지 (부록 A14)"]]];
  for (const [f, tag, title, bl, br] of P6) {
    const s = slide(tag, title, SRC, "왼쪽은 출처를 분리했을 때, 가운데는 클립을 랜덤으로 나눴을 때의 혼동 행렬입니다. 행은 메모 기반 정답, 열은 예측이고 숫자는 행 기준 백분율입니다. 주황색 사각형은 같은 신호 유형 묶음입니다. 오른쪽은 6종, 3유형, 유형 안 세부 콜의 균형 정확도를 baseline과 비교한 것이고, 회색 점은 다른 표현들입니다. " + bl.concat(br).join(" ").replace(/\*\*/g, ""));
    await fig(s, "../fig_x/ct6_" + f + ".png");
    band(s, bl, br); }
}

{ const s = slide("부록 A14  공통 대역 검증", "윗대역을 지워 0–4 kHz로 맞춰도 출처 단위 결과는 바뀌지 않음", "출처 단위 7-fold, 95% CI는 출처 단위 bootstrap 2,000회. 오른쪽 숫자 = 공통 대역 − 표준 입력 (* 차이의 CI가 0 미포함). 랜덤 분할에서는 세 추출기 모두 종 분류가 0.06–0.07 하락. 표준 입력은 다시 계산해 저장값과 일치(cosine 1.000) 확인",
    "16 kHz 입력 모델에서는 저샘플레이트 녹음의 윗대역이 비어 있어서, 그 빈 대역 자체가 샘플레이트를 알려 주는 단서가 될 수 있습니다. 이를 확인하려고 모든 클립을 8 kHz로 내렸다가 16 kHz로 올려 0–4 kHz만 남기고, AST, VGGish, MFCC를 같은 조건에서 다시 계산했습니다. 결과적으로 AST의 분류군 성능은 0.786에서 0.784로 거의 그대로였습니다. 즉 AST의 분류군 정보는 빈 윗대역 단서에서 나온 것이 아닙니다. MFCC와 VGGish는 조금 떨어졌지만, 어느 쪽이든 메타데이터 baseline을 넘지 못한다는 결론은 같습니다. 다만 4–8 kHz의 실제 신호도 함께 지워지므로, 떨어진 부분이 단서가 사라져서인지 정보가 사라져서인지는 이 실험만으로 가를 수 없습니다.");
  await fig(s, "fig_b4k.png");
  band(s, ["조건: 모든 클립을 8 kHz로 내렸다가 16 kHz로 올림 → **0–4 kHz만 남겨 빈 윗대역 단서 제거** (fs < 8 kHz 128클립 제외, 3,411클립)", "같은 모델·같은 linear probe·같은 fold에서 표준 입력과 비교"],
    ["AST 분류군 **0.786 → 0.784** (차이 0.00 [−0.05, +0.05]) → **빈 윗대역 단서가 아님**", "MFCC −0.10*, VGGish −0.04로 소폭 하락, 어느 조건에서도 baseline(0.74)을 넘지 못함 → **결과 ②의 결론 유지** (4–8 kHz 신호도 함께 지워져 하락 원인은 구분 불가)"]); }

{ const s = slide("부록 A15  분류군별 샘플레이트", "수염고래류와 이빨고래류는 샘플레이트가 거의 겹치지 않음 (공존 계열은 20 kHz뿐)", "분석 대상 10,694클립. (a) 막대 높이는 분류군별로 최댓값에 맞춤, 점선 = 중앙값. 16 kHz 점선은 AST·VGGish·MFCC의 입력 샘플레이트(모든 클립을 이 값으로 리샘플링)",
    "분류군별 샘플레이트 분포입니다. 수염고래류는 중앙값 10.2 kHz로 거의 전부 16 kHz 아래이고, 이빨고래류는 중앙값 80 kHz로 거의 전부 위입니다. 기각류만 그 사이에 걸쳐 있습니다. 그래서 샘플레이트 하나만으로도 수염고래류와 이빨고래류를 상당히 가를 수 있고, 같은 샘플레이트 안에서 두 분류군을 비교할 수 있는 자료는 거의 없습니다.");
  await fig(s, "../fig_x/taxon_fs.png");
  band(s, ["수염고래류: 중앙값 **10.2 kHz**, 97.5%가 16 kHz 미만 (10·14.9 kHz에 집중)", "이빨고래류: 중앙값 **80 kHz**, 16 kHz 미만은 **0.1%**"],
    ["기각류: 5–60 kHz, 20 kHz(42%)·40 kHz(34%)에 분포", "세 분류군이 함께 있는 계열은 **20 kHz뿐** (이빨고래류 723클립, 대부분 범고래) → 같은 샘플레이트 안의 분류군 비교 자료가 부족"]); }

await pres.writeFile({ fileName: OUT }); await applyTheme(OUT, THEME); console.log("written");
})();
