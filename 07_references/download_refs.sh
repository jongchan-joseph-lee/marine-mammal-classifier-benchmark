#!/bin/bash
# Watkins 세미나 참고문헌 일괄 다운로드 (macOS)
# 사용: 터미널에서  bash download_refs.sh
# 공개 PDF는 ~/Downloads/Watkins_refs 에 받고, 받지 못한 것(유료·봇 차단)은 브라우저로 엽니다.

DEST="$HOME/Downloads/Watkins_refs"
mkdir -p "$DEST"
cd "$DEST" || exit 1
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"

# 파일이름|URL  (공개 PDF)
PDFS=(
"01_Liang2025_AppliedSciences.pdf|https://www.mdpi.com/2076-3417/15/7/3585/pdf"
"02_Cominelli2024_EcolEvol.pdf|https://europepmc.org/articles/PMC10880131?pdf=render"
"03_Lipari2026_IEEEAccess_authorfinal.pdf|https://ricerca.ogs.it/retrieve/85bb9d63-058d-46d8-af78-41b593ab43d3/Marine_Mammals_Final.pdf"
"04_Kim2026_EcolEvol.pdf|https://europepmc.org/articles/PMC13081793?pdf=render"
"05_Hagiwara2023_BEANS.pdf|https://arxiv.org/pdf/2210.12300"
"06_Licciardi2024_WhaleNet.pdf|https://arxiv.org/pdf/2402.17775"
"07_Roberts2017_Ecography.pdf|https://www.biometrie.uni-freiburg.de/mitarbeiter/dormann/roberts-et-al-2017-ecography.pdf"
"08_Kapoor2023_Leakage.pdf|https://arxiv.org/pdf/2207.07048"
"09_Flexer2007_ISMIR.pdf|https://archives.ismir.net/ismir2007/paper/000341.pdf"
"10_Geirhos2020_Shortcut.pdf|https://arxiv.org/pdf/2004.07780"
"11_Zech2018_PLoSMed.pdf|https://journals.plos.org/plosmedicine/article/file?id=10.1371/journal.pmed.1002683&type=printable"
"12_Stowell2022_PeerJ.pdf|https://peerj.com/articles/13152.pdf"
"13_Laidre2008_EcolAppl.pdf|https://www.ourspolaire.org/images/2008QUANTIFYING-THE-SENSITIVITY-OF-ARCTIC-MARINE-MAMMALS.pdf"
"14_Gillespie2008_PAMGUARD.pdf|https://www.dcs.gla.ac.uk/~trinder/papers/Gillespie%20IOA%202008_postREview.pdf"
"15_Gong2021_AST.pdf|https://arxiv.org/pdf/2104.01778"
"16_Hershey2017_VGGish.pdf|https://arxiv.org/pdf/1609.09430"
"17_Tan2019_EfficientNet.pdf|https://arxiv.org/pdf/1905.11946"
"18_McInnes2018_UMAP.pdf|https://arxiv.org/pdf/1802.03426"
"19_Perch2_2025.pdf|https://arxiv.org/pdf/2508.04665"
)

# 파일이름|DOI 페이지  (유료이거나 PDF 직링크가 없는 것: 브라우저로 열기)
PAGES=(
"20_Sayigh2016_POMA|https://doi.org/10.1121/2.0000358"
"21_Hurlbert1984_EcolMonogr|https://doi.org/10.2307/1942661"
"22_Sturm2014_IEEE-TMM|https://doi.org/10.1109/TMM.2014.2330697"
"23_Ford1991_CanJZool|https://doi.org/10.1139/z91-206"
"24_Janik2013_JCompPhysiolA|https://doi.org/10.1007/s00359-013-0817-7"
"25_DavisMermelstein1980_MFCC|https://doi.org/10.1109/TASSP.1980.1163420"
"26_Huang1998_EMD|https://doi.org/10.1098/rspa.1998.0193"
"27_Youden1950_Cancer|https://doi.org/10.1002/1097-0142(1950)3:1%3C32::AID-CNCR2820030106%3E3.0.CO;2-3"
"28_Lipari2026_IEEE-publisher|https://doi.org/10.1109/ACCESS.2026.3714099"
)

FAILED=()
for item in "${PDFS[@]}"; do
  name="${item%%|*}"; url="${item#*|}"
  if [ -s "$name" ]; then echo "이미 있음: $name"; continue; fi
  echo "받는 중: $name"
  curl -sSL --retry 2 --max-time 120 -A "$UA" -o "$name" "$url"
  if head -c 4 "$name" 2>/dev/null | grep -q "%PDF"; then
    echo "  완료"
  else
    echo "  PDF가 아님(봇 차단·로그인 필요) -> 브라우저로 엽니다"
    rm -f "$name"; FAILED+=("$name|$url")
  fi
done

for item in "${FAILED[@]}" "${PAGES[@]}"; do
  url="${item#*|}"; open "$url"; sleep 1
done

echo ""
echo "저장 위치: $DEST"
ls -1 "$DEST"
echo "브라우저로 연 항목: ${#FAILED[@]}개(다운로드 실패) + ${#PAGES[@]}개(유료/출판사 페이지)"
open "$DEST"
