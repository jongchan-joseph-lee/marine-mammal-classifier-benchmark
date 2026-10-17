Watkins 군집 지도 — 다시 만드는 법 (2026-10-11)

필요한 것: 파이썬 3.10 이상, pip install numpy pandas scikit-learn umap-learn
입력: Watkins_Claude_Input_Package.zip 을 푼 폴더 (아래에서 PKG)

1) 목록표 변환     python adapt_real.py PKG out            → out/cuts.csv (컷은 cut_id 로만 잇는다)
2) 계산 (약 15분)  python build_maps.py --cuts out/cuts.csv --features PKG/full_cohort/features
                     --features-sample PKG/work/features --gpt-cache PKG/cached_umap3d --out out/data.json
3) 화면 만들기     python assemble.py out/data.json out/watkins_cluster_maps.html
   (한 파일짜리 HTML 이다. 브라우저에서 바로 열 수 있다.)

watkins_cluster_maps.html 은 2026-10-11 에 위 순서로 만든 결과다.
repatch.py 는 투영 좌표를 두고 군집·이웃 수치만 다시 계산할 때 쓴다.

새 컷을 더하려면: 같은 형식의 npz(cut_id, X)를 features 폴더의 파일에 합치고 cuts.csv 의 feat 열을 main 으로 바꾼 뒤 2)부터 다시 실행한다.
보류·제외 2,912컷은 특징이 없어 지도에 없다. 저주파 87컷은 특징이 있으나 주 분석과 섞지 않았다.
