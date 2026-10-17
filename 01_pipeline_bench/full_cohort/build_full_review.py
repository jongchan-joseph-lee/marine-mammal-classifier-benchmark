"""Create a Korean scientific review and figures from saved, audited results."""
from pathlib import Path
import argparse
import html
import json
import re
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output'
AN=OUT/'five_analyses/tables'
ORDER=['meta','mfcc16k','physv0','vggish','ast_zero','ast_tile']
LABEL=dict(meta='Metadata',mfcc16k='MFCC',physv0='Physical 51',vggish='VGGish',ast_zero='AST zero',ast_tile='AST tile')
PALETTE=['#6a7380','#df9c3a','#256b98','#399a80','#8c5ba8','#c25b69']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--interim',action='store_true');args=ap.parse_args()
    validation=json.loads((OUT/'provenance/final_validation.json').read_text())
    if not args.interim and not validation['complete']:raise RuntimeError('Final report requires a complete audit')
    figs=OUT/'figures';figs.mkdir(exist_ok=True)
    scores=pd.read_csv(AN/'01_task_protocol_scores.csv')
    features=[f for f in ORDER if f in set(scores.feature)]
    matched=pd.read_csv(AN/'01_matched_four_form_scores.csv')
    source=pd.read_csv(AN/'02_source_concentration.csv')
    retrieval=pd.read_csv(AN/'03_source_excluded_fit_summary.csv')
    ablation=pd.read_csv(AN/'04_ablation_probes.csv')
    stab=pd.read_csv(AN/'05_stability_summary.csv')
    sample=pd.read_csv(OUT/'stage2/probe_comparison.csv')
    labels=[LABEL[f] for f in features]
    cols=[PALETTE[ORDER.index(f)] for f in features]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','figure.dpi':150})
    def finish(fig,name):
        fig.tight_layout();path=figs/(name+'.png');fig.savefig(path,dpi=180,bbox_inches='tight',facecolor='white');plt.close(fig);return path
    fig,axs=plt.subplots(1,2,figsize=(10.2,3.35));x=np.arange(len(features));width=.36
    for ax,task,title in zip(axs,['group3','type_single'],['Three taxonomic groups','Seven note-based call forms']):
        values=[]
        for protocol,c,offset in [('filecv','#aeb9c9',-width/2),('allcv','#2b6793',width/2)]:
            v=[float(scores[(scores.feature==f)&(scores.task==task)&(scores.protocol==protocol)].BA.iloc[0])*100 for f in features]
            ax.bar(x+offset,v,width,color=c,label='Random-file CV' if protocol=='filecv' else 'Source-separated CV')
        ax.set_xticks(x,labels,rotation=25,ha='right');ax.set_ylim(0,100);ax.set_ylabel('Balanced accuracy (%)');ax.set_title(title);ax.grid(axis='y',alpha=.18)
    axs[0].legend(frameon=False,fontsize=8);f1=finish(fig,'01_task_and_source_split')
    q=matched[matched.scope.eq('40k')].set_index('feature').loc[features]
    fig,ax=plt.subplots(figsize=(9.8,3.05));ax.bar(x,q.BA.to_numpy()*100,color=cols)
    ax.vlines(x,q.BA_lo.to_numpy()*100,q.BA_hi.to_numpy()*100,color='#26374a',lw=1.5)
    ax.set_xticks(x,labels);ax.set_ylabel('Balanced accuracy (%)');ax.set_ylim(0,100);ax.set_title('40k family: four-form evaluation subset');ax.grid(axis='y',alpha=.18)
    f2=finish(fig,'01_40k_four_form')
    q=source[source.scope.eq('dev')&source.included_in_comparative_probe.eq(True)].copy().sort_values('largest_source_share',ascending=False)
    fig,ax=plt.subplots(figsize=(9.8,2.95));ax.bar(q.form,q.largest_source_share*100,color='#447d9d')
    for i,r in enumerate(q.itertuples()):ax.text(i,min(r.largest_source_share*100+2,99),f'{r.largest_source_n}/{r.n}',ha='center',fontsize=9)
    ax.set_ylim(0,109);ax.set_ylabel('Largest-source share (%)');ax.set_title('Development call labels remain concentrated in a few sources');ax.tick_params(axis='x',rotation=15);ax.grid(axis='y',alpha=.18)
    f3=finish(fig,'02_form_source_concentration')
    q=retrieval[retrieval.condition.eq('cross_source_matched')].set_index('feature').loc[features]
    fig,ax=plt.subplots(figsize=(9.8,3.25));ax.scatter(x,q.delta_meta*100,color=cols,s=65,zorder=3)
    ax.vlines(x,q.delta_meta_lo*100,q.delta_meta_hi*100,color=cols,lw=2)
    ax.axhline(0,color='#677487',ls='--',lw=1);ax.set_xticks(x,labels);ax.set_ylabel('P@5 difference vs metadata (pp)');ax.set_title('Cross-source, matched retrieval: query source excluded from preprocessing');ax.grid(axis='y',alpha=.18)
    f4=finish(fig,'03_strict_retrieval_delta')
    q=ablation[ablation.variant.isin(['all','spectral','shape','only_tonality'])].set_index('variant').loc[['all','spectral','shape','only_tonality']]
    fig,axs=plt.subplots(1,2,figsize=(10.2,3.0));xx=np.arange(len(q))
    axs[0].bar(xx-.18,q.form_BA*100,.36,label='Call forms',color='#6f5a96');axs[0].bar(xx+.18,q.group3_BA*100,.36,label='Taxonomic groups',color='#377d99')
    axs[0].set_ylabel('Source-separated development BA (%)');axs[0].set_ylim(0,100);axs[0].legend(frameon=False,fontsize=8)
    axs[1].bar(xx,q.logfs_R2,color='#ad7b4a');axs[1].set_ylim(0,1);axs[1].set_ylabel('Sampling-rate prediction R-squared')
    for ax in axs:ax.set_xticks(xx,['All 51','Spectral 38','Shape 13','Tonality 3']);ax.grid(axis='y',alpha=.18)
    f5=finish(fig,'04_feature_group_tradeoff')
    q=stab.set_index('feature').loc[[f for f in features if f in set(stab.feature)]]
    fig,axs=plt.subplots(1,2,figsize=(10.2,3.05));xx=np.arange(len(q));sl=[LABEL[f] for f in q.index]
    for ax,prefix,title in zip(axs,['cluster_ARI','NN10_Jaccard_source_macro'],['Cluster agreement after source removal','Nearest-neighbour overlap after source removal']):
        ax.scatter(xx,q[prefix+'_median'],s=45,color='#2a7291');ax.vlines(xx,q[prefix+'_q10'],q[prefix+'_q90'],color='#2a7291',lw=2)
        ax.set_ylim(0,1.05);ax.set_xticks(xx,sl,rotation=25,ha='right');ax.set_title(title,fontsize=10);ax.grid(axis='y',alpha=.18)
    f6=finish(fig,'05_source_removal_stability')
    offsets={'full_dev':-.27,'original_core_dev':-.09,'simple_random_dev_2769':.09,'taxon_proportional_dev_2769':.27}
    sc={'full_dev':'#216b95','original_core_dev':'#8a5796','simple_random_dev_2769':'#c28d3d','taxon_proportional_dev_2769':'#479980'}
    def sampling_figure(task,title,name):
        raw=sample[sample.task.eq(task)].copy();fig,ax=plt.subplots(figsize=(10,3.0))
        for cohort,offset in offsets.items():
            ys=[];lo=[];hi=[]
            for f in features:
                v=raw[(raw.feature==f)&raw.cohort.eq(cohort)].BA.to_numpy()*100
                ys.append(v.mean());lo.append(v.min());hi.append(v.max())
            ax.scatter(x+offset,ys,color=sc[cohort],s=35,label=cohort.replace('_dev','').replace('_2769','').replace('_',' '));ax.vlines(x+offset,lo,hi,color=sc[cohort],lw=1.5)
        ax.set_xticks(x,labels);ax.set_ylabel('Development source-CV BA (%)');ax.set_ylim(0,100);ax.set_title(title);ax.legend(frameon=False,fontsize=8,ncol=2);ax.grid(axis='y',alpha=.18)
        return finish(fig,name)
    f7=sampling_figure('group3','Sampling sensitivity: three taxonomic groups','sampling_sensitivity_group3')
    f7b=sampling_figure('type_single','Sampling sensitivity: seven note-based call forms','sampling_sensitivity_forms')
    weighted=pd.read_csv(OUT/'stage2/structure_comparison.csv')
    fig,axs=plt.subplots(1,2,figsize=(10.2,2.8))
    for condition,offset,c,title in [('full_eligible',-.2,'#286d94','Unweighted'),('full_source_equal_kmeans',0,'#9669a4','Equal source weight'),('full_taxon_equal_kmeans',.2,'#c99549','Equal taxon weight')]:
        q=weighted[weighted.cohort.eq(condition)].set_index('feature').loc[features]
        for ax,col in zip(axs,['form_AMI','source_AMI']):ax.scatter(x+offset,q[col],s=30,color=c,label=title)
    for ax,title in zip(axs,['Call-form / cluster agreement','Source / cluster agreement']):
        ax.set_xticks(x,labels,rotation=25,ha='right');ax.set_ylabel('Adjusted mutual information');ax.set_title(title,fontsize=10);ax.grid(axis='y',alpha=.18)
    axs[0].legend(frameon=False,fontsize=7,ncol=1)
    f7c=finish(fig,'sampling_weighted_cluster_sensitivity')
    # Publication figures are static exports. UMAP axes have no direct physical meaning.
    fig,axs=plt.subplots(2,3,figsize=(10.5,6.4));m=pd.read_csv(OUT/'provenance/main_full_metadata.csv',dtype={'cut_id':str},low_memory=False).fillna('')
    groupcolors={'Mysticeti':'#b88044','Odontoceti':'#2d789b','Pinnipedia':'#789949'}
    for f,ax in zip(ORDER,axs.flat):
        p=OUT/'stage1'/(f+'_structure.npz')
        if not p.exists():ax.set_visible(False);continue
        with np.load(p,allow_pickle=False) as z:coords=z['coords']
        for group,c in groupcolors.items():
            keep=m.group3.eq(group).to_numpy();ax.scatter(coords[keep,0],coords[keep,1],s=1.5,color=c,alpha=.55,rasterized=True,label=group)
        ax.set_title(LABEL[f],fontsize=10);ax.set_xlabel('UMAP 1');ax.set_ylabel('UMAP 2');ax.set_xticks([]);ax.set_yticks([])
    axs[0,0].legend(frameon=False,fontsize=6,markerscale=4)
    f8=finish(fig,'full_cohort_umap_taxon')
    # Machine-readable headline tables preserve numeric values and source paths.
    summary=OUT/'summary';summary.mkdir(exist_ok=True)
    for name,d in [('task_protocol_scores',scores),('40k_four_form_scores',matched[matched.scope.eq('40k')]),
                   ('strict_cross_source_retrieval',retrieval[retrieval.condition.eq('cross_source_matched')]),
                   ('physical_feature_groups',ablation),('source_removal_stability',stab)]:d.to_csv(summary/(name+'.csv'),index=False)
    font=ROOT/'input/fonts/DroidSansFallback.ttf'
    if not font.exists():font=Path('/usr/local/share/fonts/geohub_report/DroidSansFallback.ttf')
    pdfmetrics.registerFont(TTFont('Ko',str(font)))
    pdfmetrics.registerFont(TTFont('Latin',str(ROOT/'input/fonts/DejaVuSans.ttf')))
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyKo',fontName='Latin',fontSize=9.1,leading=14,spaceAfter=7,textColor=colors.HexColor('#243447')))
    styles.add(ParagraphStyle(name='TitleKo',fontName='Latin',fontSize=22,leading=29,spaceAfter=17,textColor=colors.HexColor('#1f4f76')))
    styles.add(ParagraphStyle(name='HeadKo',fontName='Latin',fontSize=16,leading=23,spaceAfter=11,textColor=colors.HexColor('#1f4f76')))
    styles.add(ParagraphStyle(name='SmallKo',fontName='Latin',fontSize=7.8,leading=11,spaceAfter=5,textColor=colors.HexColor('#536375')))
    def text(s,style='BodyKo'):
        s=html.escape(str(s)).replace('\n','<br/>')
        s=re.sub(r'([^\x00-\x7f]+)',r'<font name="Ko">\1</font>',s)
        return Paragraph(s,styles[style])
    story=[]
    def para(s):story.append(text(s))
    def heading(s):story.append(text(s,'HeadKo'))
    def image(p,width=490):
        from PIL import Image as PILImage
        with PILImage.open(p) as im:w,h=im.size
        story.append(Image(str(p),width=width,height=width*h/w));story.append(Spacer(1,8))
    def tab(headers,rows,widths):
        content=[[text(c,'SmallKo') for c in headers]]+[[text(c,'SmallKo') for c in r] for r in rows]
        t=Table(content,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eaf1f7')),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#d0dae3')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]));story.append(t);story.append(Spacer(1,10))
    story.append(text('Watkins 전체 자료\n다섯 분석 검토','TitleKo'))
    para('중간 결과: AST 전체 계산 진행 중' if args.interim else '전체 주 분석 코호트 재분석 완료')
    para('전체 재고 13,693컷 중 기존 포함·제외 기준을 유지한 10,694컷을 주 분석에 사용했습니다. 기존 3,539컷의 특징은 재사용하고 빠진 컷만 같은 모델과 전처리로 확장했습니다. 이 보고서는 데이터 확대, 표집 민감도, 다섯 심화 분석을 연결해 해석합니다.')
    tab(['구분','컷 수','범위'],[['전체 재고','13,693','49종, 63개 메타데이터 출처'],['사용 가능 주 분석','10,694','36종, 49개 출처'],['main 보류','2,315','사육 1,961 · 다종 227 · 공중 60 · 불량 8 · 복합 59'],['별도 LF','115','사용 가능 89 + 보류 26'],['참고래 제외','567','기존 제외 조건 유지'],['기타 분류군','2','세 분류군 밖'],['개발 / 평가','8,592 / 2,102','출처 29 / 20, 기존 출처 분할 유지']], [125,85,280])
    para('주 분석 분류군은 이빨고래 8,820컷(82.5%), 수염고래 1,385컷(13.0%), 기각류 489컷(4.6%)입니다. 전체로 늘려도 출처가 새로 늘어난 것은 아니며, 기존 3,539컷과 같은 49개 출처를 사용합니다.')
    status=[]
    for f in ORDER:
        p=OUT/'features'/(f+'.npz');status.append([LABEL[f], '10,694' if p.exists() else '진행 중', '전체 배열·구조·표집·1~5' if f in features and (OUT/'stage1'/(f+'_structure.npz')).exists() else '전체 결과 미완료'])
    status.append(['ImageNet EfficientNet-B3','3,539','원 전처리 코드 미확보; 기존 표본만'])
    tab(['표현','확보 컷','범위'],status,[155,80,255])
    para('VGGish는 기존 분석의 마지막 활성화 이전 벡터를 유지했습니다. AST zero와 tile은 같은 고정 모델 revision을 사용합니다. 물리 특징 13행은 0.054~0.112초의 짧은 컷으로, 원 알고리즘의 최소 8 STFT 프레임 조건을 충족하지 않아 전 열 결측입니다. 완료된 행 수와 유효 특징의 수는 구분하고, 평가에서는 학습 행의 중앙값으로 대체했습니다.')
    story.append(PageBreak());heading('전체 비지도 구조')
    image(f8)
    para('UMAP은 라벨을 입력하지 않고 계산했습니다. 위 색상은 사후 표시한 분류군이며, UMAP 1~3축에 주파수·시간 같은 물리적 의미가 직접 부여되는 것은 아닙니다. 그림에서 분류군 분리가 보이더라도 출처·샘플링률의 영향을 함께 확인해야 합니다.')
    structure=pd.read_csv(OUT/'stage1/structure_summary.csv').set_index('feature')
    tab(['표현','같은 출처 NN10','같은 fs 계열 NN10','출처 AMI','형태 AMI'],[[LABEL[f],f'{structure.loc[f,"same_source_NN10"]*100:.1f}%',f'{structure.loc[f,"same_fs_NN10"]*100:.1f}%',f'{structure.loc[f,"cluster_source_AMI"]:.3f}',f'{structure.loc[f,"cluster_form_AMI"]:.3f}'] for f in features],[130,100,110,75,75])
    para('NN10은 각 컷의 최근접 10개 이웃에서 조건이 같은 비율이고, AMI는 라벨과 8군집 사이의 우연 보정 일치도입니다. AMI는 분류 정확도가 아닙니다. 메타데이터만으로도 형태 AMI가 나타나므로 군집의 형태 색상을 음향 의미의 독립 증거로 취급하지 않습니다.')
    story.append(PageBreak());heading('비례 표집과 균형 진단')
    image(f7);image(f7b);image(f7c)
    para('위 두 그림의 점은 평균, 선은 seed 최솟값~최댓값입니다. 성능 비교는 개발 출처만 사용해 2,769컷을 10개 seed로 표집합니다. 표집 변동은 신뢰구간이 아닙니다. 별도의 100회 표집 감사는 분포 점검이며 성능 평가 100회를 뜻하지 않습니다.')
    para('구조 비교의 표집은 길이 0.25~60초의 10,471컷에서 3,539컷을 뽑습니다. 균형 진단은 역빈도 가중 KMeans의 목적함수만 변경했습니다. 분류군 가중 조건은 분류군 라벨을 사용한 진단이고, 전처리·이웃 검색을 가중한 결과와는 다릅니다. 모든 분류기의 class_weight=balanced는 고정했습니다.')
    story.append(PageBreak());heading('1. 동일 형태·조건 비교')
    image(f1);image(f2)
    q=matched[matched.scope.eq('40k')].set_index('feature')
    rows=[[LABEL[f],f'{q.loc[f,"BA"]*100:.1f}%',f'{q.loc[f,"BA_lo"]*100:.1f}~{q.loc[f,"BA_hi"]*100:.1f}%'] for f in features]
    tab(['40k 계열 표현','균형정확도','조건부 95% 출처 bootstrap 구간'],rows,[180,100,210])
    para('40k 계열 네 형태는 click, pulse_train, pulsed_call, whistle이며 평가 컷은 621개, 출처는 19개입니다. 7개 형태로 학습한 예측에서 해당 평가 행을 제한한 결과입니다. 별도의 4개 클래스만 학습한 분류기의 성능이나 전체 자료의 통합 정확도로 해석하지 않습니다.')
    seven=scores[scores.task.eq('type_single')&scores.protocol.eq('allcv')].set_index('feature')
    seven_positive=[f for f in features if f!='meta' and seven.loc[f,'delta_lo']>0]
    if seven_positive:
        para('전체 7개 형태의 출처 분리 교차검증에서 메타 대비 차이 구간이 양수인 표현은 '+', '.join(LABEL[f] for f in seven_positive)+'입니다. 조건별 결과와 통합 결과를 함께 확인해야 합니다.')
    else:
        para(f'전체 7개 형태의 출처 분리 교차검증에서는 메타 대조군의 균형정확도가 {seven.loc["meta","BA"]*100:.1f}%이고, 메타 대비 차이의 95% 구간이 양수인 음향 표현은 없었습니다. 40k 조건의 이득을 전체 형태의 개선으로 일반화하지 않습니다.')
    story.append(PageBreak());heading('2. 형태별 출처 구성')
    image(f3)
    q=source[source.scope.eq('dev')&source.form.isin(['song','moan','whistle'])]
    tab(['형태','개발 컷 / 출처','최대 출처 점유율'],[[r.form,f'{r.n} / {r.sources}',f'{r.largest_source_share*100:.2f}% ({r.largest_source})'] for r in q.itertuples()],[120,140,230])
    para('전체 개발 자료에서도 song의 94/95컷, moan의 246/269컷이 각각 하나의 출처에 집중되었습니다. 전체 데이터 사용은 표본 선정의 비율 왜곡을 줄이지만, 형태와 출처가 연결된 자료의 구조를 자동으로 해소하지 않습니다.')
    para('출처 단위는 메타데이터의 연결 관계로 만든 g_site입니다. 실제 장비·개체·녹음 세션이 확인된 식별자와 같다고 단정하지 않습니다. 출처 효과에는 녹음 조건뿐 아니라 실제 생태·행동·지역 차이도 섞일 수 있습니다.')
    story.append(PageBreak())
    heading('3. 출처 간 같은 형태 검색')
    image(f4)
    q=retrieval[retrieval.condition.eq('cross_source_matched')].set_index('feature')
    tab(['표현','형태별 평균 P@5','메타 대비 차이의 구간'],[[LABEL[f],f'{q.loc[f,"P5_macro"]*100:.1f}%',f'{q.loc[f,"delta_meta_lo"]*100:.2f}~{q.loc[f,"delta_meta_hi"]*100:.2f} pp'] for f in features],[150,120,220])
    para('질의 출처를 전처리 적합에서도 제외하고, 다른 출처 중 같은 샘플링률 계열·길이 2배 이내의 후보를 검색했습니다. 질의 2,911컷, 23개 개발 출처의 조건부 결과입니다. 대조군 대비 차이 구간이 0을 포함하는 표현의 우위를 확정하지 않습니다.')
    positive=[f for f in features if f!='meta' and q.loc[f,'delta_meta_lo']>0]
    if positive:
        para('이 탐색 조건에서 메타 대비 차이의 부트스트랩 구간이 0 위에 있는 표현은 '+', '.join(LABEL[f] for f in positive)+'입니다. '+
             ' / '.join(f'{LABEL[f]}: 차이 {q.loc[f,"delta_meta"]*100:.2f} pp, 조건부 95% 구간 {q.loc[f,"delta_meta_lo"]*100:.2f}~{q.loc[f,"delta_meta_hi"]*100:.2f} pp' for f in positive)+'.')
    para('메타 대조군은 샘플링률과 컷 길이의 두 특징입니다. 이 비교는 해당 대조군을 넘어서는 검색 정보를 검토하며, 알려지지 않은 모든 녹음 조건이나 형태·출처의 생태적 연결을 제거한 인과 검증은 아닙니다.')
    story.append(PageBreak());heading('4. 물리 특징군 비교')
    image(f5)
    para('51개 전체, 6개 특징군만 사용, 각 특징군 제외, 스펙트럼 38개, 형태 13개 등 사전에 정한 15조건을 모두 실행했습니다. 개발 출처에서만 분류·샘플링률 회귀를 적합했고, 기존 전체 물리 특징의 예측과 일치도를 확인했습니다.')
    q=ablation.set_index('variant')
    rows=[]
    for variant in ['all','spectral','shape','only_tonality']:
        r=q.loc[variant];rows.append([variant,str(int(r.dimensions)),f'{r.form_BA*100:.1f}%',f'{r.group3_BA*100:.1f}%',f'{r.logfs_R2:.3f}'])
    tab(['조건','차원','형태 BA','분류군 BA','샘플링률 R2'],rows,[145,55,85,90,115])
    r=q.loc['shape'];para(f'형태 13개는 전체 51개보다 형태 균형정확도가 {(r.form_delta_all)*100:.2f} pp 높았지만, 차이 구간 {r.form_delta_all_lo*100:.2f}~{r.form_delta_all_hi*100:.2f} pp는 0을 포함합니다. 메타 대조군 대비 분류 성능 개선 근거는 없습니다. 특징의 개수가 적다는 이유만으로 우수한 표현으로 채택하지 않습니다.')
    ar=pd.read_csv(AN/'04_ablation_retrieval_summary.csv');r=ar[ar.feature.eq('shape')&ar.condition.eq('cross_source_matched')].iloc[0]
    para(f'별도의 전체 개발 전처리를 사용하는 출처 간 맞춤 검색에서는 형태 13개의 P@5가 {r.P5_macro*100:.2f}%이고 메타 대비 차이 구간은 {r.delta_meta_lo*100:.2f}~{r.delta_meta_hi*100:.2f} pp입니다. 이 ablation 검색은 질의 출처까지 전처리에서 제외한 3번 추가 검사와 구분해야 합니다.')
    sr=pd.read_csv(OUT/'five_analyses/physical_strict/tables/03_source_excluded_fit_summary.csv')
    r=sr[sr.feature.eq('shape')&sr.condition.eq('cross_source_matched')].iloc[0]
    b=sr[sr.feature.eq('meta')&sr.condition.eq('cross_source_matched')].iloc[0]
    para(f'추가 탐색 검사에서는 형태 13개도 질의 출처를 전처리에서 제외했습니다. 같은 조건의 {int(r.n_query):,}개 질의에서 P@5 {r.P5_macro*100:.2f}%, 메타 {b.P5_macro*100:.2f}%이며 차이 {r.delta_meta*100:.2f} pp의 조건부 95% 구간은 {r.delta_meta_lo*100:.2f}~{r.delta_meta_hi*100:.2f} pp입니다. 검색 후보로 검토할 근거는 남지만, 분류에서의 우수성이나 새 독립 자료에서의 재현을 확정하지 않습니다.')
    story.append(PageBreak());heading('5. 출처를 제거한 구조 안정성')
    image(f6)
    tab(['표현','출처 제거 수','군집 ARI 중앙값','NN10 겹침 중앙값'],[[LABEL[r.feature],str(int(r.source_removals)),f'{r.cluster_ARI_median:.3f}',f'{r.NN10_Jaccard_source_macro_median:.3f}'] for r in stab.itertuples()],[145,80,125,140])
    para('각 개발 출처를 하나씩 제거한 뒤 전처리와 군집을 다시 적합했습니다. 그림의 선은 29개 제거 조건의 10~90백분위입니다. 안정성이 높다는 사실만으로 콜 형태의 의미를 잘 반영한다고 판단할 수 없습니다. 메타데이터에서도 안정적인 구조가 만들어집니다.')
    heading('현재 근거의 해석 범위')
    para('출처·샘플링 조건과 음향 형태 정보는 함께 존재합니다. 어떤 표현이 유용한지는 분류군, 형태, 샘플링률 계열, 출처 간 검색 중 무엇을 평가하느냐에 따라 달라집니다. 전체 자료의 통합 점수 하나로 모델의 순위를 확정하지 않습니다.')
    para('노트 기반 약한 형태 라벨을 사용한 탐색 결과입니다. 단일 7개 형태 라벨은 3,784컷이며 bark 20컷은 비교 분류에서 제외했습니다. 사용자의 6개 콜 목표와 현재 탐색 라벨 7개는 동일한 분류 체계가 아닙니다. 다음 단계에서 확인할 부분으로 남깁니다.')
    para('기존 평가 출처의 결과는 앞선 작업에서 이미 확인했으므로 이번 분석은 회고적 재분석입니다. 새로운 독립 외부 평가로 설명하지 않습니다. 동일한 선형 분류기 설정을 고정했으며 모델을 새로 지도학습하거나 약한 라벨의 정답성을 확정한 작업은 아닙니다.')
    story.append(PageBreak());heading('재현 조건과 결과 위치')
    tab(['항목','실행 조건'],[['공통 분류기','LogisticRegression(C=1, class_weight=balanced, max_iter=3000)'],['평가 전처리','학습 행의 중앙값 대체·표준화·클리핑, 필요시 PCA 64차원'],['비지도 구조','3D UMAP: neighbors 15, min_dist 0.1, epochs 300; KMeans 8군집'],['부트스트랩','출처 단위 2,000회, 고정 형태가 모두 존재하는 draw만 사용'],['AST 모델','MIT/ast-finetuned-audioset-10-10-0.4593, revision f826b80d28226b62986cc218e5cec390b1096902'],['AST 표현','zero: 앞 10.24초, 짧으면 mel 뒤 0 채움. tile: 짧은 파형 반복, 긴 컷 창별 평균'],['시간 범위','원음은 최대 120초 분석; 25개 긴 컷은 일부 시간만 사용'],['VGGish','16 kHz, 파일별 patch 평균, 원 캐시의 마지막 활성화 이전 표현'],['B3 범위','torchvision ImageNet EfficientNet-B3. Perch v2 가중치가 아님. 기존 3,539컷만']], [120,370])
    para('전체 결과는 full_cohort/output/, 이전 표본 결과는 deep_review/output/tables/에 있습니다. 결과표·예측·좌표·검사·코드는 ZIP에 포함합니다. 공개 원음은 별도이며, 확보한 public_manifest.json과 다운로드 스크립트로 같은 해시·샘플링률·프레임 수를 재검증할 수 있습니다.')
    para('실제 완료 판정은 final_validation.json의 배열 ID, 예측 행, 출처 중복, 표집 결과, 특징군 비교, 출처 제거 체크포인트를 기준으로 합니다. 최종 패키지는 이 검사가 성공했을 때만 만들도록 제한했습니다.')
    para('이 보고서의 모든 수치는 저장된 CSV에서 직접 읽었으며, 각 그림의 원 자료도 함께 저장했습니다. 학습 기반 임베딩의 샘플링률 변환은 물리 특징의 원 주파수 범위와 같지 않으므로 두 표현의 차이를 모델 아키텍처 하나로 설명하지 않습니다.')
    target=OUT/'Watkins_Full_Cohort_Review_KO.pdf'
    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#d5dfe8'));canvas.line(42,42,553,42)
        canvas.setFont('Latin',8);canvas.setFillColor(colors.HexColor('#647489'))
        canvas.drawString(42,29,'Watkins - full eligible cohort - '+('interim' if args.interim else 'completed analyses'))
        canvas.drawRightString(553,29,str(doc.page))
    doc=SimpleDocTemplate(str(target),pagesize=A4,leftMargin=43,rightMargin=43,topMargin=42,bottomMargin=53)
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    from pypdf import PdfReader
    pages=len(PdfReader(str(target)).pages)
    print(json.dumps(dict(pdf=str(target),pages=pages,features=features,interim=args.interim),ensure_ascii=False))

if __name__=='__main__':main()
