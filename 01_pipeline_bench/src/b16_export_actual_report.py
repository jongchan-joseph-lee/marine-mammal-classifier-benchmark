"""Export verified results, development-only UMAP, and selected original-audio examples."""
from __future__ import annotations
import base64, contextlib, html, io, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from wb_common import DATA_DIR, PKG_ROOT, SEED, load_config, read_audio, work_dir, log

DISPLAY={'meta':'SR·길이 대조군','mfcc16k':'MFCC','physv0':'물리 특징',
         'vggish':'VGGish','ast_zero':'AST','vggish_head10p24':'VGGish 앞 10.24초'}
MODELS=['meta','mfcc16k','physv0','vggish','ast_zero','vggish_head10p24']

def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.generic):value=value.item()
    if isinstance(value,float) and not np.isfinite(value):return None
    return value

def cell(row):
    return f'{row.value*100:.1f}% [{row.ci_lo*100:.1f}, {row.ci_hi*100:.1f}]'

def read_results():
    a=pd.read_csv(work_dir('eval','actual_ast_vggish_primary')/'results.csv')
    b=pd.read_csv(work_dir('eval','actual_ast_vggish_secondary')/'results.csv')
    return pd.concat([a,b],ignore_index=True)

def score_table(results):
    records=[]
    for name in MODELS:
        row={'특징':DISPLAY[name]}
        for task,title,metric in [('group3','분류군','balanced_acc'),('type_single','단일 콜','balanced_acc'),
                                   ('type_multi','다중 콜','macro_auroc'),('species','종 진단','balanced_acc')]:
            for protocol,plabel in [('devcv','개발'),('test','시험')]:
                part=results[(results.feature==name)&(results.task==task)&(results.protocol==protocol)&
                             (results.subset=='all')&(results.metric==metric)]
                row[title+' '+plabel]=cell(part.iloc[0]) if len(part) else '평가 불가'
        records.append(row)
    return pd.DataFrame(records)

def make_chart(results,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for ax,(task,title,metric) in zip(axes.flat,[('group3','Taxonomic group / balanced accuracy','balanced_acc'),
              ('type_single','Single call / WEAK labels / balanced accuracy','balanced_acc'),
              ('type_multi','Multi call / WEAK labels / macro AUROC','macro_auroc'),
              ('species','Species diagnostic / balanced accuracy','balanced_acc')]):
        for protocol,offset,color in [('devcv',-.16,'#3478c7'),('test',.16,'#ee8b47')]:
            for i,name in enumerate(MODELS):
                q=results[(results.feature==name)&(results.task==task)&(results.protocol==protocol)&
                          (results.subset=='all')&(results.metric==metric)]
                if q.empty:continue
                r=q.iloc[0];y=i+offset
                ax.plot([r.ci_lo,r.ci_hi],[y,y],color=color,alpha=.75,lw=2)
                ax.scatter([r.value],[y],color=color,s=32,label=protocol if i==0 else None)
        ax.set_yticks(range(len(MODELS)),['SR + duration','MFCC','Physical','VGGish','AST','VGGish head10.24'])
        ax.set_xlim(0,1);ax.grid(axis='x',alpha=.2);ax.set_title(title,fontsize=10);ax.invert_yaxis()
        ax.legend(fontsize=8)
    fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)

def interpretation(results,paired):
    def score(feature,task,protocol,metric='balanced_acc'):
        row=results[(results.feature==feature)&(results.task==task)&
                    (results.protocol==protocol)&(results.subset=='all')&
                    (results.metric==metric)].iloc[0]
        return row.value
    delta=paired[(paired.A=='vggish')&(paired.B=='ast_zero')&
                 (paired.task=='type_single')&(paired.protocol=='test')].iloc[0]
    return f'''<div class="box"><h2>이번 결과가 말해 주는 것</h2>
<p>개발 해역의 단일 콜 균형정확도는 AST {score('ast_zero','type_single','devcv')*100:.1f}%, VGGish {score('vggish','type_single','devcv')*100:.1f}%입니다. 시험에서도 점추정은 AST {score('ast_zero','type_single','test')*100:.1f}%, VGGish {score('vggish','type_single','test')*100:.1f}%이지만, AST − VGGish 차이의 95% 구간은 [{-delta.ci_hi*100:.1f}, {-delta.ci_lo*100:.1f}]%p로 0을 포함합니다. 단일 콜의 시험 순위는 확정할 수 없습니다.</p>
<p>단일 콜 시험에서는 SR·길이 대조군 {score('meta','type_single','test')*100:.1f}%, 물리 특징 {score('physv0','type_single','test')*100:.1f}%가 AST보다 높은 점추정을 보였습니다. 두 기준선과 AST의 직접 차이 구간도 0을 포함합니다. 현재 결과는 사전학습 모델이 콜 형태를 녹음 조건과 독립적으로 더 잘 구분한다는 근거가 충분하지 않음을 보여 줍니다.</p>
<p>다중 콜 시험 macro AUROC는 AST {score('ast_zero','type_multi','test','macro_auroc'):.3f}, VGGish {score('vggish','type_multi','test','macro_auroc'):.3f}, 물리 특징 {score('physv0','type_multi','test','macro_auroc'):.3f}입니다. 이 역시 자동 weak label과의 일치도입니다. 개발의 다중 콜은 6범주, 시험은 8범주이므로 두 열의 차이를 성능 향상으로 해석하지 않습니다. 종 진단도 개발 10종·시험 9종으로 평가 범주가 다릅니다.</p>
<p>참고 평가에서 AST 단일 콜은 같은 컷의 파일 분할 {score('ast_zero','type_single','filecv')*100:.1f}%에 비해 해역 분할 {score('ast_zero','type_single','allcv')*100:.1f}%였습니다. 파일을 섞으면 점수가 크게 높아집니다. 개발 자료의 군집도 콜 형태뿐 아니라 SR 계열과 연관돼 있으므로 원음과 녹음 조건을 함께 확인할 필요가 있습니다.</p>
<p>수동 120개 표기는 이번 임베딩·비교 실험의 필수 조건이 아닙니다. 실제 콜 정확도를 보고하는 단계에서 독립 정답이 필요합니다. 다음 특징 개선은 개발 자료에서 진행하고, 아래 불일치 20개는 라벨 오류와 모델 오류를 구분하는 점검 후보로 사용합니다.</p></div>'''

def development_views(out):
    from b07_evaluate import fit_transform_fold
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_mutual_info_score,adjusted_rand_score
    import umap
    manifest=pd.read_csv(DATA_DIR/'bench_core.csv',dtype=str,keep_default_na=False)
    dev=manifest[(manifest.pool=='main')&(manifest.split=='dev')].reset_index(drop=True)
    views={};cluster_rows=[]
    weak=dev.task_type_single.to_numpy()
    counts=dev.task_type_single.value_counts()
    supported={c for c,n in counts.items() if c and n>=23 and dev[dev.task_type_single==c].g_site.nunique()>=2}
    use=np.isin(weak,list(supported))
    for name in ['ast_zero','vggish','mfcc16k','physv0']:
        with np.load(work_dir('features')/(name+'.npz'),allow_pickle=False) as z:
            positions={str(cid):i for i,cid in enumerate(z['cut_id'])}
            X=z['X'][[positions[c] for c in dev.cut_id]].astype(float)
        Z,_=fit_transform_fold(X,X[:1])
        km=KMeans(n_clusters=len(supported),n_init=10,random_state=SEED).fit_predict(Z[use])
        cluster_rows.append(dict(feature=name,n=int(use.sum()),k=len(supported),
              ari_weak_call=float(adjusted_rand_score(weak[use],km)),
              ami_weak_call=float(adjusted_mutual_info_score(weak[use],km)),
              ami_site=float(adjusted_mutual_info_score(dev.g_site.to_numpy()[use],km)),
              ami_fs_family=float(adjusted_mutual_info_score(dev.fs_family.to_numpy()[use],km))))
        embedding=umap.UMAP(n_components=3,n_neighbors=20,min_dist=.15,metric='cosine',
                           random_state=SEED,n_epochs=200,n_jobs=1).fit_transform(Z)
        views[name]=np.round(embedding,4).tolist()
        pd.DataFrame(embedding,columns=['UMAP1','UMAP2','UMAP3']).assign(cut_id=dev.cut_id).to_csv(
            out/(name+'_dev_umap3d.csv'),index=False)
        log('Development UMAP and clustering: '+name)
    table=pd.DataFrame(cluster_rows);table.to_csv(out/'development_clustering.csv',index=False)
    return dev,views,table

def priority_examples(dev,predictions,out):
    p=predictions[(predictions.task=='type_single')&(predictions.protocol=='devcv')&
                  predictions.feature.isin(['ast_zero','vggish','physv0','mfcc16k'])].copy()
    wide=p.pivot(index='cut_id',columns='feature',values='pred')
    confidence=p.pivot(index='cut_id',columns='feature',values='p_max')
    frame=dev.set_index('cut_id').join(wide.add_prefix('pred_')).join(confidence.add_prefix('score_'))
    frame=frame[(frame.task_type_single!='')&frame.pred_ast_zero.notna()&frame.pred_vggish.notna()]
    frame['models_disagree']=frame.pred_ast_zero!=frame.pred_vggish
    frame['ast_diff_weak']=frame.pred_ast_zero!=frame.task_type_single
    frame['vggish_diff_weak']=frame.pred_vggish!=frame.task_type_single
    frame['priority_score']=(3*frame.models_disagree.astype(int)+frame.ast_diff_weak.astype(int)+
                frame.vggish_diff_weak.astype(int)+1-frame[['score_ast_zero','score_vggish']].mean(axis=1))
    ranked=frame.sort_values('priority_score',ascending=False)
    chosen=[];sites={};types={}
    for cid,r in ranked.iterrows():
        if not (r.models_disagree or r.ast_diff_weak or r.vggish_diff_weak):continue
        if sites.get(r.g_site,0)>=2 or types.get(r.task_type_single,0)>=4:continue
        chosen.append(cid);sites[r.g_site]=sites.get(r.g_site,0)+1
        types[r.task_type_single]=types.get(r.task_type_single,0)+1
        if len(chosen)>=20:break
    queue=ranked.loc[chosen].reset_index()
    queue.to_csv(out/'review_priority_20.csv',index=False,encoding='utf-8-sig')
    previews={}
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram
    for cid in chosen:
        path=Path(load_config()['audio_root'])/(cid+'.flac')
        x,fs=read_audio(str(path),10.24)
        nfft=min(max(256,2**int(np.ceil(np.log2(fs*.025)))),2**int(np.floor(np.log2(len(x)))))
        f,t,S=spectrogram(x,fs=fs,nperseg=nfft,noverlap=nfft//2)
        power=10*np.log10(S+1e-24);power-=np.max(power)
        fig,axes=plt.subplots(2,1,figsize=(7,3.8),gridspec_kw={'height_ratios':[1,3]},layout='constrained')
        step=max(1,len(x)//4000)
        axes[0].plot(np.arange(0,len(x),step)/fs,x[::step],lw=.5,color='#245876')
        axes[0].set_ylabel('Amplitude');axes[0].set_title(cid+' / native '+str(fs)+' Hz',fontsize=10)
        axes[1].pcolormesh(t,f/1000,power,shading='auto',vmin=-75,vmax=0,cmap='magma')
        axes[1].set_ylabel('Frequency (kHz)');axes[1].set_xlabel('Time (s)')
        buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=110);plt.close(fig)
        previews[cid]={'image':base64.b64encode(buffer.getvalue()).decode(),
                       'audio':base64.b64encode(path.read_bytes()).decode()}
    return queue,previews,wide

def notebook(out):
    cells=[{'cell_type':'markdown','id':'watkins-results-intro','metadata':{},'source':['# Watkins AST · VGGish 실제 계산 결과\n',
        '3,539개 main 표본의 실제 사전학습 임베딩과 동일 조건 평가 결과입니다. 콜 형태는 노트 기반 weak label이며 수동 골드가 아닙니다.\n',
        '아래 셀은 저장한 결과를 읽습니다. 모델 설치나 재추론 없이 결과를 재검토할 수 있습니다.\n']}]
    blocks=[
    "from pathlib import Path\nimport numpy as np, pandas as pd\nROOT = Path.cwd()\nif not (ROOT/'data/bench_core.csv').exists(): ROOT = ROOT.parent\nFEATURES = ROOT/'work/features'\nfor name in ['ast_zero','vggish','vggish_head10p24']:\n    with np.load(FEATURES/(name+'.npz'), allow_pickle=False) as z:\n        print(name, z['X'].shape, 'finite:', bool(np.isfinite(z['X']).all()))",
    "R = pd.read_csv(ROOT/'work/eval/actual_ast_vggish_primary/results.csv')\nS = R[(R.subset=='all') & R.metric.isin(['balanced_acc','macro_auroc'])]\nprint(S[['task','protocol','feature','metric','value','ci_lo','ci_hi','n_eval','n_classes']].to_string(index=False))",
    "C = pd.read_csv(ROOT/'work/actual_report/development_clustering.csv')\nprint(C.to_string(index=False))\nprint('\\n콜 형태: weak label 일치도. 실제 정확도와 구별해야 합니다.')"
    ]
    namespace={}
    previous=Path.cwd()
    os.chdir(PKG_ROOT)
    try:
        for count,source in enumerate(blocks,1):
            buffer=io.StringIO()
            with contextlib.redirect_stdout(buffer):exec(source,namespace)
            cells.append({'cell_type':'code','id':f'watkins-results-{count}','metadata':{},'execution_count':count,'source':source.splitlines(True),
                          'outputs':[{'output_type':'stream','name':'stdout','text':buffer.getvalue().splitlines(True)}]})
    finally:os.chdir(previous)
    payload={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
             'language_info':{'name':'python','version':sys.version.split()[0]}},'nbformat':4,'nbformat_minor':5}
    (PKG_ROOT/'notebooks/Watkins_AST_VGGish_Actual_Results.ipynb').write_text(json.dumps(payload,ensure_ascii=False,indent=1))

def main():
    out=work_dir('actual_report');R=read_results()
    table=score_table(R);table.to_csv(out/'comparison_summary.csv',index=False,encoding='utf-8-sig')
    make_chart(R,out)
    dev,views,clusters=development_views(out)
    P=pd.read_csv(work_dir('eval','actual_ast_vggish_primary')/'predictions.csv',dtype={'cut_id':str},keep_default_na=False)
    queue,previews,wide=priority_examples(dev,P,out)
    columns=['cut_id','species','species_ko','group3','fs_hz','fs_family','dur_s','task_type_single',
             'type_weak','type_reading','g_site','location','year','note']
    points=dev[columns].set_index('cut_id').join(wide.add_prefix('pred_')).reset_index()
    data=dict(points=clean(points.to_dict('records')),views=views,cases=clean(queue.to_dict('records')),previews=previews)
    data_json=json.dumps(clean(data),ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    chart=base64.b64encode((out/'comparison.png').read_bytes()).decode()
    scope=json.loads((work_dir('actual_reference_run')/'label_and_input_scope.json').read_text())
    primary_info=json.loads((work_dir('eval','actual_ast_vggish_primary')/'run_info.json').read_text())
    paired=pd.read_csv(work_dir('eval','actual_ast_vggish_primary')/'paired.csv')
    comparison=paired[(paired.A=='vggish')&(paired.B=='ast_zero')][['task','protocol','diff_balanced_acc','ci_lo','ci_hi','verdict']]
    count_table=R[(R.feature=='meta')&(R.subset=='all')&R.protocol.isin(['devcv','test'])&
                  R.metric.isin(['balanced_acc','macro_auroc'])][['task','protocol','n_eval','n_sites','n_classes','classes','dropped']]
    region_table=R[(R.subset=='region')&R.protocol.isin(['devcv','test'])&(R.metric=='balanced_acc')&
                   R.feature.isin(['meta','physv0','vggish','ast_zero'])][
                   ['task','protocol','feature','value','ci_lo','ci_hi','n_eval','n_sites','n_classes']]
    sr_table=R[(R.subset=='fs10k-20k')&(R.task=='group3')&R.protocol.isin(['devcv','test'])&
               (R.metric=='balanced_acc')&R.feature.isin(['meta','physv0','vggish','ast_zero'])][
               ['protocol','feature','value','ci_lo','ci_hi','n_eval','n_sites','n_classes']]
    html_head='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Watkins AST · VGGish 실제 비교 결과</title><style>
*{box-sizing:border-box}body{margin:0;background:#f2f5f7;color:#203344;font-family:system-ui,"Malgun Gothic",sans-serif;line-height:1.6}
main{max-width:1450px;margin:auto;padding:28px}h1{font-size:29px}h2{margin-top:30px;font-size:21px}.box{background:white;border:1px solid #dce3e8;border-radius:12px;padding:20px;margin:18px 0}
.note{border-left:4px solid #d5932b;background:#fff7e5;padding:14px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}
th,td{padding:9px;border-bottom:1px solid #e0e7ed;text-align:left}th{background:#eef4f8}img{max-width:100%}select,button{padding:9px;margin:4px;border:1px solid #bac9d5;border-radius:7px;background:white}
.two{display:grid;grid-template-columns:2fr 1fr;gap:18px}.small{font-size:13px;color:#5a6c7b}#plot{height:670px}#details{overflow-wrap:anywhere}audio{width:100%;margin:9px 0}pre{white-space:pre-wrap}.pill{display:inline-block;padding:5px 12px;background:#e6f1f9;border-radius:20px;margin:3px}
@media(max-width:850px){.two{grid-template-columns:1fr}main{padding:15px}#plot{height:500px}}</style><main>
<h1>Watkins AST · VGGish: 실제 계산과 동일 조건 평가</h1>
<p>기존 Watkins_Bench_v1.1의 main 핵심 표본 전체를 사용했습니다. 실행 준비를 넘어 실제 사전학습 가중치 추론, 컷별 임베딩 저장, 동일 분할 평가까지 완료한 결과입니다.</p>
<div class="pill">3,539개 원음</div><div class="pill">49개 해역</div><div class="pill">개발 2,769 · 시험 770</div><div class="pill">36종 · 3개 분류군</div>
<div class="note"><b>콜 형태 점수는 weak label 일치도입니다.</b> Watkins 노트 기반 자동 라벨을 사용했습니다. 사용자는 약 30개를 육안 확인했지만 이 패키지에는 개별 수동 표기 CSV가 없어 골드 정확도에 포함하지 않았습니다. 수동 120개 표기를 이번 계산의 선행 조건으로 두지 않았습니다.</div>
<div class="box"><h2>같은 조건에서 무엇을 비교했나</h2>
<p>동일한 컷·기존 해역 분할·겹 안에서만 맞춘 전처리·PCA 최대 64개·균형 로지스틱 회귀(C=1)·지표를 적용했습니다. 개발은 해역 5겹 교차검증, 시험은 개발 해역 전체에서 학습한 탐침을 별도 20개 해역에 적용했습니다. 동결 표현을 평가했으며 네트워크를 새로 학습하거나 미세조정하지 않았습니다.</p>
<p>AST는 768차원, VGGish는 128차원입니다. 두 모델은 DC 제거·최댓값 정규화·16 kHz 변환을 같은 방식으로 수행했습니다. 물리 특징은 원래 fs를 유지하고, MFCC는 기존 16 kHz 설정을 유지했습니다. 원본 3,539개는 공개본 MD5와 원래 fs·프레임 수를 검증했습니다.</p>
<p>AST는 앞 10.24초와 멜 0 채움, VGGish는 전체 컷의 0.96초 단위 평균을 사용합니다. 길이 차이를 통제하기 위해 VGGish 앞 10.24초 조건도 계산했습니다. 212개 컷이 10.24초를 넘습니다. 짧은 VGGish 입력은 기존 계획대로 반복 채움합니다.</p>
<p class="small">AST는 CPU BF16 추론 후 FP32로 저장했습니다. 실제 8개 음원의 FP32 대조에서 평균 코사인 일치도 0.99991, 최소 0.999857을 확인했습니다. VGGish는 FP32입니다. 두 모델의 16 kHz 입력은 원래 8 kHz 위의 정보를 보존하지 못하며, 원래 fs가 낮은 컷을 올려 변환해도 관측되지 않은 정보가 생기지 않습니다.</p></div>
<div class="box"><h2>실제 비교 점수</h2><p class="small">균형정확도 / 다중 콜은 macro AUROC. 대괄호는 평가 해역을 2,000회 재추출한 95% 구간입니다. 학습 자료 변동을 포함한 구간은 아닙니다. 단일 콜의 bark는 학습 자료가 부족해 평가에서 빠집니다. song은 시험 양성 컷이 한 해역뿐이므로 해역 일반화의 근거가 약합니다.</p>'''
    body=html_head+'<div class="scroll">'+table.to_html(index=False,escape=True)+'</div><img alt="비교 점수와 구간" src="data:image/png;base64,'+chart+'">'
    body+='<h3>실제 평가 컷·해역·범주</h3><div class="scroll">'+count_table.to_html(index=False,escape=True)+'</div>'
    body+='<h3>VGGish − AST의 직접 비교</h3><p class="small">같은 평가 컷과 같은 해역 재추출을 사용했습니다. 구간에 0이 들어 있으면 순위를 확정하지 않습니다.</p>'+comparison.to_html(index=False,escape=True)+'</div>'
    body+=interpretation(R,paired)
    body+='<div class="box"><h2>북극권·한반도 대상 종 부분집합</h2><p>전체 개발 자료에서 학습하고 기존 region 표시가 있는 컷에서만 평가합니다. 부분집합의 범주 수와 표본 수를 함께 확인하세요.</p><div class="scroll">'+region_table.to_html(index=False,escape=True,float_format=lambda x:f'{x:.3f}')+'</div>'
    body+='<h3>SR 10k·20k 계열에서의 분류군</h3><p class="small">세 분류군이 함께 있는 샘플링 계열입니다. 학습은 전체 개발 자료, 평가는 해당 계열로 제한합니다. 공통 4 kHz 재추론과는 별도 조건입니다.</p><div class="scroll">'+sr_table.to_html(index=False,escape=True,float_format=lambda x:f'{x:.3f}')+'</div></div>'
    body+='<div class="box"><h2>개발 자료의 비지도 군집</h2><p>동일한 k-평균 군집을 weak 콜 형태·해역·fs 계열과 비교했습니다. ARI와 AMI는 우연 수준이 0입니다. 아래 진단과 UMAP은 개발 자료만 사용했습니다.</p>'+clusters.to_html(index=False,escape=True,float_format=lambda x:f'{x:.3f}')+'</div>'
    body+='''<div class="box"><h2>3D UMAP과 원음 점검</h2><p class="small">라벨을 입력하지 않은 UMAP입니다. 모델마다 좌표는 독립적이며 축은 물리량이 아닙니다. 점을 눌러 메타데이터와 교차검증 예측을 확인할 수 있습니다. 개발 자료 2,769개만 표시하며 시험 결과를 바탕으로 후보를 골라내지 않았습니다.</p>
<select id="model"><option value="ast_zero">AST</option><option value="vggish">VGGish</option><option value="mfcc16k">MFCC</option><option value="physv0">물리 특징</option></select>
<select id="color"><option value="task_type_single">Weak 콜 형태</option><option value="group3">분류군</option><option value="fs_family">SR 계열</option><option value="g_site">해역</option></select>
<div class="two"><div id="plot"></div><div id="details"><p>점이나 아래 검토 후보를 선택하세요.</p></div></div></div>
<div class="box"><h2>개발 자료의 우선 검토 후보 20개</h2><p>AST·VGGish가 서로 다르거나 weak label과 다른 사례를 해역·형태에 쏠리지 않게 골랐습니다. 실패 확정이나 자동 정답 수정이 아니라 원인 확인 후보입니다. 원래 FLAC와 원래 fs의 파형·스펙트로그램을 담았습니다. 그림은 앞 10.24초까지입니다.</p><select id="case"><option value="">후보 선택</option></select></div>
<div class="box"><h2>재현과 해석</h2><p>동일 조건의 결과를 재검토하려면 제공한 NPZ 임베딩과 결과 CSV를 사용하면 됩니다. 모델을 다시 설치하거나 다운로드할 필요가 없습니다. 새로 추론할 때 사용할 수정 코드와 실행 기록도 전체 묶음에 포함했습니다.</p>
<p>Weak label은 비교 실험을 진행하기 위한 대리 라벨입니다. 실제 콜 정확도를 보고하려면 독립적으로 확인한 표본이 필요합니다. 120개 전량 수작업을 요구하기보다 이 검토 후보의 불일치 원인과 라벨이 애매한 사례부터 확인하면 다음 특징 개선의 방향을 정할 수 있습니다. 라벨 오류·입력 대역 손실·녹음 조건·특징 한계를 구분해 기록하세요.</p>
<p class="small">공식 정의: <a href="https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593">MIT AST 가중치</a> · <a href="https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer">AST 문서</a> · <a href="https://github.com/tensorflow/models/tree/master/research/audioset/vggish">Google VGGish</a> · <a href="https://docs.pytorch.org/audio/2.8.0/prototype.pipelines.html">TorchAudio 2.8 모델</a></p></div></main>'''
    from plotly.offline import get_plotlyjs
    body+='<script>'+get_plotlyjs()+'</script><script id="payload" type="application/json">'+data_json+'</script>'
    body+=r'''<script>
const DATA=JSON.parse(document.getElementById('payload').textContent);
const COLORS=['#2878b7','#e38035','#5b9b58','#a568bd','#d35465','#5cbbb5','#8c735c','#a6a6a6','#33456b','#c2a928'];
function describe(p){const box=document.getElementById('details');box.replaceChildren();
const h=document.createElement('h3');h.textContent=p.cut_id+' · '+p.species_ko;box.appendChild(h);
const text=document.createElement('p');text.textContent=[p.species,p.fs_hz+' Hz',p.dur_s+' s',p.group3,p.location,p.year].join(' / ');box.appendChild(text);
const labels=document.createElement('p');labels.textContent='Weak label: '+(p.task_type_single||p.type_weak||'없음')+' | AST: '+(p.pred_ast_zero||'단일 콜 평가 대상 아님')+' | VGGish: '+(p.pred_vggish||'단일 콜 평가 대상 아님');box.appendChild(labels);
const note=document.createElement('pre');note.textContent=p.note;box.appendChild(note);
const preview=DATA.previews[p.cut_id];if(preview){const audio=document.createElement('audio');audio.controls=true;audio.src='data:audio/flac;base64,'+preview.audio;box.appendChild(audio);const img=document.createElement('img');img.src='data:image/png;base64,'+preview.image;img.alt='Native waveform and spectrogram';box.appendChild(img);}}
function draw(){const model=document.getElementById('model').value,key=document.getElementById('color').value;
const groups={};DATA.points.forEach((p,i)=>{const group=p[key]||'라벨 없음';(groups[group]??=[]).push(i);});
const traces=Object.entries(groups).map(([group,ids],j)=>({type:'scatter3d',mode:'markers',name:group,
x:ids.map(i=>DATA.views[model][i][0]),y:ids.map(i=>DATA.views[model][i][1]),z:ids.map(i=>DATA.views[model][i][2]),
customdata:ids,text:ids.map(i=>DATA.points[i].cut_id+'<br>'+DATA.points[i].species+'<br>'+group),
hovertemplate:'%{text}<extra></extra>',marker:{size:3,opacity:.6,color:group==='라벨 없음'?'#bbb':COLORS[j%COLORS.length]}}));
Plotly.react('plot',traces,{margin:{l:0,r:0,b:0,t:15},scene:{xaxis:{title:{text:'UMAP 1'}},yaxis:{title:{text:'UMAP 2'}},zaxis:{title:{text:'UMAP 3'}}},
legend:{orientation:'h',font:{size:10}},paper_bgcolor:'white'},{responsive:true,displaylogo:false}).then(()=>{const plot=document.getElementById('plot');plot.removeAllListeners('plotly_click');plot.on('plotly_click',ev=>describe(DATA.points[ev.points[0].customdata]));});}
document.getElementById('model').onchange=draw;document.getElementById('color').onchange=draw;
DATA.cases.forEach(p=>{const o=document.createElement('option');o.value=p.cut_id;o.textContent=p.cut_id+' · '+p.task_type_single+' · AST '+p.pred_ast_zero+' / VGGish '+p.pred_vggish;document.getElementById('case').appendChild(o);});
document.getElementById('case').onchange=ev=>{const p=DATA.points.find(p=>p.cut_id===ev.target.value);if(p)describe(p);};draw();
</script>'''
    (out/'Watkins_AST_VGGish_Actual_Report.html').write_text(body,encoding='utf-8')
    summary={'n_core':primary_info['n_core_used'],'coverage':primary_info['coverage'],'human_gold_rows':0,
             'scores':clean(R[(R.subset=='all')&R.metric.isin(['balanced_acc','macro_auroc'])].to_dict('records')),
             'primary_pairwise_vggish_minus_ast':clean(comparison.to_dict('records')),'review_examples':len(queue)}
    (out/'actual_result_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    notebook(out)
    log('Report, executed result notebook, figures and review queue exported')
    return 0

if __name__=='__main__':raise SystemExit(main())
