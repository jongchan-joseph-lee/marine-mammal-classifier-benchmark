import pandas as pd, requests, base64, hashlib, concurrent.futures as cf, os
d=pd.read_csv('labels_3539.csv',dtype={'cut_id':str})
def one(r):
    p=f'audio/{r.cut_id}.wav'
    if os.path.exists(p): return 'skip'
    for a in range(3):
        try:
            x=requests.get(r.url,timeout=60).content
            if base64.b64encode(hashlib.md5(x).digest()).decode()!=r.md5: raise ValueError('md5')
            open(p,'wb').write(x); return 'ok'
        except Exception as e: err=str(e)
    return 'fail '+err
with cf.ThreadPoolExecutor(16) as ex: res=list(ex.map(one,d.itertuples()))
print(pd.Series([r.split()[0] for r in res]).value_counts())
