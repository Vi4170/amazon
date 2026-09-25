import os; os.environ['POLARS_MAX_THREADS']='8'
import polars as pl, sys, collections, unicodedata
sys.path.insert(0,'code/business_entity_resolution/src'); import io_utils as u
sys.stdout.reconfigure(encoding='utf-8')
for sp in ['train','test']:
    d=pl.concat([u.load(sp,'source2'),u.load(sp,'source3')]).sample(400000,seed=0)
    txt=''.join(d['business_name'].to_list())+''.join(d['business_address'].to_list())
    blocks=collections.Counter()
    for ch in txt:
        o=ord(ch)
        if o>=0x250: blocks[unicodedata.name(ch,'?').split(' ')[0]]+=1
    print(sp,blocks.most_common(20))
    fr=d.filter(pl.col('country')=='France')
    if fr.height: print(fr.head(15).select('business_name','business_address').rows())
