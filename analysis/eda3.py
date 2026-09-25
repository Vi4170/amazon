import os; os.environ['POLARS_MAX_THREADS']='8'
import polars as pl, sys, re
sys.path.insert(0,'code/business_entity_resolution/src'); import io_utils as u
sys.stdout.reconfigure(encoding='utf-8')
s1=u.load('train','source1'); gt=u.load('train','ground_truth')
tr=pl.concat([u.load('train','source2'),u.load('train','source3')])
dev=r'[ऀ-ॿ]'
for nm,d in [('s1',s1),('tgt',tr),('test s1',u.load('test','source1'))]:
    print(nm,'devanagari name',d['business_name'].str.contains(dev).mean(),'addr',d['business_address'].str.contains(dev).mean())
pairs=gt.with_columns(pl.col('matched_entity_ids').str.split(',')).explode('matched_entity_ids').filter(pl.col('matched_entity_ids')!='')
pairs=pairs.join(s1,left_on='source1_entity_id',right_on='entity_id').join(tr,left_on='matched_entity_ids',right_on='entity_id',suffix='_t')
low=lambda c: pl.col(c).str.to_lowercase().str.replace_all(r'[^\p{L}\p{N}]+',' ').str.strip_chars()
p=pairs.with_columns(low('business_name').alias('n1'),low('business_name_t').alias('n2'),low('business_address').alias('a1'),low('business_address_t').alias('a2'))
print('exact norm name',(p['n1']==p['n2']).mean(),' exact norm addr',(p['a1']==p['a2']).mean(), ' empty tgt addr',(p['a2']=='').mean())
# first number token agreement
num=lambda c: pl.col(c).str.extract_all(r'\d+')
p=p.with_columns(num('a1').alias('d1'),num('a2').alias('d2'))
print('tgt addr has digits',(p['d2'].list.len()>0).mean(),' share >=1 number', p.select(pl.col('d1').list.set_intersection(pl.col('d2')).list.len()>0).to_series().mean())
# token diff: tokens in target name not in s1 name
t=p.select(pl.col('n1').str.split(' ').alias('t1'),pl.col('n2').str.split(' ').alias('t2'),'country').with_columns(pl.col('t2').list.set_difference('t1').alias('extra'),pl.col('t1').list.set_difference('t2').alias('miss'))
for c in ['India','US']:
    tc=t.filter(pl.col('country')==c)
    print(c,'EXTRA',tc.select(pl.col('extra').explode()).to_series().value_counts(sort=True).head(40).rows())
    print(c,'MISSING',tc.select(pl.col('miss').explode()).to_series().value_counts(sort=True).head(25).rows())
    print(c,'name token share any',tc.select(pl.col('t1').list.set_intersection(pl.col('t2')).list.len()>0).to_series().mean())
