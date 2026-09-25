import polars as pl, io_utils as u
s1,s2,s3,gt=[u.load('train',n) for n in ['source1','source2','source3','ground_truth']]
t1,t2,t3=[u.load('test',n) for n in ['source1','source2','source3']]
for nm,d in [('tr s1',s1),('tr s2',s2),('tr s3',s3),('te s1',t1),('te s2',t2),('te s3',t3)]:
    print(nm, d['country'].value_counts().sort('count',descending=True).rows(),
          'empty name',(d['business_name']=='').sum(),'empty addr',(d['business_address']=='').sum(),
          'dup id', d.height-d['entity_id'].n_unique())
g=gt.with_columns(pl.col('matched_entity_ids').str.split(',').list.eval(pl.element().filter(pl.element()!='')).alias('m'))
g=g.with_columns(pl.col('m').list.len().alias('k'),
  pl.col('m').list.eval(pl.element().str.starts_with('S2-').cast(pl.Int32)).list.sum().alias('k2'))
g=g.with_columns((pl.col('k')-pl.col('k2')).alias('k3'))
print('matches per S1:',g['k'].value_counts().sort('k').rows())
print('S2 per S1:',g['k2'].value_counts().sort('k2').rows())
print('S3 per S1:',g['k3'].value_counts().sort('k3').rows())
ex=g.select('source1_entity_id','m').explode('m').drop_nulls()
print('total links',ex.height,'unique targets',ex['m'].n_unique())
ids=pl.concat([s2['entity_id'],s3['entity_id']])
print('S2+S3 records',ids.len(),'linked',ex['m'].n_unique(),'frac unlinked',1-ex['m'].n_unique()/ids.len())
print('targets not in files',(~ex['m'].is_in(ids.implode())).sum())
# country of s1 vs singleton rate
g2=g.join(s1,left_on='source1_entity_id',right_on='entity_id')
print(g2.group_by('country').agg(pl.len(),(pl.col('k')==0).mean().alias('singleton'),pl.col('k').mean().alias('mean_k')).rows())
# cross-country links
allr=pl.concat([s2,s3])
x=ex.join(s1.select('entity_id',pl.col('country').alias('c1')),left_on='source1_entity_id',right_on='entity_id').join(allr.select('entity_id',pl.col('country').alias('c2')),left_on='m',right_on='entity_id')
print('country agree',(x['c1']==x['c2']).mean(), x.filter(pl.col('c1')!=pl.col('c2')).group_by('c1','c2').len().rows()[:10])
