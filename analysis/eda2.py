import polars as pl, io_utils as u, random, sys
sys.stdout.reconfigure(encoding='utf-8')
s1=u.load('train','source1'); gt=u.load('train','ground_truth')
allr=pl.concat([u.load('train','source2'),u.load('train','source3')])
rec={r[0]:r for r in allr.iter_rows()}
s1d={r[0]:r for r in s1.iter_rows()}
random.seed(1)
rows=gt.rows(); 
for cid in ['US','India']:
  n=0
  for s,m in random.sample(rows,400):
    if s1d[s][3]!=cid: continue
    print('##',s1d[s][1],' | ',s1d[s][2])
    for x in (m.split(',') if m else []): print('   ',x[:2],rec[x][1],' | ',rec[x][2])
    n+=1
    if n==8: break
