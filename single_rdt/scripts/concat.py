import pandas as pd
import numpy as np
import sys
import os


corpus=sys.argv[1]
ID=sys.argv[2]
N=int(sys.argv[3])
WEIGHT = sys.argv[4]

path = './analysis/dfs/w_'+WEIGHT
if not os.path.exists(path):
    os.makedirs(path)

dfs = []
for num in range(1,N):
  df=pd.read_csv('./analysis/'+corpus+'/w_'+WEIGHT+'/'+ID+'_'+str(num)+'_test.csv')
  print("analyzing :"+corpus+str(num))
  df.subject = num
  df['choice'] = np.where(df.dec >= 0.5, 1, 0)
  dfs.append(df)
fin = pd.concat(dfs)
fin.to_csv(path+'/test_'+corpus+'.csv')

