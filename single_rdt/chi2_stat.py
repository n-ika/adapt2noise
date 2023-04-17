import os
import pandas as pd
import math
import numpy as np

hum = pd.read_csv('/Users/nika/Documents/UMD/Projects/RDT_VAE/holt_dataset/SETSAT_Data/SetSat_Clear.csv')
cwd = os.getcwd()
df_path = os.path.abspath(cwd+'/analysis/dfs/w_')

res = {'corpus':[],'weight':[],'chi2':[],'p':[], 'lik_attn':[], 'lik_base':[]} #,'dur':[],'f1':[]
for corpus in ['timit','buc','wsj']:
    for weight in [0.7,0.8,0.85,0.9,0.99,0.999]:
        chi2_all = []
        likelihoods_base = []
        likelihoods_attn = []
        for dur in range(1,8):
            for f1 in range(1,8):
                ae_num = hum[(hum.spec==f1) & (hum.duration==dur) & (hum.Response=='sat') & (hum.Block=='Baseline')].shape[0]
                eh_num = hum[(hum.spec==f1) & (hum.duration==dur) & (hum.Response=='set') & (hum.Block=='Baseline')].shape[0]
                df_base = pd.read_csv(df_path+str(1.0)+'/wu_test_'+corpus+'.csv')
                df_base = df_base[(df_base.dur_step == dur) & (df_base.f1_step == f1)]
                weighted_path = df_path+str(weight)+'/wu_test_'+corpus+'.csv'
                if os.path.isfile(weighted_path):
                    df_weighted = pd.read_csv(weighted_path)
                    df_weighted = df_weighted[(df_weighted.dur_step == dur) & (df_weighted.f1_step == f1)]
                    stat_base = ae_num*math.log10(df_base.dec.mean())+eh_num*math.log10(1-df_base.dec.mean())
                    stat_weighted = ae_num*math.log10(df_weighted.dec.mean())+eh_num*math.log10(1-df_weighted.dec.mean())
                    likelihoods_base.append(stat_base)
                    likelihoods_attn.append(stat_weighted)

        chi2 = 2*(np.sum(likelihoods_attn)-np.sum(likelihoods_base))
        if chi2 > 10.83:
            p_val = "significant"
        else:
            p_val = "not"
        res['corpus'].append(corpus)
        res['weight'].append(weight)
        res['chi2'].append(chi2)
        res['p'].append(p_val)
        res['lik_attn'].append(np.sum(likelihoods_attn))
        res['lik_base'].append(np.sum(likelihoods_base))

df = pd.DataFrame(res)
df.to_csv(os.path.abspath(cwd+'/chi2.csv'))