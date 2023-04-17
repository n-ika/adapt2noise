import pandas as pd
import numpy as np
from itertools import product

# read test stim
# df = pd.read_json('set_sat_results.json',orient='index')
# df.columns=['f1','f2','dur']
# df['file'] = df.index
# df.file = df.file.str.replace('setsat/', '')
# df.file = df.file.str.replace('.wav', '')
# df.file = df.file.str.replace('dur', '')
# df.file = df.file.str.replace('spec', '')
# df[['dur_step','f1_step']] = df.file.str.split('_',expand=True)

spec = np.linspace(800,1000,7)
durs = np.linspace(0.175,0.475,7)
stim = np.array(list(product(durs,spec)))
step = np.array(list(product(np.arange(1,8),np.arange(1,8))))
df = pd.DataFrame(columns=["dur","f1"])
df["dur"] = stim[:,0]
df["f1"] = stim[:,1]
df["f1_step"] = step[:,1]
df["dur_step"] = step[:,0]



# read corpus data
for corpus in ["wsj","buc","timit"]:
    data_csv = pd.read_csv('./'+corpus+'_vowels_norm.csv')
    data_csv['label'] = 0
    data_csv.loc[(data_csv.phone == 'AE'), 'label'] = 1

    res = {'spk':[],'mu_f1':[],'mu_dur':[],'var_f1':[],'var_dur':[]}
    data_csv=data_csv[(data_csv.gender=='f')]
    for spk in data_csv.speaker.unique():
        # get mean and std
        f1_mean = data_csv[(data_csv.speaker==spk)].f1.mean()
        f1_var = data_csv [(data_csv.speaker==spk)].f1.std()**2
        dur_mean =data_csv[(data_csv.speaker==spk)].duration.mean()
        dur_var = data_csv[(data_csv.speaker==spk)].duration.std()**2

        res['mu_f1'].append(f1_mean)
        res['mu_dur'].append(dur_mean)
        res['var_f1'].append(f1_var)
        res['var_dur'].append(dur_var)

    # square std (of each separate spk) and then take average and then sq root
    mean_f1 = np.average(res['mu_f1'])
    mean_dur = np.average(res['mu_dur'])
    std_f1 = np.sqrt(np.average(res['var_f1']))
    std_dur = np.sqrt(np.average(res['var_dur']))

    # z score f1 and dur
    df['norm_f1_'+corpus] = (df.f1-mean_f1)/std_f1
    df['norm_dur_'+corpus] = (df.dur-mean_dur)/std_dur

    #find the z-scored step size  
    f1_diff = np.diff(df[(df.dur_step==1)]['norm_f1_'+corpus])[0]
    dur_diff = np.diff(df[(df.f1_step==1)]['norm_dur_'+corpus])[0]
    ratio = float(dur_diff)/float(f1_diff)
    
    #data_csv['norm_duration_fix']= data_csv['norm_duration'] / ratio
    #data_csv.to_csv(corpus+'_scaled.csv')
    df['scaled_dur_'+corpus] = df['norm_dur_'+corpus] / ratio
    df['z_step_dur_'+corpus] = dur_diff 
    df['z_step_f1_'+corpus] = f1_diff


df.to_csv('norm_test_wu.csv')