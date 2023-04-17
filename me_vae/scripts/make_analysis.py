import numpy as np
from itertools import product
from util_fun import *

def get_avg(dec):
    dec_avg = np.average(dec)
    return(dec_avg)

def get_dec(data,per,vae):
    data = np.expand_dims(data, axis=(2,3)).astype("float32")
    _,_,dec_e1low,dec_e1high=dec_data(vae,data,per*0.01)
    dec_e1low_avg=get_avg(dec_e1low)
    dec_e1high_avg=get_avg(dec_e1high)
    return(dec_e1low_avg,dec_e1high_avg)

def do_analysis(vae,percentage=100):
    res = {'vot':[],'f0':[],'dec':[],'condition':[],'subject':[],'accuracy':[]}
    for per in range(0,percentage,1):
        per2 = abs(100-per)
        VOTs = np.arange(-30,31)
        f0s = np.arange(-30,31)

        upper = np.array(list(product(VOTs,f0s[10:21])))
        lower = np.array(list(product(VOTs,f0s[-20:-9])))
        left = np.array(list(product(VOTs[-20:-9],f0s)))
        right = np.array(list(product(VOTs[10:21],f0s)))

        e1low_left,e1high_left = get_dec(left,per,vae)
        e1low_right,e1high_right = get_dec(right,per,vae)
        e1low_upper,e1high_upper = get_dec(upper,per,vae)
        e1low_lower,e1high_lower = get_dec(lower,per,vae)

        res['subject'].append(str(per)+'_'+str(per2))
        res['vot'].append(2)
        res['f0'].append(4)
        res['dec'].append(e1high_left)
        res['condition'].append('clear')
        res['accuracy'].append(e1high_left)
        res['subject'].append(str(per)+'_'+str(per2))
        res['vot'].append(2)
        res['f0'].append(4)
        res['dec'].append(e1low_left)
        res['condition'].append('noise')
        res['accuracy'].append(e1low_left)

        res['subject'].append(str(per)+'_'+str(per2))
        res['vot'].append(6)
        res['f0'].append(4)
        res['dec'].append(e1high_right)
        res['condition'].append('clear')
        res['accuracy'].append(e1high_left)
        res['subject'].append(str(per)+'_'+str(per2))
        res['vot'].append(6)
        res['f0'].append(4)
        res['dec'].append(e1low_right)
        res['condition'].append('noise')
        res['accuracy'].append(e1low_left)

        res['subject'].append(str(per)+'_'+str(per2))
        res['vot'].append(4)
        res['f0'].append(2)
        res['dec'].append(e1low_upper)
        res['condition'].append('noise')
        res['accuracy'].append(e1low_lower)
        res['subject'].append(str(per)+'_'+str(per2))
        res['vot'].append(4)
        res['f0'].append(2)
        res['dec'].append(e1high_upper)
        res['condition'].append('clear')
        res['accuracy'].append(e1high_lower)

        res['subject'].append(str(per)+'_'+str(per2))
        res['vot'].append(4)
        res['f0'].append(6)
        res['dec'].append(e1low_lower)
        res['condition'].append('noise')
        res['accuracy'].append(e1low_lower)
        res['subject'].append(str(per)+'_'+str(per2))
        res['vot'].append(4)
        res['f0'].append(6)
        res['dec'].append(e1high_lower)
        res['condition'].append('clear')
        res['accuracy'].append(e1high_lower)
        
    metadata = pd.DataFrame(res)
    metadata.to_csv('./analysis.csv')

    return(metadata)
