import numpy as np
import pandas as pd
from scipy import stats
from sklearn.utils import shuffle
from itertools import product
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd

def make_noisy_stims(start_v=1,stop_v=4,start_f=1,stop_f=4,n_ex=100,var=1):
    VOTs = np.linspace(start=-30,stop=30, num=9)
    f0s = np.linspace(start=-30,stop=30, num=9)
    stims = []
    for VOT in VOTs[start_v:stop_v]:
        noisy_VOTs = np.random.default_rng().normal(VOT, var, n_ex)
        for f0 in f0s[start_f:stop_f]:
            noisy_f0s = np.random.default_rng().normal(f0, var, n_ex)
            for n_VOT in noisy_VOTs:
                for n_f0 in noisy_f0s:
                    stims.append([n_VOT,n_f0])
    return(np.array(stims))

def pb_stims(N=100,canonical=True):
    if canonical==True:
        b_canon=make_noisy_stims(start_v=1,stop_v=4,start_f=1,stop_f=4,n_ex=N,var=2)
        p_canon=make_noisy_stims(start_v=-4,stop_v=-1,start_f=-4,stop_f=-1,n_ex=N,var=2)
        return(b_canon,p_canon)
    else:
        b_reverse=make_noisy_stims(start_v=1,stop_v=4,start_f=-5,stop_f=-2,n_ex=N,var=1)
        p_reverse=make_noisy_stims(start_v=-5,stop_v=-2,start_f=1,stop_f=4,n_ex=N,var=1)
        return(b_reverse,p_reverse)
    
def make_VAE_dataset(canonical=True, N=100):
    np.random.seed(987654321)
    b_list,p_list = pb_stims(canonical=canonical,N=N)
    b_labels = np.zeros(b_list.shape[0])
    p_labels = np.ones(p_list.shape[0])
    stims = np.concatenate((b_list,p_list))
    labels = np.concatenate((b_labels,p_labels))
    # create a dataset for VAE training
     # shuffle all stims and according labels, then expand their dims for model training
    stims_shuff,labels_shuff = shuffle(stims,labels,random_state=4)
    stims_shuff = np.expand_dims(stims_shuff, -1).astype("float32")
    stims_shuff = np.expand_dims(stims_shuff, -1).astype("float32")
    labels_shuff = np.expand_dims(labels_shuff, -1).astype("float32")
    return(stims_shuff,labels_shuff)

def test_stims(vot=1,f0=3,N=100,var=1):
    stims = make_noisy_stims(start_v=vot,stop_v=int(vot+1),start_f=f0,stop_f=int(f0+1),n_ex=N,var=var)
    stims_shuff = shuffle(stims,random_state=4)
    stims_shuff = np.expand_dims(stims_shuff, -1).astype("float32")
    return(stims_shuff)

def get_z(t,vae):
    mu1, sig1, z1 = vae.encoder1.predict(t)
    mu2, sig2, z2 = vae.encoder2.predict(t)
    return(z1,z2)

def recon_data(vae,test,per_enc1):
    z1,z2 = get_z(test,vae)
    preds1 = vae.decoder(z1)
    preds2 = vae.decoder(z2)
    return(preds1,preds2)

def dec_data(vae,test,perc_low):
    perc_high = 1 - perc_low
    z1,z2 = get_z(test,vae)
    dec1 = vae.dec(z1)
    dec2 = vae.dec(z2)
    dec_e1low = vae.dec(tf.multiply(perc_low,z1)+tf.multiply(perc_high,z2))
    dec_e1high = vae.dec(tf.multiply(perc_high,z1)+tf.multiply(perc_low,z2))
    return(dec1,dec2,dec_e1low,dec_e1high)

