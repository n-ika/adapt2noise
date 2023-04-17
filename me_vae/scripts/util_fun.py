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
    """
    Make stimuli in 7-step continuum for each dimension
    and add variance around each point to increase number
    of trainind points. This function creates datapoints
    for a single category.
    
    Parameters
    ----------
    start_v : num
        index of the continuum to start with for VOT
    stop_v : num
        index of the continuum to stop with for VOT
    start_f : num
        index of the continuum to start with for F0
    stop_f : num
        index of the continuum to stop with for F0
    n_ex : num
        number of points to add around each point of the continuum
    var : num
        variance of the normal distribution to determine how
        widely we want the additional points to vary around the
        original 7 points of the continuum
    Returns
    -------
    stims : np array
        stimuli as a np array
    """
    VOTs = np.linspace(start=-3,stop=3, num=9)
    f0s = np.linspace(start=-3,stop=3, num=9)
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
    """
    Make stimuli in 7-step continuum for each sound with typical 
    (canonical) correlation or not (mis-correlation): create stimuli
    for both categories (p and b like values).
    
    Parameters
    ----------
    N : num
        number of points to add around each point of the continuum
    canonical : bool
        true if making typically correlated data, false otherwise
    Returns
    -------
    b_canon,p_canon : np arrays
        b like stimuli with normally correlated VOT and F0
        p like stimuli with normally correlated VOT and F0
    b_reverse,p_reverse : np arrays
        b like stimuli with mis-correlated VOT and F0
        p like stimuli with mis-correlated VOT and F0
    """
    if canonical==True:
        b_canon=make_noisy_stims(start_v=1,stop_v=4,start_f=1,stop_f=4,n_ex=N,var=2)
        p_canon=make_noisy_stims(start_v=-4,stop_v=-1,start_f=-4,stop_f=-1,n_ex=N,var=2)
        return(b_canon,p_canon)
    else:
        b_reverse=make_noisy_stims(start_v=1,stop_v=4,start_f=-4,stop_f=-1,n_ex=N,var=2)
        p_reverse=make_noisy_stims(start_v=-4,stop_v=-1,start_f=1,stop_f=4,n_ex=N,var=2)
        return(b_reverse,p_reverse)
    
def make_VAE_dataset(canonical=True, N=100):
    """
    Make stimuli for training the VAE: b and p like stimuli,
    shuffled so they do not appear in any specific order.
    
    Parameters
    ----------
    N : num
        number of points to add around each point of the continuum
    canonical : bool
        true if making typically correlated data, false otherwise
    Returns
    -------
    stims_shuff : np arrays
        b and p like stimuli with correlated VOT and F0, shuffled
    labels_shuff : np arrays
        labels (0 or 1) corresponding to the b and p like values
    """
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
    """
    Make stimuli for testing the VAE: call make_noisy_stims to create
    desired datapoints for defined vot and f0 values
    
    Parameters
    ----------
    vot : num
        vot value expressed as index of the 7-step continuum
    f0 : num
        f0 value expressed as index of the 7-step continuum
    N : num
        number of points to add around each point of the continuum
    var : num
        variance of the normal distribution to determine how
        widely we want the additional points to vary around the
        original 7 points of the continuum
    Returns
    -------
    stims_shuff : np arrays
        b and p like stimuli with correlated VOT and F0, shuffled
    labels_shuff : np arrays
        labels (0 or 1) corresponding to the b and p like categories
    """
    stims = make_noisy_stims(start_v=vot,stop_v=int(vot+1),start_f=f0,stop_f=int(f0+1),n_ex=N,var=var)
    stims_shuff = shuffle(stims,random_state=4)
    stims_shuff = np.expand_dims(stims_shuff, -1).astype("float32")
    return(stims_shuff)

def get_z(t,vae):
    """
    Make stimuli for testing the VAE.
    
    Parameters
    ----------
    t : tf tensor or np array
        stimuli to encode
    vae : tf model
        VAE fully trained model
    Returns
    -------
    z1,z2 : tf tensors
        latent variables corresponding to the data encoded with
        encoder 1 (z1) and encoder 2 (z2)
    """
    mu1, sig1, z1 = vae.encoder1.predict(t)
    mu2, sig2, z2 = vae.encoder2.predict(t)
    return(z1,z2)

def recon_data(vae,test):
    """
    Reconstruct the data using VAE.
    
    Parameters
    vae : tf model
        VAE fully trained model
    test : tf tensor or np array
        stimuli to reconstruct
    Returns
    -------
    z1,z2 : tf tensors
        latent variables corresponding to the data encoded with
        encoder 1 (z1) and encoder 2 (z2)
    """
    z1,z2 = get_z(test,vae)
    preds1 = vae.decoder(z1)
    preds2 = vae.decoder(z2)
    return(preds1,preds2)

def dec_data(vae,test,perc_low):
    """
    Reconstruct the data using VAE.
    
    Parameters
    vae : tf model
        VAE fully trained model
    test : tf tensor or np array
        stimuli to reconstruct
    perc_low : num
        the percentage of using encoder's latent dimension in category mapping 
        i.e. 10% of encoder 1, 90% of encoder 2
    Returns
    -------
    dec1 : tf tensor
        category decision based on z1 (encoder 1)
    dec2 : tf tensor
        category decision based on z2 (encoder 2)
    dec_e1low : tf tensor
        category decision based on low percentage * z1 + high percentage * z2 (encoder 1 and 2)
    dec_e1high : tf tensor
        category decision based on high percentage * z1 + low percentage * z2 (encoder 1 and 2)
    """
    perc_high = 1 - perc_low
    z1,z2 = get_z(test,vae)
    dec1 = vae.dec(z1)
    dec2 = vae.dec(z2)
    dec_e1low = vae.dec(tf.multiply(perc_low,z1)+tf.multiply(perc_high,z2))
    dec_e1high = vae.dec(tf.multiply(perc_high,z1)+tf.multiply(perc_low,z2))
    return(dec1,dec2,dec_e1low,dec_e1high)

