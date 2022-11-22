import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn.utils import shuffle
from itertools import product
import seaborn as sns
from me_vae import *
from util_fun import *

def plot_dec_boundary(all_points,decision,per_enc,enc_num):
    dec_lbl = decision.numpy()
    dec_lbl[dec_lbl<0.5] = 0
    dec_lbl[dec_lbl>=0.5] = 1
    dec_lbl2 = np.flip(dec_lbl)
    plt.figure(figsize=(10,10))
    plt.scatter(x=all_points[:,0], y=all_points[:,1], c=dec_lbl2, alpha=0.5)
    plt.xlabel("VOT (ms)")
    plt.ylabel("f0 (Hz)")
    plt.legend(labels=["b","p"])
    plt.xticks(ticks=np.arange(-30,31,10),labels=[5*x for x in range(1,8)])
    plt.yticks(ticks=np.arange(-30,31,10),labels=[20*x+200 for x in range(0,7)])
    plt.savefig("./figs/dec_enc"+str(enc_num)+"_"+str(per_enc)+".jpg")

    return()

def plot_recon(enc_num,stims_train,test_03,test_30,test_33,test_36,test_63,preds_03,preds_30,preds_33,preds_36,preds_63):
    plt.figure(figsize=(15,15))
    plt.scatter(x=stims_train[:,0], y=stims_train[:,1], c="silver", alpha=0.5)
    plt.scatter(x=preds_36[:,0], y=preds_36[:,1], c="rosybrown", alpha=0.5)
    plt.scatter(x=preds_63[:,0], y=preds_63[:,1], c="salmon", alpha=0.5)
    plt.scatter(x=preds_30[:,0], y=preds_30[:,1], c="lemonchiffon", alpha=0.5)
    plt.scatter(x=preds_03[:,0], y=preds_03[:,1], c="yellowgreen", alpha=0.5)
    plt.scatter(x=preds_33[:,0], y=preds_33[:,1], c="turquoise", alpha=0.5)
    plt.scatter(x=test_36[:,0], y=test_36[:,1], c="indianred", alpha=0.5)
    plt.scatter(x=test_63[:,0], y=test_63[:,1], c="lightsalmon", alpha=0.5)
    plt.scatter(x=test_30[:,0], y=test_30[:,1], c="khaki", alpha=0.5)
    plt.scatter(x=test_03[:,0], y=test_03[:,1], c="olivedrab", alpha=0.5)
    plt.scatter(x=test_33[:,0], y=test_33[:,1], c="lightseagreen", alpha=0.5)
    plt.xlabel("VOT")
    plt.ylabel("f0")
    plt.legend(labels=["Train","Preds1","Preds2","Preds3","Preds4","Preds5","Test1","Test2","Test3","Test4","Test5"])
    plt.xticks(ticks=np.arange(-30,31,10),labels=[5*x for x in range(1,8)])
    plt.yticks(ticks=np.arange(-30,31,10),labels=[20*x+200 for x in range(0,7)])
    plt.savefig("./figs/recon_enc"+str(enc_num)+".jpg")

    return()

def plot_recon_dec(vae,per_enc1):
    per_enc2 = 1 - per_enc1

    stims_train, labels_train = make_VAE_dataset(canonical=True, N=100)
    test_36 = test_stims(vot=4,f0=7,N=100)
    test_63 = test_stims(vot=7,f0=4,N=100)
    test_30 = test_stims(vot=4,f0=1,N=100)
    test_03 = test_stims(vot=1,f0=4,N=100)
    test_33 = test_stims(vot=4,f0=4,N=100)

    preds_36_e1,preds_36_e2 = recon_data(vae,test_36)
    preds_63_e1,preds_63_e2 = recon_data(vae,test_63)
    preds_30_e1,preds_30_e2 = recon_data(vae,test_30)
    preds_03_e1,preds_03_e2 = recon_data(vae,test_03)
    preds_33_e1,preds_33_e2 = recon_data(vae,test_33)

    plot_recon(1,stims_train,test_03,test_30,test_33,test_36,test_63,preds_03_e1,preds_30_e1,preds_33_e1,preds_36_e1,preds_63_e1)
    plot_recon(2,stims_train,test_03,test_30,test_33,test_36,test_63,preds_03_e2,preds_30_e2,preds_33_e2,preds_36_e2,preds_63_e2)

    VOTs = np.arange(-30,31)
    f0s = np.arange(-30,31)
    all_points = np.array(list(product(VOTs,f0s)))
    all_points = np.expand_dims(all_points, -1).astype("float32")
    all_points = np.expand_dims(all_points, -1).astype("float32")

    decision1,decision2,decision_per1,decision_per2 = dec_data(vae,all_points,per_enc1)
    
    plot_dec_boundary(all_points,decision1,100,1)
    plot_dec_boundary(all_points,decision2,100,2)
    plot_dec_boundary(all_points,decision_per1,per_enc1*100,12)
    plot_dec_boundary(all_points,decision_per2,per_enc1*100,21)

    return()


