import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib as plt
from me_vae import *
from util_fun import *


def plot_strf(strf,x,enc_num):
    # g.set_xticks(range(len(strf)))
    # g.set_yticks(range(len(strf)))
    # g.set_xticklabels([5*x for x in range(0,7)])
    # g.set_yticklabels([20*x+200 for x in range(0,7)]) # ticks=np.arange(-30,31,10),labels=
    fig = sns.heatmap(strf,xticklabels=False,yticklabels=False,square=True).get_figure()
    fig.savefig("./figs/strfs/e"+str(enc_num)+"_strf_"+str(x)+".jpg",dpi=200)
    fig.clf()
    return()

def get_strfs(vae):
    VOTs = np.arange(-30,31)
    f0s = np.arange(-30,31)
    all_points = np.array(list(product(VOTs,f0s)))
    all_points_ = np.expand_dims(all_points, -1).astype("float32")
    all_points_ = np.expand_dims(all_points_, -1).astype("float32")

    z1,z2 = get_z(all_points_,vae)

    for x in range(500):
        neur_activ1 = np.array([z1[i][x] for i in range (z1.shape[0])]) #257
        neur_activ2 = np.array([z2[i][x] for i in range (z2.shape[0])])

        coordinates = pd.DataFrame(all_points)
        coordinates = coordinates.rename(columns={0: "VOT", 1: "F0"})
        coordinates['value1'] = neur_activ1
        coordinates['value2'] = neur_activ2
        strf1 = pd.pivot(coordinates, index="VOT", columns="F0", values='value1').transpose()
        strf2 = pd.pivot(coordinates, index="VOT", columns="F0", values='value2').transpose()
        plot_strf(strf1,x,1)
        plot_strf(strf2,x,2)

    return()