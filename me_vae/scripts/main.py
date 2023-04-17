import argparse
import timeit
from me_vae import *
from util_fun import *
from plot_data import *
from make_analysis import *
from extract_strf import *
import numpy as np

if __name__ == '__main__':
     
     parser = argparse.ArgumentParser()

     # Model configuration.
     parser.add_argument('-t', '--train', choices=('y','n'), default='y',
          help='train the model from scratch; if false will load weights, default = y')
     parser.add_argument('-p', '--path', type=str, default='./models/vae_prior-vot-f0_var-f0-vot/ckpt', 
          help='name the path for saving or loading model weights')
     parser.add_argument('-e', '--num_epochs', type=int, default=10, 
          help='number of epochs to train the model, default = 10')
     parser.add_argument('-bs', '--batch_size', type=int, default=32, 
          help='batch size for training, default = 32')
     parser.add_argument('-l', '--latent', type=int, default=500, 
          help='vae latent dimension size, default = 500')
     parser.add_argument('-d', '--cat_dim', type=int, default=1, 
          help='dimension of the category outcome, default = 1')
     parser.add_argument('-c', '--cat_size', type=int, default=100, 
          help='number of the category model units, default = 100')
     parser.add_argument('-b', '--beta', type=float, default=0.0025, 
          help='value of parameter beta (multiplied with KL divergence), default = 0.0025')
     parser.add_argument('-w', '--weight', type=float, default=0.99, 
          help='weight of the upweighted feature, downweighted feature is 1-w, default = 0.99')
     parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, 
          help='learning rate of the vae, default is 0.001')
     parser.add_argument('-v', '--variance', choices=('y','n'), default='y',
          help='variance on one of the dimensions of the input - for each encoder, \
          the variance is set on one unique dim, default = y')
     parser.add_argument('-s', '--seed', type=int, default=27, 
          help='random seed for generating data, default = 27')
     parser.add_argument('-pe', '--per_enc1', type=float, default=0.8, 
          help='percentage of encoder 1 for weighting the information from \
          the decision model; percentage of encoder 2 is 1-per_enc1. Default = 0.8')
     parser.add_argument('-pd', '--plot_data', choices=('y','n'), default='y',
          help='plot the data? default = y')
     parser.add_argument('-ad', '--analyze_data', choices=('y','n'), default='y',
          help='analyze the data? default = y')
     parser.add_argument('-es', '--extract_strf', choices=('y','n'), default='y',
          help='extract the strfs? default = y')
     parser.add_argument('-cr', '--corpus', type=str, default=None,
          help='take data from corpus -- if so specify name, else will create data, default = None')
     

     args = parser.parse_args()
     print(args)

     start = timeit.default_timer()

     vae = train_model(args.path,args.train,args.num_epochs,\
          args.batch_size,args.variance,args.weight,args.beta,\
          args.learning_rate,args.latent,args.cat_dim,args.cat_size,args.corpus)
     if args.plot_data == 'y':
          plot_recon_dec(vae,args.per_enc1)
     if args.analyze_data == 'y':
          metadata = do_analysis(vae)

     # VOTs = np.arange(-30,31)
     # f0s = np.arange(-30,31)
     # all_points = np.array(list(product(VOTs,f0s)))
     # all_points_ = np.expand_dims(all_points, -1).astype("float32")
     # all_points_ = np.expand_dims(all_points_, -1).astype("float32")
 
     # z1,z2 = get_z(all_points_,vae)
 
     # for x in range(500):
     #     neur_activ1 = np.array([z1[i][x] for i in range (z1.shape[0])]) #257
     #     neur_activ2 = np.array([z2[i][x] for i in range (z2.shape[0])])
 
     #     coordinates = pd.DataFrame(all_points)
 
     #     coordinates['value1'] = neur_activ1
     #     coordinates['value2'] = neur_activ2
     #     strf1 = pd.pivot(coordinates, index=0, columns=1, values='value1')
     #     strf2 = pd.pivot(coordinates, index=0, columns=1, values='value2')
 
     #     sns.heatmap(strf1)

     corpus = "wsj"
     data_csv = pd.read_csv('./'+corpus+'_vowels_norm.csv')
     data_csv['label'] = 0
     data_csv.loc[(data_csv.phone == 'AE'), 'label'] = 1
     labels_train = np.array(data_csv.label)
     labels_train = np.expand_dims(labels_train, -1).astype("float32")
     stims_train = np.array(data_csv[['norm_f1','norm_duration']])
     VOTs = np.arange(-10,10.1,0.1)
     f0s = np.arange(-10,10.1,0.1)
     all_points = np.array(list(product(VOTs,f0s)))
     all_points = np.expand_dims(all_points, -1).astype("float32")
     all_points = np.expand_dims(all_points, -1).astype("float32")
     z1,z2=get_z(all_points,vae)
     preds1 = vae.decoder(z1)
     preds2 = vae.decoder(z2)
     dec1 = vae.dec(z1)
     dec2 = vae.dec(z2)
     plt.figure(figsize=(15,15))
     plt.scatter(x=all_points[:,0], y=all_points[:,1], alpha=0.5)
     plt.scatter(x=preds1[:,0], y=preds1[:,1], alpha=0.5)
     plt.xlabel("VOT (z-scored)",fontsize=40)
     plt.ylabel("f0 (z-scored)",fontsize=40)
     plt.legend(labels=["b","p"])
     plt.savefig("./figs/"+corpus+"_preds_enc1.jpg")

     plt.figure(figsize=(15,15))
     plt.scatter(x=all_points[:,0], y=all_points[:,1], alpha=0.5)
     plt.scatter(x=preds2[:,0], y=preds2[:,1], alpha=0.5)
     plt.xlabel("VOT (z-scored)",fontsize=40)
     plt.ylabel("f0 (z-scored)",fontsize=40)
     plt.legend(labels=["b","p"])
     plt.savefig("./figs/"+corpus+"_preds_enc2.jpg")

     dec_lbl = dec1.numpy()
     dec_lbl[dec_lbl<0.5] = 0
     dec_lbl[dec_lbl>=0.5] = 1
     dec_lbl2 = np.flip(dec_lbl)
     plt.figure(figsize=(10,10))
     plt.scatter(x=all_points[:,0], y=all_points[:,1], c=dec_lbl2, alpha=0.5)
     plt.xlabel("VOT (z-scored)",fontsize=40)
     plt.ylabel("f0 (z-scored)",fontsize=40)
     plt.legend(labels=["b","p"])
     plt.savefig("./figs/"+corpus+"_dec_enc1.jpg")

     dec_lbl = dec2.numpy()
     dec_lbl[dec_lbl<0.5] = 0
     dec_lbl[dec_lbl>=0.5] = 1
     dec_lbl2 = np.flip(dec_lbl)
     plt.figure(figsize=(10,10))
     plt.scatter(x=all_points[:,0], y=all_points[:,1], c=dec_lbl2, alpha=0.5)
     plt.xlabel("VOT (z-scored)",fontsize=40)
     plt.ylabel("f0 (z-scored)",fontsize=40)
     plt.legend(labels=["b","p"])
     plt.savefig("./figs/"+corpus+"_dec_enc2.jpg")

     if args.extract_strf == 'y':
          get_strfs(vae)

     stop = timeit.default_timer()
     print('Time: ', stop - start)
     print('DONE')  
