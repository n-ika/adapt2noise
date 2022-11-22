import argparse
import timeit
from me_vae import *
from util_fun import *
from plot_data import *
from make_analysis import *

if __name__ == '__main__':
     
     parser = argparse.ArgumentParser()

     # Model configuration.
     parser.add_argument('-t', '--train', choices=('True','False'), default=True,
          help='train the model from scratch; if false will load weights')
     parser.add_argument('-p', '--path', type=str, default='./models/vae_prior-vot-f0_var-f0-vot/ckpt', 
          help='name the path for saving or loading model weights')
     parser.add_argument('-e', '--num_epochs', type=int, default=10, 
          help='number of epochs to train the model')
     parser.add_argument('-bs', '--batch_size', type=int, default=32, 
          help='batch size for training')
     parser.add_argument('-l', '--latent', type=int, default=500, 
          help='vae latent dimension size')
     parser.add_argument('-d', '--dec_dim', type=int, default=1, 
          help='dimension of the decision outcome')
     parser.add_argument('-ds', '--dec_size', type=int, default=100, 
          help='number of the decision model units')
     parser.add_argument('-b', '--beta', type=float, default=0.0025, 
          help='value of parameter beta (multiplied with KL divergence)')
     parser.add_argument('-w', '--weight', type=float, default=0.99, 
          help='weight of the upweighted feature, downweighted feature is 1-w')
     parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, 
          help='learning rate of the vae')
     parser.add_argument('-v', '--variance', choices=('True','False'), default=True,
          help='variance on one of the dimensions of the input - for each encoder, \
          the variance is set on one unique dim')
     parser.add_argument('-s', '--seed', type=int, default=27, 
          help='random seed for generating data')
     parser.add_argument('-pe', '--per_enc1', type=float, default=0.8, 
          help='percentage of encoder 1 for weighting the information from \
          the decision model; percentage of encoder 2 is 1-per_enc1')
     parser.add_argument('-pd', '--plot_data', choices=('True','False'), default=True,
          help='plot the data?')
     parser.add_argument('-ad', '--analyze_data', choices=('True','False'), default=True,
          help='analyze the data?')

     args = parser.parse_args()
     print(args)

     start = timeit.default_timer()

     vae = train_model(args.path,args.train,args.num_epochs,\
          args.batch_size,args.variance,args.weight,args.beta,\
          args.learning_rate,args.latent,args.dec_dim,args.dec_size)
     if args.plot_data == True:
          plot_recon_dec(vae,args.per_enc1)
     if args.analyze_data == True:
          metadata = do_analysis(vae)

     stop = timeit.default_timer()
     print('Time: ', stop - start)
     print('DONE')  
