import argparse
import timeit
from rdt import *
from util_fun import *
from plot_data import *
from make_analysis import *

if __name__ == '__main__':
     
     parser = argparse.ArgumentParser()

     # Model configuration.
     parser.add_argument('-t', '--train', choices=('y','n'), default='y',
          help='train the model from scratch; if false will load weights, default = y')
     parser.add_argument('-p', '--path', type=str, default='./models/vae_prior-vot_var-0/ckpt', 
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
     parser.add_argument('-pd', '--plot_data', choices=('y','n'), default='y',
          help='plot the data? default = y')
     parser.add_argument('-ad', '--analyze_data', choices=('y','n'), default='n',
          help='analyze the data? default = n')

     args = parser.parse_args()
     print(args)

     start = timeit.default_timer()

     vae = train_model(args.path,args.train,args.num_epochs,\
          args.batch_size,args.variance,args.weight,args.beta,\
          args.learning_rate,args.latent,args.cat_dim,args.cat_size)
     if args.plot_data == 'y':
          plot_recon_dec(vae)
     if args.analyze_data == 'y':
          metadata = do_analysis(vae)

     stop = timeit.default_timer()
     print('Time: ', stop - start)
     print('DONE')  
