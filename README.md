# Adaptation to speech in noise through selective attention

This repository contains the code for simulating the computational experiment of modeling speech adaptation.

![plot](./figs/me_vae.png) 

To run the entire experiment, run:
`python main.py`

The flags/arguments are set by default as those used in the experiment. You can manually change them:

     '-t', '--train', choices=('True','False', default=True,
          help='train the model from scratch; if false will load weights'
     '-p', '--path', type=str, default='./models/vae_prior-vot-f0_var-f0-vot/ckpt', 
          help='name the path for saving or loading model weights'
     '-e', '--num_epochs', type=int, default=10, 
          help='number of epochs to train the model'
     '-bs', '--batch_size', type=int, default=32, 
          help='batch size for training'
     '-l', '--latent', type=int, default=500, 
          help='vae latent dimension size'
     '-d', '--cat_dim', type=int, default=1, 
          help='dimension of the category outcome'
     '-c', '--cat_size', type=int, default=100, 
          help='number of the category model units'
     '-b', '--beta', type=float, default=0.0025, 
          help='value of parameter beta (multiplied with KL divergence'
     '-w', '--weight', type=float, default=0.99, 
          help='weight of the upweighted feature, downweighted feature is 1-w'
     '-lr', '--learning_rate', type=float, default=0.001, 
          help='learning rate of the vae'
     '-v', '--variance', choices=('True','False', default=True,
          help='variance on one of the dimensions of the input - for each encoder, \
          the variance is set on one unique dim'
     '-s', '--seed', type=int, default=27, 
          help='random seed for generating data'
     '-pe', '--per_enc1', type=float, default=0.8, 
          help='percentage of encoder 1 for weighting the information from \
          the decision model; percentage of encoder 2 is 1-per_enc1'
     '-pd', '--plot_data', choices=('True','False', default=True,
          help='plot the data?'
     '-ad', '--analyze_data', choices=('True','False', default=True,
          help='analyze the data?'

To run just the analysis and plot the data with an already trained model, run:
`python main.py -t=False`

To bypass plotting the data:
`python main.py -pd=False`

