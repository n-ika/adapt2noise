# Adaptation to speech in noise through selective attention

This repository contains the code for simulating the computational experiment of modeling speech adaptation.

This experiment modeled how human listeners can flexibly weight the speech features according to their reliability, such that they can adapt to any environment or speaker. The structure of the model is a $\beta$-ME-VAE (multiple encoder auto encoder). The encoders' information is probabilistically weighted to simulate the fact that listeners rely more on some features than the other in specific situations. The human behavior modeled in this project is .

![plot](./figs/me_vae.jpeg) 

## Dependencies
The dependencies needed to run this code:<br/>
`tensorflow`<br/>
`keras`<br/>
`itertools`<br/>
`scipy`<br/>
`sklearn`<br/>
`numpy`<br/>
`pandas`<br/>
`matplotlib`<br/>

## Run Experiment
To run the entire experiment, run:
`python main.py`

## Optional Arguments
The flags/arguments are set by default as those used in the experiment. You can manually change them:

optional arguments:   <br/>
` -h, --help           ` <br/>
&emsp; &emsp;  show this help message and exit   <br/>
` -t {True,False}, --train {True,False} ` <br/>
&emsp; &emsp;  train the model from scratch; if false will load weights <br/>
` -p PATH, --path PATH` <br/>
&emsp; &emsp;  name the path for saving or loading model weights <br/>
` -e NUM_EPOCHS, --num_epochs NUM_EPOCHS` <br/>
&emsp; &emsp;  number of epochs to train the model <br/>
` -bs BATCH_SIZE, --batch_size BATCH_SIZE` <br/>
&emsp; &emsp;  batch size for training   <br/>
` -l LATENT, --latent LATENT ` <br/>
&emsp; &emsp;  vae latent dimension size   <br/>
` -d CAT_DIM, --cat_dim CAT_DIM ` <br/>
&emsp; &emsp;  dimension of the category outcome   <br/>
` -c CAT_SIZE, --cat_size CAT_SIZE ` <br/>
&emsp; &emsp;  number of the decision model units   <br/>
` -b BETA, --beta BETA ` <br/>
&emsp; &emsp;  value of parameter beta (multiplied with KL divergence)    <br/>
` -w WEIGHT, --weight WEIGHT ` <br/>
&emsp; &emsp;  weight of the upweighted feature, downweighted feature is 1-w  <br/>
` -lr LEARNING_RATE, --learning_rate LEARNING_RATE ` <br/>
&emsp; &emsp;  learning rate of the vae   <br/>
` -v {True,False}, --variance {True,False} ` <br/>
&emsp; &emsp;  variance on one of the dimensions of the input - for each encoder, the variance is set on one unique dim   <br/>
` -s SEED, --seed SEED ` <br/>
&emsp; &emsp;  random seed for generating data   <br/>
` -pe PER_ENC1, --per_enc1 PER_ENC1 ` <br/>
&emsp; &emsp;  percentage of encoder 1 for weighting the information from the decision model; percentage of encoder 2 is 1-per_enc1   <br/>
` -pd {True,False}, --plot_data {True,False} ` <br/>
&emsp; &emsp;  plot the data?   <br/>
` -ad {True,False}, --analyze_data {True,False} ` <br/>
&emsp; &emsp;  analyze the data? <br/>


## Using pretrained model
To run just the analysis and plot the data with an already trained model, run:
`python main.py -t=False`

To bypass plotting the data:
`python main.py -pd=False`

