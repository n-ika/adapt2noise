# Adaptation to speech in noise through selective attention

This repository contains the code for simulating the computational experiment of modeling speech adaptation.

![plot](./figs/me_vae.jpeg) 

To run the entire experiment, run:
`python main.py`

The flags/arguments are set by default as those used in the experiment. You can manually change them:

`optional arguments:  `<br/>
` -h, --help            show this help message and exit  `<br/>
` -t {True,False}, --train {True,False}  `<br/>
`                       train the model from scratch; if false will load  `<br/>
`                       weights  `<br/>
` -p PATH, --path PATH  name the path for saving or loading model weights  `<br/>
` -e NUM_EPOCHS, --num_epochs NUM_EPOCHS  `<br/>
`                       number of epochs to train the model  `<br/>
` -bs BATCH_SIZE, --batch_size BATCH_SIZE  `<br/>
`                       batch size for training  `<br/>
` -l LATENT, --latent LATENT  `<br/>
`                       vae latent dimension size  `<br/>
` -d CAT_DIM, --cat_dim CAT_DIM  `<br/>
`                       dimension of the category outcome  `<br/>
` -c CAT_SIZE, --cat_size CAT_SIZE  `<br/>
`                       number of the decision model units  `<br/>
` -b BETA, --beta BETA  value of parameter beta (multiplied with KL  `<br/>
`                       divergence)  `<br/>
` -w WEIGHT, --weight WEIGHT  `<br/>
`                       weight of the upweighted feature, downweighted feature  `<br/>
`                       is 1-w  `<br/>
` -lr LEARNING_RATE, --learning_rate LEARNING_RATE  `<br/>
`                       learning rate of the vae  `<br/>
` -v {True,False}, --variance {True,False}  `<br/>
`                       variance on one of the dimensions of the input - for  `<br/>
`                       each encoder, the variance is set on one unique dim  `<br/>
` -s SEED, --seed SEED  random seed for generating data  `<br/>
` -pe PER_ENC1, --per_enc1 PER_ENC1  `<br/>
`                       percentage of encoder 1 for weighting the information  `<br/>
`                       from the decision model; percentage of encoder 2 is  `<br/>
`                       1-per_enc1  `<br/>
` -pd {True,False}, --plot_data {True,False}  `<br/>
`                       plot the data?  `<br/>
` -ad {True,False}, --analyze_data {True,False}  `<br/>
`                       analyze the data?`<br/>

To run just the analysis and plot the data with an already trained model, run:
`python main.py -t=False`

To bypass plotting the data:
`python main.py -pd=False`

