# Adaptation to speech in noise through selective attention

This repository contains the code for simulating the computational experiment of modeling speech adaptation.

![plot](./figs/me_vae.jpeg) 

To run the entire experiment, run:
`python main.py`

The flags/arguments are set by default as those used in the experiment. You can manually change them:

`optional arguments:`
`  -h, --help            show this help message and exit`
`  -t {True,False}, --train {True,False}`
`                        train the model from scratch; if false will load`
`                        weights`
`  -p PATH, --path PATH  name the path for saving or loading model weights`
`  -e NUM_EPOCHS, --num_epochs NUM_EPOCHS`
`                        number of epochs to train the model`
`  -bs BATCH_SIZE, --batch_size BATCH_SIZE`
`                        batch size for training`
`  -l LATENT, --latent LATENT`
`                        vae latent dimension size`
`  -d CAT_DIM, --cat_dim CAT_DIM`
`                        dimension of the category outcome`
`  -c CAT_SIZE, --cat_size CAT_SIZE`
`                        number of the decision model units`
`  -b BETA, --beta BETA  value of parameter beta (multiplied with KL`
`                        divergence)`
`  -w WEIGHT, --weight WEIGHT`
`                        weight of the upweighted feature, downweighted feature`
`                        is 1-w`
`  -lr LEARNING_RATE, --learning_rate LEARNING_RATE`
`                        learning rate of the vae`
`  -v {True,False}, --variance {True,False}`
`                        variance on one of the dimensions of the input - for`
`                        each encoder, the variance is set on one unique dim`
`  -s SEED, --seed SEED  random seed for generating data`
`  -pe PER_ENC1, --per_enc1 PER_ENC1`
`                        percentage of encoder 1 for weighting the information`
`                        from the decision model; percentage of encoder 2 is`
`                        1-per_enc1`
`  -pd {True,False}, --plot_data {True,False}`
`                        plot the data?`
`  -ad {True,False}, --analyze_data {True,False}`
                        `analyze the data?`

To run just the analysis and plot the data with an already trained model, run:
`python main.py -t=False`

To bypass plotting the data:
`python main.py -pd=False`

