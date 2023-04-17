import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Flatten
from util_fun import *
import os

class Sampling(layers.Layer):
    """
    Sample latent variable z given the mean and log variance from the
    last encoder's layer + reparametrization trick: mu + var*epsilon,
    where epsilon is randomly taken from a normal distribution N(0,1)
    
    Parameters
    ----------
    layer : tf layer
        layer containing mean and log variance
    Returns
    -------
    z :
        z = mu + var*epsilon
    """
    def call(self, inputs):
        mu, z_log_var = inputs
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return mu + tf.exp(0.5 * z_log_var) * epsilon

def build_encoder(input_shape = (64,64,1), latent_dim = 500):
    """
    Build encoder with 2 convolutional 2D layers, 1 dense 2000 unit layer,
    1 500 unit layer expressing the mean, 1 500 unit layer expressing the log
    variance.
    
    Parameters
    ----------
    input_shape : tuple
        shape of the input as a tuple
    latent_dim : num
        number of units for the latent dimension (mu, log_var and z will be this size)
    Returns
    -------
    mu, z_log_var, z : layer
        mu, z_log_var, z to be used by the decoder
    """
    encoder_inputs = keras.Input(shape=(input_shape))
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(2000, activation="relu")(x)
    mu = layers.Dense(latent_dim, name="mu")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([mu, z_log_var])
    encoder = keras.Model(encoder_inputs, [mu, z_log_var, z], name="encoder")

    return(encoder)

def build_decoder(latent_dim = 500, recall_dim = 2):
    """
    Build decoder with 1 dense 2 unit layer.
    
    Parameters
    ----------
    latent_dim : num
        number of units of the encoder's latent dimension, input to the decoder
    recall_dim : num
        size of the 'recalled' output, which corresponds to the 2D summarized input,
        containing i.e. only VOT and F0 values
    Returns
    -------
    recall : layer
        output of the recall_dim dimension
    """
    latent_inputs = keras.Input(shape=(latent_dim))
    # bridge = tf.keras.layers.Dense(8 * 8 * 64, activation="relu")(latent_inputs)
    # bridge = tf.reshape(bridge, [-1, 8, 8, 64])
    # conv1_dec = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same")(bridge)
    # conv2_dec = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same")(conv1_dec)
    # reconstruction = tf.keras.layers.Conv2DTranspose(1, 3, strides=2, padding="same", name="dec/reconstruction")(conv2_dec)
    recall = tf.keras.layers.Dense(recall_dim,activation=None)(latent_inputs)
    recall = tf.expand_dims(recall, -1)
    recall = tf.expand_dims(recall, -1)
    decoder = keras.Model(latent_inputs, recall, name="decoder")

    return(decoder)

def build_dec_model(input_dim=500, model_dim=100, dec_dim=1):
    """
    Build decoder with 1 dense 2 unit layer.
    
    Parameters
    ----------
    latent_dim : num
        number of units of the encoder's latent dimension, input to the decoder
    model_dim : num
        size of the units in the dense layer
    dec_dim : num
        dimension of the ouput
    Returns
    -------
    decision_sig : num
        output, number between 0 and 1, denoting the decision between 2 categories
    """
    latent_inputs = keras.Input(shape=(input_dim))
    decision_hidden = tf.keras.layers.Dense(model_dim,activation="relu",name="dec/decision_hidden")(latent_inputs)
    decision = tf.keras.layers.Dense(dec_dim,activation=None,name="dec/decision")(decision_hidden)
    decision_sig = tf.sigmoid(decision)
    decision = keras.Model(latent_inputs, decision_sig, name="dec")
    return(decision)

class VAE(keras.Model):
    """
    Build VAE.
    """
    def __init__(self, encoder, decoder, dec, cue_weights, beta=0.0025, **kwargs):
        """
        Initialize training.
        
        Parameters
        ----------
        encoder1 : tf model
            initialized encoder 1 for the VAE with appropriate architecture
        encoder2 : tf model
            initialized encoder 2 for the VAE with appropriate architecture
        decoder : tf model
            initialized decoder for the VAE with appropriate architecture
        dec : tf model
            initialized category model for the VAE with appropriate architecture
        cue_weights1 : np array
            2D array for feature weights multiplied with the MSE of encoder 1
        cue_weights2 : np array
            2D array for feature weights multiplied with the MSE of encoder 2
        beta : num
            parameter beta, scaling the KL divergence of the loss term
        """
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.dec = dec
        self.beta = beta
        self.cue_weights = cue_weights
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.recall_loss_tracker = keras.metrics.Mean(
            name="recall_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.decision_tracker = keras.metrics.Mean(name="dec_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.recall_loss_tracker,
            self.kl_loss_tracker,
            self.decision_tracker,
        ]

    def train_step(self, all_data):
        """
        Training step.
        
        Parameters
        ----------
        all_data : list
            list of input data: 
            0 - ideal typically correlated data,
            1 - data with more variance on F0 dimension
            2 - data with more variance on VOT dimension
            3 - category labels used to train the category model
        Returns
        -------
        trackers : num
            loss trackers containing the loss value for this training step
        """
        data_true = all_data[0][0]
        data_fake = all_data[0][1]
        cat_labels = all_data[0][2]
        with tf.GradientTape(persistent=True) as tape:
            mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
            bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

            mu, z_log_var, z = self.encoder(data_fake)
            recall = self.decoder(z)
            recall_loss = tf.reduce_mean(
                                  tf.reduce_sum(
                                      tf.math.multiply(self.cue_weights,mse(data_true,recall)), axis=(1, 2)
                                  )
                              )

            kl_loss =  -0.5 * (1 + z_log_var - tf.square(mu) - tf.exp(z_log_var))
            kl_loss = tf.math.multiply(np.array([self.beta],dtype='float32'),tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1)))


            decision = self.dec(z)
            decision_loss = bce(cat_labels,decision)
            total_loss = recall_loss + kl_loss + decision_loss

# average all losses
        grads = tape.gradient([total_loss], self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.recall_loss_tracker.update_state(recall_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.decision_tracker.update_state(decision_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "recall_loss": self.recall_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "dec_loss": self.decision_tracker.result(),
        }

def train_model(path,train,num_epochs,batch_size,variance,weight,beta,lr,latent,dec_dim,dec_size):
    """
    Train the beta-ME-VAE model.
    
    Parameters
    ----------
    path : str
        path to save newly trained model or to load the already trained one
    train : bool
        True if training the model, False if loading already trained one
    num_epochs : num
        number of epochs in training
    batch_size : num
        size of training batch
    variance : bool
        adding variance on one of the dimensions of the input data - if true,
        will add variance on dimension 2 for encoder 1 and on dimension 1 for 
        encoder 2, such that encoder 1 prioritizes dimension 1 and encoder 2
        prioritizes dimension 2
    weight : num
        feature weight for the higher weighted dimension, lower weighted dimension
        is 1-weight
    beta : num
        parameter beta, scaling the KL divergence of the loss term
    lr : num
        learning rate
    latent : num
        dimension of the size of the latent variable z (encoder's output and 
        decoder's input)
    dec_dim : num
        dimension of the ouput
    dec_size : num
        size of the units in the dense layer

    Returns
    -------
    vae : tf model
        full VAE model, either newly trained or loaded
    """
    model_path = os.path.abspath(path+'/ckpt')
    stims_train, labels_train = make_VAE_dataset(canonical=True, N=100)
    opt = tf.keras.optimizers.Adagrad(lr,clipnorm=50.)
    cue_weights = np.array([weight, float(1-weight)]) # VOT emphasized
    cue_weights = np.expand_dims(cue_weights, -1).astype("float32")
    encoder = build_encoder(latent_dim=latent,input_shape=stims_train.shape[1:]) # for VOT emphasized
    decoder = build_decoder(latent_dim=latent,recall_dim=stims_train.shape[1])
    dec_model = build_dec_model(input_dim=latent,model_dim=dec_size, dec_dim=dec_dim)
    vae = VAE(encoder, decoder, dec_model, cue_weights, beta)
    vae.compile(optimizer=opt)
    if train=='y':
        print("*************************training the model from scratch*************************")
        stm = stims_train.copy()
        if variance=='y':
            print("*************************using data with added variance*************************")
            stm[:,1,0,0] = np.random.default_rng().normal(stm[:,1,0,0], 5, stm.shape[0:1]) # variance on f0
        vae.fit([stims_train,stm,labels_train], epochs=num_epochs, batch_size=batch_size)
        vae.save_weights(model_path)
    else:
        print("*************************loading weights of the pre-trained model*************************")
        vae.load_weights(model_path)

    return(vae)
