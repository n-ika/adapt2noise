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

class Sampling(layers.Layer):
    def call(self, inputs):
        mu, z_log_var = inputs
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return mu + tf.exp(0.5 * z_log_var) * epsilon

def build_encoder(input_shape = (64,64,1), latent_dim = 500):
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
    latent_inputs = keras.Input(shape=(input_dim))
    decision_hidden = tf.keras.layers.Dense(model_dim,activation="relu",name="dec/decision_hidden")(latent_inputs)
    decision = tf.keras.layers.Dense(dec_dim,activation=None,name="dec/decision")(decision_hidden)
    decision_sig = tf.sigmoid(decision)
    decision = keras.Model(latent_inputs, decision_sig, name="dec")
    return(decision)

class VAE(keras.Model):
    def __init__(self, encoder1, encoder2, decoder, dec, cue_weights1, cue_weights2, beta=0.0025, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.decoder = decoder
        self.dec = dec
        self.beta = beta
        self.cue_weights1 = cue_weights1
        self.cue_weights2 = cue_weights2
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.recall_loss_tracker1 = keras.metrics.Mean(
            name="recall_loss1"
        )
        self.recall_loss_tracker2 = keras.metrics.Mean(
            name="recall_loss2"
        )
        self.kl_loss_tracker1 = keras.metrics.Mean(name="kl_loss1")
        self.kl_loss_tracker2 = keras.metrics.Mean(name="kl_loss2")
        self.decision_tracker = keras.metrics.Mean(name="dec_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.recall_loss_tracker1,
            self.recall_loss_tracker2,
            self.kl_loss_tracker1,
            self.kl_loss_tracker2,
            self.decision_tracker,
        ]

    def train_step(self, all_data):
        data_true = all_data[0][0]
        data_fake1 = all_data[0][1]
        data_fake2 = all_data[0][2]
        cat_labels = all_data[0][3]
        with tf.GradientTape(persistent=True) as tape:
            mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
            bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

            mu1, z_log_var1, z1 = self.encoder1(data_fake1)
            recall1 = self.decoder(z1)
            recall_loss1 = tf.reduce_mean(
                                  tf.reduce_sum(
                                      tf.math.multiply(self.cue_weights1,mse(data_true,recall1)), axis=(1, 2)
                                  )
                              )

            kl_loss1 =  -0.5 * (1 + z_log_var1 - tf.square(mu1) - tf.exp(z_log_var1))
            kl_loss1 = tf.math.multiply(np.array([self.beta],dtype='float32'),tf.reduce_mean(tf.reduce_sum(kl_loss1, axis=1)))


            mu2, z_log_var2, z2 = self.encoder2(data_fake2)
            recall2 = self.decoder(z2)
            recall_loss2 = tf.reduce_mean(
                                  tf.reduce_sum(
                                      tf.math.multiply(self.cue_weights2,mse(data_true,recall2)), axis=(1, 2)
                                  )
                              )

            kl_loss2 =  -0.5 * (1 + z_log_var2 - tf.square(mu2) - tf.exp(z_log_var2))
            kl_loss2 = tf.math.multiply(np.array([self.beta],dtype='float32'),tf.reduce_mean(tf.reduce_sum(kl_loss2, axis=1)))
            
            decision = self.dec(tf.math.multiply(z1,z2))
            decision_loss = bce(cat_labels,decision)

            total_loss = tf.math.multiply(recall_loss1,recall_loss2) + (0.5 * (kl_loss1) + 0.5 * (kl_loss2)) + decision_loss

        grads = tape.gradient([total_loss], self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.recall_loss_tracker1.update_state(recall_loss1)
        self.recall_loss_tracker2.update_state(recall_loss2)
        self.kl_loss_tracker1.update_state(kl_loss1)
        self.kl_loss_tracker2.update_state(kl_loss2)
        self.decision_tracker.update_state(decision_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "recall_loss1": self.recall_loss_tracker1.result(),
            "recall_loss2": self.recall_loss_tracker2.result(),
            "kl_loss1": self.kl_loss_tracker1.result(),
            "kl_loss2": self.kl_loss_tracker2.result(),
            "dec_loss": self.decision_tracker.result(),
        }

def train_model(path,train,num_epochs,batch_size,variance,weight,beta,lr,latent,dec_dim,dec_size):
    stims_train, labels_train = make_VAE_dataset(canonical=True, N=100)
    opt = tf.keras.optimizers.Adagrad(lr,clipnorm=50.)
    cue_weights1 = np.array([weight, float(1-weight)]) # VOT emphasized
    cue_weights1 = np.expand_dims(cue_weights1, -1).astype("float32")
    cue_weights2 = np.array([float(1-weight), weight]) # f0 emphasized
    cue_weights2 = np.expand_dims(cue_weights2, -1).astype("float32")
    encoder1 = build_encoder(latent_dim=latent,input_shape=stims_train.shape[1:]) # for VOT emphasized
    encoder2 = build_encoder(latent_dim=latent,input_shape=stims_train.shape[1:]) # for f0 emphasized
    decoder = build_decoder(latent_dim=latent,recall_dim=stims_train.shape[1])
    dec_model = build_dec_model(input_dim=latent,model_dim=dec_size, dec_dim=dec_dim)
    vae = VAE(encoder1, encoder2, decoder, dec_model, cue_weights1, cue_weights2, beta)
    vae.compile(optimizer=opt)
    if train==True:
        stm1 = stims_train.copy()
        stm2 = stims_train.copy()
        if variance==True:
            stm1[:,1,0,0] = np.random.default_rng().normal(stm1[:,1,0,0], 5, stm1.shape[0:1]) # variance on f0
            stm2[:,0,0,0] = np.random.default_rng().normal(stm2[:,0,0,0], 5, stm2.shape[0:1]) # variance on VOT
        vae.fit([stims_train,stm1,stm2,labels_train], epochs=num_epochs, batch_size=batch_size)
        vae.save_weights(path)
    else:
        vae.load_weights(path)

    return(vae)
