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
    def __init__(self, encoder1, encoder2, decoder, dec, cue_weights1, cue_weights2, beta=0.0025, **kwargs):
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
        self.decision_tracker1 = keras.metrics.Mean(name="dec_loss1")
        # self.decision_tracker2 = keras.metrics.Mean(name="dec_loss2")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.recall_loss_tracker1,
            self.recall_loss_tracker2,
            self.kl_loss_tracker1,
            self.kl_loss_tracker2,
            self.decision_tracker1,
            # self.decision_tracker2,
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
            
            p_j = np.random.rand(1) # sample a number in [0,1)]
            decision1 = self.dec(tf.math.multiply(z1,p_j)+tf.math.multiply(z2,(1-p_j)))
            decision_loss1 = bce(cat_labels,decision1)
            # decision2 = self.dec(z2)
            # decision_loss2 = bce(cat_labels,decision2)

            # total_loss = tf.math.multiply(recall_loss1,recall_loss2) + tf.math.multiply(0.5, (kl_loss1 + kl_loss2)) + decision_loss1
            total_loss = tf.math.multiply(0.5, (kl_loss1 + kl_loss2 + recall_loss1 + recall_loss2)) + decision_loss1

# average all losses
        grads = tape.gradient([total_loss], self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.recall_loss_tracker1.update_state(recall_loss1)
        self.recall_loss_tracker2.update_state(recall_loss2)
        self.kl_loss_tracker1.update_state(kl_loss1)
        self.kl_loss_tracker2.update_state(kl_loss2)
        self.decision_tracker1.update_state(decision_loss1)
        # self.decision_tracker2.update_state(decision_loss2)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "recall_loss1": self.recall_loss_tracker1.result(),
            "recall_loss2": self.recall_loss_tracker2.result(),
            "kl_loss1": self.kl_loss_tracker1.result(),
            "kl_loss2": self.kl_loss_tracker2.result(),
            "dec_loss1": self.decision_tracker1.result(),
            # "dec_loss2": self.decision_tracker2.result()
        }

def train_model(path,train,num_epochs,batch_size,variance,weight,beta,lr,latent,dec_dim,dec_size,corpus=None):
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
    variance : str ('y','n')
        adding variance on one of the dimensions of the input data - if true ('y'),
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

    if corpus != None:
        data_csv = pd.read_csv('./'+corpus+'_vowels_norm.csv')
        data_csv['label'] = 0
        data_csv.loc[(data_csv.phone == 'AE'), 'label'] = 1
        labels_train = np.array(data_csv.label)
        data_csv = pd.concat([data_csv,data_csv])
        labels_train = np.append(labels_train,labels_train)
        labels_train = np.expand_dims(labels_train, -1).astype("float32")
        stims_train = np.array(data_csv[['norm_f1','norm_duration']])
        stims_train = np.expand_dims(stims_train, -1).astype("float32")
        stims_train = np.expand_dims(stims_train, -1).astype("float32")
    else:
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
    if train=='y':
        print("*"*15 + "training the model from scratch"+"*"*15)
        stm1 = stims_train.copy()
        stm2 = stims_train.copy()
        if variance=='y':
            print("*"*15 + "using data with added variance"+"*"*15)
            stm1[:,1,0,0] = np.random.default_rng().normal(stm1[:,1,0,0], 5, stm1.shape[0:1]) # variance on f0
            stm2[:,0,0,0] = np.random.default_rng().normal(stm2[:,0,0,0], 5, stm2.shape[0:1]) # variance on VOT
        vae.fit([stims_train,stm1,stm2,labels_train], epochs=num_epochs, batch_size=batch_size)
        vae.save_weights(model_path)
    else:
        print("*"*15 + "loading weights of the pre-trained model"+"*"*15)
        vae.load_weights(model_path)

    return(vae)

# train_model('./','y',10,32,'n',0.999,0.0025,0.001,500,1,100)
