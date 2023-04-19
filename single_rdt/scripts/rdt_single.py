import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Flatten
import seaborn as sns
from scipy.stats import multivariate_normal, truncnorm
from itertools import product
import sys
import os


CORPUS = sys.argv[1]  # 'buc'
ID = sys.argv[2]  # 'weighted'
NUM = str(sys.argv[3])  # 11
WEIGHT = float(sys.argv[4])
root = '/fs/clip-realspeech/projects/speech_in_noise'

path_f = os.path.abspath(root+'/figs/'+CORPUS+'/w_'+str(WEIGHT))
if not os.path.exists(path_f):
    print(path_f)
    os.makedirs(path_f)

path_a = os.path.abspath(root+'/analysis/'+CORPUS+'/w_'+str(WEIGHT))
if not os.path.exists(path_a):
    os.makedirs(path_a)

path_m = os.path.abspath(root+'/models/'+CORPUS+'/w_'+str(WEIGHT))
if not os.path.exists(path_m):
    os.makedirs(path_m)

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_encoder(input_shape = [2,1], latent_dim = 500):
    encoder_inputs = keras.Input(shape=(input_shape[0], input_shape[1], 1))
    x = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')(encoder_inputs)
    x = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(2000, activation='relu')(x)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

    return(encoder)

def build_decoder(latent_dim = 500, decision_dim = 1, recall_dim = 2):
    latent_inputs = keras.Input(shape=(latent_dim))
    recall = tf.keras.layers.Dense(recall_dim,activation=None)(latent_inputs)
    recall = tf.expand_dims(recall, -1)
    recall = tf.expand_dims(recall, -1)
    decoder = keras.Model(latent_inputs, recall, name='decoder')

    return(decoder)

def build_dec_model(input_dim=500, model_dim=100, dec_dim=1):
    latent_inputs = keras.Input(shape=(input_dim))
    decision_hidden = tf.keras.layers.Dense(model_dim,activation='relu',name='dec/decision_hidden')(latent_inputs)
    decision = tf.keras.layers.Dense(dec_dim,activation=None,name='dec/decision')(decision_hidden)
    decision_sig = tf.sigmoid(decision)
    decision = keras.Model(latent_inputs, decision_sig, name='dec')
    return(decision)

class VAE(keras.Model):
    def __init__(self, encoder, decoder, dec, beta, cue_weights, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.dec = dec
        self.beta = beta
        self.cue_weights = cue_weights
        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self.recall_loss_tracker = keras.metrics.Mean(
            name='recall_loss'
        )
        self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')
        self.decision_tracker = keras.metrics.Mean(name='dec_loss')


    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.recall_loss_tracker,
            self.kl_loss_tracker,
            self.decision_tracker,
        ]

    def train_step(self, all_data):
        data_true = all_data[0][0]
        data_fake = all_data[0][1]
        cat_labels = all_data[0][2]
        with tf.GradientTape(persistent=True) as tape:
            z_mean, z_log_var, z = self.encoder(data_fake)
            recall = self.decoder(z)
            mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
            recall_loss = tf.reduce_mean(
                                  tf.reduce_sum(
                                      tf.math.multiply(self.cue_weights,mse(data_true,recall)), axis=(1, 2)
                                  )
                              )


            kl_loss =  -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.math.multiply(np.array([self.beta],dtype='float32'),tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1)))
            
            bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            
            decision = self.dec(z)
            decision_loss = bce(cat_labels,decision)

            total_loss = recall_loss + kl_loss + decision_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.recall_loss_tracker.update_state(recall_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.decision_tracker.update_state(decision_loss)
        return {
            'loss': self.total_loss_tracker.result(),
            'recall_loss': self.recall_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result(),
            'dec_loss': self.decision_tracker.result(),
        }



def make_noisy_stims(start_v=1,stop_v=4,start_f=1,stop_f=4,n_ex=100,var=1):
    VOTs = np.linspace(start=-3,stop=3, NUM=9)
    f0s = np.linspace(start=-3,stop=3, NUM=9)
    stims = []
    for VOT in VOTs[start_v:stop_v]:
        noisy_VOTs = np.random.default_rng().normal(VOT, var, n_ex)
        for f0 in f0s[start_f:stop_f]:
            noisy_f0s = np.random.default_rng().normal(f0, var, n_ex)
            for n_VOT in noisy_VOTs:
                for n_f0 in noisy_f0s:
                    stims.append([n_VOT,n_f0])
    return(np.array(stims))

def pb_stims(N=100,canonical=True):
    if canonical==True:
        b_canon=make_noisy_stims(start_v=1,stop_v=4,start_f=1,stop_f=4,n_ex=N,var=2)
        p_canon=make_noisy_stims(start_v=-4,stop_v=-1,start_f=-4,stop_f=-1,n_ex=N,var=2)
        return(b_canon,p_canon)
    else:
        b_reverse=make_noisy_stims(start_v=1,stop_v=4,start_f=-5,stop_f=-2,n_ex=N,var=2)
        p_reverse=make_noisy_stims(start_v=-5,stop_v=-2,start_f=1,stop_f=4,n_ex=N,var=2)
        return(b_reverse,p_reverse)
    

def make_VAE_dataset(canonical=True, N=100):
    np.random.seed(987654321)
    b_list,p_list = pb_stims(canonical=canonical,N=N)
    # b_labels = np.column_stack((np.ones(b_list.shape[0]), np.zeros(b_list.shape[0])))
    # p_labels = np.column_stack((np.zeros(p_list.shape[0]), np.ones(p_list.shape[0])))
    b_labels = np.zeros(b_list.shape[0])
    p_labels = np.ones(p_list.shape[0])
    stims = np.concatenate((b_list,p_list))
    labels = np.concatenate((b_labels,p_labels))
    # create a dataset for VAE training
     # shuffle all stims and according labels, then expand their dims for model training
    stims_shuff,labels_shuff = shuffle(stims,labels,random_state=4)
    stims_shuff = np.expand_dims(stims_shuff, -1).astype('float32')
    stims_shuff = np.expand_dims(stims_shuff, -1).astype('float32')
    labels_shuff = np.expand_dims(labels_shuff, -1).astype('float32')
    return(stims_shuff,labels_shuff)


def test_stims(LBL=0,vot=1,f0=3,N=100,var=1):
    # if LBL==1:
    #     LBL_1 = 0
    # else:
    #     LBL_1 = 1
    stims = make_noisy_stims(start_v=vot,stop_v=int(vot+1),start_f=f0,stop_f=int(f0+1),n_ex=N,var=var)
    # labels = np.column_stack((np.ones(len(stims))*LBL,np.ones(len(stims))*LBL_1))
    labels = np.ones(len(stims))*LBL
    stims_shuff,labels_shuff = shuffle(stims,labels,random_state=4)
    stims_shuff = np.expand_dims(stims_shuff, -1).astype('float32')
    labels_shuff = np.expand_dims(labels_shuff, -1).astype('float32')
    return(stims_shuff,labels_shuff)



def train_and_checkpoint(vae, manager, NUM_restore=10):
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print('Restored from {}'.format(manager.latest_checkpoint))
    else:
        print('Initializing from scratch.')

    for NUM in range(NUM_restore):
        if int(NUM) % 10 == 0:
            save_path = manager.save()
            print('Saved checkpoint for step {}: {}'.format(int(ckpt.step), save_path))
            stims_train, labels_train = make_VAE_dataset(canonical=True, N=100)
        vae.fit(stims_train, epochs=1, batch_size=32)
    return(vae)


opt = tf.keras.optimizers.Adagrad(0.001,clipnorm=50.)
beta=0.0025
# cue_weights = np.array([0.999,0.001]) # DUR emphasized
if WEIGHT != 1:
    cue_weights = np.array([np.abs(1-WEIGHT),WEIGHT]) # f1 emphasized
else:
    cue_weights = np.array([1,1]) # NEUTRAL

cue_weights = np.expand_dims(cue_weights, -1).astype('float32')
encoder = build_encoder()
decoder = build_decoder()
dec_model = build_dec_model()
vae = VAE(encoder, decoder, dec_model, beta, cue_weights)
vae.compile(optimizer=opt)

# stims_train, labels_train = make_VAE_dataset(canonical=True, N=100)

#data_csv = pd.read_csv(os.path.abspath(root+'/data/'+CORPUS+'_vowels_norm.csv'))
data_csv = pd.read_csv(os.path.abspath(root+'/data/'+CORPUS+'_scaled.csv'))
data_csv['label'] = 0
data_csv.loc[(data_csv.phone == 'AE'), 'label'] = 1

data_csv = data_csv[(data_csv.f1<=1200.0) & (data_csv.duration>=0.03)]
labels_train = np.array(data_csv.label)

labels_train = np.expand_dims(labels_train, -1).astype('float32')
stims_train = np.array(data_csv[['norm_duration','norm_f1']])
stims_train = np.expand_dims(stims_train, -1).astype('float32')
stims_train = np.expand_dims(stims_train, -1).astype('float32')
stm = stims_train.copy()


vae.fit([stims_train,stm,labels_train], epochs=10, batch_size=32)
vae.save_weights(os.path.abspath(path_m+'/'+ID+'_single_vae_'+NUM+'/ckpt'))

# vae.load_weights('./models/simp_single_vae_'+CORPUS+'/ckpt')

durs = np.arange(-10,10.1,0.1)
f1s = np.arange(-10,10.1,0.1)
all_points = np.array(list(product(durs,f1s)))
all_points = np.expand_dims(all_points, -1).astype('float32')
all_points = np.expand_dims(all_points, -1).astype('float32')

mu, sig, z = vae.encoder.predict(all_points)
preds = vae.decoder(z)

plt.figure(figsize=(8, 7),dpi=100)
plt.scatter(x=stm[:,0], y=stm[:,1], c=labels_train, alpha=0.5)
plt.xlabel('Duration (z-score)', fontsize=20)
plt.ylabel('F1 (z-score)', fontsize=20)
plt.legend(labels=['eh','ae'])
plt.savefig(os.path.abspath(path_f+'/'+ID+'_'+NUM+'_train.jpg'))


plt.figure(figsize=(8, 7),dpi=100)
plt.scatter(x=all_points[:,0], y=all_points[:,1], c='silver', alpha=0.5)
plt.scatter(x=preds[:,0], y=preds[:,1], c='rosybrown', alpha=0.5)
plt.xlabel('duration (z-score)', fontsize=20)
plt.ylabel('f1 (z-score)', fontsize=20)
plt.legend(labels=['Training data','Reconstruction'])
plt.savefig(os.path.abspath(path_f+'/'+ID+'_'+NUM+'_recon.jpg'))

decision=vae.dec(z)

dec_lbl = decision.numpy()
dec_lbl[dec_lbl<0.5] = 0
dec_lbl[dec_lbl>=0.5] = 1

plt.figure(figsize=(8, 7),dpi=100)
plt.scatter(x=all_points[:,0], y=all_points[:,1], c=dec_lbl, alpha=0.5)
plt.xlabel('duration (z-score)', fontsize=20)
plt.ylabel('f1 (z-score)', fontsize=20)
plt.legend(labels=['b','p'])
# plt.xticks(ticks=np.arange(-30,31,10),labels=[5*x for x in range(1,8)])
# plt.yticks(ticks=np.arange(-30,31,10),labels=[20*x+200 for x in range(0,7)])
plt.savefig(path_f+'/'+ID+'_'+NUM+'_preds.jpg')


res = {'dur':[],'f1':[],'dec':[],'subject':[]}
durs = np.arange(-10,10.1,0.1)
f1s = np.arange(-10,10.1,0.1)
all_points = np.array(list(product(durs,f1s)))
all_points_ = np.expand_dims(all_points, -1).astype('float32')
all_points_ = np.expand_dims(all_points_, -1).astype('float32')
mu, sig, z = vae.encoder.predict(all_points_)
decision=vae.dec(z)

dur_col = all_points[:,0]
f1_col = all_points[:,1]
dec = decision.numpy()

res['dur'] = all_points[:,0]
res['f1'] = all_points[:,1]
res['dec'] = decision.numpy()[:,0]
res['subject'] = 0



metadata = pd.DataFrame(res)
metadata.to_csv(os.path.abspath(path_a+'/'+ID+'_'+NUM+'_res.csv'))



