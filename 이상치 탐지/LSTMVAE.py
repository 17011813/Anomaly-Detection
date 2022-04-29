from setting import bigyo_data
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector
from tensorflow.keras.layers import Flatten, Dense, Dropout, Lambda
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras import losses
from tensorflow.python.framework.ops import disable_eager_execution
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd
from evaluating import evaluating_change_point, save_result
import argparse
disable_eager_execution()
parser = argparse.ArgumentParser(description='LSTM-VAE')
parser.add_argument('--num', help='몇번째 데이터 셋 비교 실험인지', type = int)
args = parser.parse_args()

def create_lstm_vae(input_dim, 
    timesteps, 
    batch_size, 
    intermediate_dim, 
    latent_dim,
    epsilon_std):

    """
    Creates an LSTM Variational Autoencoder (VAE). Returns VAE, Encoder, Generator. 
    # Arguments
        input_dim: int.
        timesteps: int, input timestep dimension.
        batch_size: int.
        intermediate_dim: int, output shape of LSTM. 
        latent_dim: int, latent z-layer shape. 
        epsilon_std: float, z-layer sigma.
    # References
        - [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
        - [Generating sentences from a continuous space](https://arxiv.org/abs/1511.06349)
    """
    x = Input(shape=(timesteps, input_dim,))

    # LSTM encoding
    h = LSTM(intermediate_dim)(x)

    # VAE Z layer
    z_mean = Dense(latent_dim)(h)
    z_log_sigma = Dense(latent_dim)(h)
    
    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim),
                                  mean=0., stddev=epsilon_std)
        return z_mean + z_log_sigma * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])
    
    # decoded LSTM layer
    decoder_h = LSTM(intermediate_dim, return_sequences=True)
    decoder_mean = LSTM(input_dim, return_sequences=True)

    h_decoded = RepeatVector(timesteps)(z)
    h_decoded = decoder_h(h_decoded)

    # decoded layer
    x_decoded_mean = decoder_mean(h_decoded)
    
    # end-to-end autoencoder
    vae = Model(x, x_decoded_mean)

    # encoder, from inputs to latent space
    encoder = Model(x, z_mean)

    # generator, from latent space to reconstructed inputs
    decoder_input = Input(shape=(latent_dim,))

    _h_decoded = RepeatVector(timesteps)(decoder_input)
    _h_decoded = decoder_h(_h_decoded)

    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)
    
    def vae_loss(x, x_decoded_mean):
        mse = losses.MeanSquaredError()
        xent_loss = mse(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
        loss = xent_loss + kl_loss
        return loss

    vae.compile(optimizer='rmsprop', loss=vae_loss)
    
    return vae, encoder, generator

def arch(data):

    input_dim = data.shape[-1] # 13
    timesteps = data.shape[1] # 8
    BATCH_SIZE = 1
    
    model, enc, gen = create_lstm_vae(input_dim, 
        timesteps=timesteps, 
        batch_size=BATCH_SIZE, 
        intermediate_dim=32,
        latent_dim=16,
        epsilon_std=1.)

    history = model.fit(
        data,
        data,
        epochs=20,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=0)
        ],
    )
    return history, model

def create_sequences(values, time_steps):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)

N_STEPS = 1
Q = 0.05

train_set, test_set = bigyo_data(args.num)
total, labels = np.empty((0,5), float), ['normal','anomaly']

X_train = train_set.drop(['anomaly'], axis=1)

StSc = MinMaxScaler()
StSc.fit(X_train)

X_train = create_sequences(StSc.transform(X_train), N_STEPS)
true_outlier = np.array([test_set.anomaly]).reshape(-1,)

for i in range(5):
    history, model = arch(X_train)

    residuals = pd.Series(np.sum(np.mean(np.abs(X_train - model.predict(X_train)), axis=1), axis=1))
    UCL = residuals.quantile(Q)

    # results predicting
    X_test = create_sequences(StSc.transform(test_set.drop(['anomaly'], axis=1)), N_STEPS)
    cnn_residuals = pd.Series(np.sum(np.mean(np.abs(X_test - model.predict(X_test)), axis=1), axis=1))

    # data i is an anomaly if samples [(i - timesteps + 1) to (i)] are anomalies
    anomalous_data = cnn_residuals > (3/2 * UCL)
    anomalous_data_indices = []
    for data_idx in range(N_STEPS - 1, len(X_test) - N_STEPS + 1):
        if np.all(anomalous_data[data_idx - N_STEPS + 1 : data_idx]):
            anomalous_data_indices.append(data_idx)

    prediction = pd.Series(data=0, index=test_set.index)
    prediction.iloc[anomalous_data_indices] = 1
    prediction = np.array(prediction).reshape(-1,)
    
    total = np.append(total, np.array([[precision_score(true_outlier, prediction), recall_score(true_outlier, prediction), f1_score(true_outlier, prediction), accuracy_score(true_outlier, prediction), roc_auc_score(true_outlier, prediction)]]), axis = 0)
    
total = total.mean(axis = 0)   # 열끼리 평균

save = pd.DataFrame({'Name':"LSTM-VAE_{}".format(args.num), 'Precision':"{:.2f}".format(total[0]), 'Recall':"{:.2f}".format(total[1]), 'F1':"{:.2f}".format(total[2]), 'ACC':"{:.2f}".format(total[3]), 'AUC':"{:.2f}".format(total[4])}, index = [0])
save.to_csv('C:/Users/yoona/Desktop/아침/비교실험.csv', index = False, mode='a', header = False)
print(save)