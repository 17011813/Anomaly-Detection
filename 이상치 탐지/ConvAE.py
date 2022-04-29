from setting import bigyo_data
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, f1_score
import sys, random, os
from evaluating import evaluating_change_point, save_result
import argparse
parser = argparse.ArgumentParser(description='Conv-AE')
parser.add_argument('--num', help='몇번째 데이터 셋 비교 실험인지', type = int)
args = parser.parse_args()

def arch(data):
    model = keras.Sequential(
        [
            layers.Input(shape=(data.shape[1], data.shape[2])),
            layers.Conv1D(
                filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1D(
                filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(
                filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1DTranspose(
                filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    # model.summary()

    history = model.fit(
        data,
        data,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=0)],
    )
    return history, model

# hyperparameters 선택
N_STEPS = 4
Q = 0.99 # quantile for upper control limit (UCL) selection

# Generated training sequences for use in the model.
def create_sequences(values, time_steps=N_STEPS):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)

train_set, test_set = bigyo_data(args.num)
total, labels = np.empty((0,3), float), ['normal','anomaly']

X_train = train_set.drop(['anomaly'], axis=1)
true_outlier = np.array([test_set.anomaly]).reshape(-1,)

# X_train 데이터만 가지고 스케일링
StSc = MinMaxScaler()
StSc.fit(X_train)

# convert into input/output
X_train = create_sequences(StSc.transform(X_train), N_STEPS)

for i in range(5):
    # 모델 정의 및 훈련
    print(X_train.shape)
    history, model = arch(X_train)

    # 결과 예측             앞의 100까지 잘라서 train으로 fit 하고
    residuals = pd.Series(np.sum(np.mean(np.abs(X_train - model.predict(X_train)), axis=1), axis=1))
    UCL = residuals.quantile(Q)

    # 전체 데이터 셋 중 5천개만 떼어다가 예측
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

    total = np.append(total, np.array([[precision_score(true_outlier, prediction), recall_score(true_outlier, prediction), f1_score(true_outlier, prediction)]]), axis = 0)
total = total.mean(axis = 0)   # 열끼리 평균

save = pd.DataFrame({'Name':"Conv-AE_{}".format(args.num), 'Precision':"{:.2f}".format(total[0]), 'Recall':"{:.2f}".format(total[1]), 'F1':"{:.2f}".format(total[2])}, index = [0])
save.to_csv('C:/Users/yoona/Desktop/아침/비교실험.csv', index = False, mode='a', header = False)