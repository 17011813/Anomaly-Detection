from setting import bigyo_data
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from evaluating import evaluating_change_point, save_result
import argparse
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
parser = argparse.ArgumentParser(description='LSTM-AE')
parser.add_argument('--num', help='몇번째 데이터 셋 비교 실험인지', type = int)
args = parser.parse_args()

def arch(data):
    EPOCHS = 100
    BATCH_SIZE = 32
    VAL_SPLIT = 0.1
   
    # model defining
    # define encoder
    inputs = keras.Input(shape=(data.shape[1], data.shape[2]))
    encoded = layers.LSTM(100, activation='relu')(inputs)

    # define reconstruct decoder
    decoded = layers.RepeatVector(data.shape[1])(encoded)
    decoded = layers.LSTM(100, activation='relu', return_sequences=True)(decoded)
    decoded = layers.TimeDistributed(layers.Dense(data.shape[2]))(decoded)

    # tie it together
    model = keras.Model(inputs, decoded)
    encoder = keras.Model(inputs, encoded)

    model.compile(optimizer='adam', loss='mae', metrics=["mse"])

    early_stopping = EarlyStopping(patience=5, verbose=0)
    
    history = model.fit(data, data,
                        validation_split=VAL_SPLIT,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        verbose=0,
                        shuffle=False,     # LSTM은 순서대로 들어가야 해서 섞으면 안되지
                        callbacks=[early_stopping]   )
    return history, model

N_STEPS = 5   # 13만 5로 하고 나머지는 10으로 돌렸음
Q = 0.05

# Generated training sequences for use in the model.
def create_sequences(values, time_steps=N_STEPS):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)

train_set, test_set = bigyo_data(args.num)
total, labels = np.empty((0,5), float), ['normal','anomaly']
X_train = train_set.drop(['anomaly'], axis=1)

# X_train 데이터만 가지고 스케일링
StSc = MinMaxScaler()
StSc.fit(X_train)

# convert into input/output
X_train = create_sequences(StSc.transform(X_train), N_STEPS)
true_outlier = np.array([test_set.anomaly]).reshape(-1,)

for i in range(5):
    history, model = arch(X_train)

    residuals = pd.Series(np.sum(np.mean(np.abs(X_train - model.predict(X_train)), axis=1), axis=1))
    UCL = 0.05

    # results predicting
    X_test = create_sequences(StSc.transform(test_set.drop(['anomaly'], axis=1)), N_STEPS)
    cnn_residuals = pd.Series(np.sum(np.mean(np.abs(X_test - model.predict(X_test)), axis=1), axis=1))

    # data i is an anomaly if samples [(i - timesteps + 1) to (i)] are anomalies
    anomalous_data = cnn_residuals > UCL
    anomalous_data_indices = []
    for data_idx in range(N_STEPS - 1, len(X_test) - N_STEPS + 1):
        if np.all(anomalous_data[data_idx - N_STEPS + 1 : data_idx]):
            anomalous_data_indices.append(data_idx)

    prediction = pd.Series(data=0, index=test_set.index)
    prediction.iloc[anomalous_data_indices] = 1
    prediction = np.array(prediction).reshape(-1,)

    total = np.append(total, np.array([[precision_score(true_outlier, prediction), recall_score(true_outlier, prediction), f1_score(true_outlier, prediction), accuracy_score(true_outlier, prediction), roc_auc_score(true_outlier, prediction)]]), axis = 0)
    print(confusion_matrix(true_outlier, prediction))
total = total.mean(axis = 0)   # 열끼리 평균
print(total)
save = pd.DataFrame({'Name':"LSTM-AE_{}".format(args.num), 'Precision':"{:.2f}".format(total[0]), 'Recall':"{:.2f}".format(total[1]), 'F1':"{:.2f}".format(total[2]), 'ACC':"{:.2f}".format(total[3]), 'AUC':"{:.2f}".format(total[4])}, index = [0])
save.to_csv('C:/Users/yoona/Desktop/아침/비교실험.csv', index = False, mode='a', header = False)