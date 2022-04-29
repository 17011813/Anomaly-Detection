from setting import bigyo_data
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from evaluating import evaluating_change_point, save_result
import argparse
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
parser = argparse.ArgumentParser(description='LSTM')
parser.add_argument('--num', help='몇번째 데이터 셋 비교 실험인지', type = int)
args = parser.parse_args()

def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]   # 과거부터 어제까지를 X로 오늘 값을 y로
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

N_STEPS = 5
EPOCHS = 25
BATCH_SIZE = 32
VAL_SPLIT = 0.2

train_set, test_set = bigyo_data(args.num) 

X_train = train_set.drop(['anomaly'], axis=1)
total, labels = np.empty((0,5), float), ['normal','anomaly']
# X_train 데이터만 가지고 스케일링
StSc = MinMaxScaler()
StSc.fit(X_train)

# convert into input/output
X, y = split_sequences(StSc.transform(X_train), N_STEPS)

# the dataset knows the number of features
n_features = X.shape[2]
true_outlier = np.array([test_set.anomaly]).reshape(-1,)

for i in range(1):    # 여기서 5번 돌릴수가 없다 -- 모델 초기화 해야해서... 그냥 배치 파일에서 다섯번 돌려주는 수 밖에 없음
    # model defining
    model = Sequential()
    model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(N_STEPS, n_features)))
    model.add(LSTM(100, activation='relu'))
    model.add(Dense(n_features))
    model.compile(optimizer='adam', loss='mae', metrics=["mse"])

    history = model.fit(X, y,
                        validation_split=VAL_SPLIT,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        verbose=0,
                        shuffle=False )

    # results predicting
    residuals = pd.DataFrame(y - model.predict(X)).abs().sum(axis=1)
    UCL = residuals.quantile(0.05)

    # results predicting
    X, y = split_sequences(StSc.transform(test_set.drop(['anomaly'], axis=1)), N_STEPS)

    anomalous_data = residuals > (3/2 * UCL)
    anomalous_data_indices = []
    for data_idx in range(N_STEPS - 1, len(X) - N_STEPS + 1):
        if np.all(anomalous_data[data_idx - N_STEPS + 1 : data_idx]):
            anomalous_data_indices.append(data_idx)

    prediction = pd.Series(data=0, index=test_set.index)
    prediction.iloc[anomalous_data_indices] = 1
    prediction = np.array(prediction).reshape(-1,)
    
    total = np.append(total, np.array([[precision_score(true_outlier, prediction), recall_score(true_outlier, prediction), f1_score(true_outlier, prediction), accuracy_score(true_outlier, prediction), roc_auc_score(true_outlier, prediction)]]), axis = 0)
    print(total)
total = total.mean(axis = 0)   # 열끼리 평균

save = pd.DataFrame({'Name':"LSTM_{}".format(args.num), 'Precision':"{:.2f}".format(total[0]), 'Recall':"{:.2f}".format(total[1]), 'F1':"{:.2f}".format(total[2]), 'ACC':"{:.2f}".format(total[3]), 'AUC':"{:.2f}".format(total[4])}, index = [0])
save.to_csv('C:/Users/yoona/Desktop/아침/비교실험.csv', index = False, mode='a', header = False)