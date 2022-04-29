from setting import bigyo_data
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from evaluating import evaluating_change_point, save_result
import numpy as np
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, Dropout, TimeDistributed
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import argparse
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
parser = argparse.ArgumentParser(description='AE')
parser.add_argument('--num', help='몇번째 데이터 셋 비교 실험인지', type = int)
args = parser.parse_args()

def arch(param, data):
    input_dots = Input((8,))

    x = Dense(param[0])(input_dots)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Dense(param[1])(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    bottleneck = Dense(param[2], activation='linear')(x)

    x = Dense(param[1])(bottleneck)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Dense(param[0])(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    out = Dense(8, activation='linear')(x)

    model = Model(input_dots, out)
    model.compile(optimizer=Adam(param[3]), loss='mae', metrics=["mse"])
    early_stopping = EarlyStopping(patience=3, verbose=0)
    model.fit(data, data,
                validation_split=0.2,
                epochs=40,
                batch_size=param[4],
                verbose=0,
                shuffle=True,
                callbacks=[early_stopping]
               )
    return model

BEST_PARAMS = [5, 4, 2, 0.005, 32]
Q = 0.05 # quantile for upper control limit (UCL) selection

train_set, test_set = bigyo_data(args.num)
total, labels = np.empty((0,5), float), ['normal','anomaly']
# 추론
X_train = train_set.drop(['anomaly'], axis=1)
# X_train 데이터만 가지고 스케일링
StSc = MinMaxScaler()
StSc.fit(X_train)
true_outlier = np.array([test_set.anomaly]).reshape(-1,)
for i in range(5):
    # model defining and fitting
    model = arch(BEST_PARAMS, StSc.transform(X_train))

    # results predicting
    residuals = pd.DataFrame(StSc.transform(X_train) - model.predict(StSc.transform(X_train))).abs().sum(axis=1)
    UCL = residuals.quantile(Q)

    df_sc = StSc.transform(test_set.drop(['anomaly'], axis=1))
    ae_residuals = df_sc - model.predict(df_sc)
    ae = pd.DataFrame(ae_residuals).abs().sum(axis=1)

    prediction = np.array(pd.Series((ae > 3/2*UCL).astype(int).values, index=test_set.index).fillna(0)).reshape(-1,)
    total = np.append(total, np.array([[precision_score(true_outlier, prediction), recall_score(true_outlier, prediction), f1_score(true_outlier, prediction), accuracy_score(true_outlier, prediction), roc_auc_score(true_outlier, prediction)]]), axis = 0)
print(confusion_matrix(true_outlier, prediction))
total = total.mean(axis = 0)   # 열끼리 평균

save = pd.DataFrame({'Name':"AE_{}".format(args.num), 'Precision':"{:.2f}".format(total[0]), 'Recall':"{:.2f}".format(total[1]), 'F1':"{:.2f}".format(total[2]), 'ACC':"{:.2f}".format(total[3]), 'AUC':"{:.2f}".format(total[4])}, index = [0])
save.to_csv('C:/Users/yoona/Desktop/아침/비교실험.csv', index = False, mode='a', header = False)