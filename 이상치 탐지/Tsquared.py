from setting import bigyo_data
import pandas as pd
from t2 import T2
import numpy as np
from evaluating import evaluating_change_point, save_result
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import argparse
parser = argparse.ArgumentParser(description='T-Squared')
parser.add_argument('--num', help='몇번째 데이터 셋 비교 실험인지', type = int)
args = parser.parse_args()

t2 = T2(scaling=True, using_PCA=False)

train_set, test_set = bigyo_data(args.num)
total, labels = np.empty((0,5), float), ['normal','anomaly']

X_train = train_set.drop(['anomaly'], axis=1)
true_outlier = np.array([test_set.anomaly]).reshape(-1,)

for i in range(5):
    t2.fit(X_train)
    t2.predict(test_set.drop(['anomaly'], axis=1), window_size = 5, plot_fig=False)  # window_size가 아무짝에도 쓸모 없잖아;;;
    prediction = pd.Series((t2.T2>np.float64(0.05)).astype(int), index=test_set.index).fillna(0)   # np.float64로 0.05 Threshold UCL 값을 감싸줘야 돌아간다.
    total = np.append(total, np.array([[precision_score(true_outlier, prediction), recall_score(true_outlier, prediction), f1_score(true_outlier, prediction), accuracy_score(true_outlier, prediction), roc_auc_score(true_outlier, prediction)]]), axis = 0)
total = total.mean(axis = 0)   # 열끼리 평균
print(total)
save = pd.DataFrame({'Name':"Tsquared_{}".format(args.num), 'Precision':"{:.2f}".format(total[0]), 'Recall':"{:.2f}".format(total[1]), 'F1':"{:.2f}".format(total[2]), 'ACC':"{:.2f}".format(total[3]), 'AUC':"{:.2f}".format(total[4])}, index = [0])
save.to_csv('C:/Users/yoona/Desktop/아침/비교실험.csv', index = False, mode='a', header = False)