from setting import bigyo_data
import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np
import shap
from evaluating import evaluating_change_point, save_result
import argparse
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
parser = argparse.ArgumentParser(description='Isolation Forest')
parser.add_argument('--num', help='몇번째 데이터 셋 비교 실험인지', type = int)
args = parser.parse_args()

clf = IsolationForest(n_jobs=-1, contamination=0.01)   # 6일때는 0.1

train_set, test_set = bigyo_data(args.num)
total, labels = np.empty((0,5), float), ['normal','anomaly']

X_train = train_set.drop(['anomaly'], axis=1)
true_outlier = np.array([test_set.anomaly]).reshape(-1,)
for i in range(5):
    clf.fit(X_train)

    prediction = np.array(pd.Series(clf.predict(test_set.drop(['anomaly'], axis=1))*-1, index=test_set.index).rolling(3).median().fillna(0).replace(-1,0)).reshape(-1,)
    total = np.append(total, np.array([[precision_score(true_outlier, prediction), recall_score(true_outlier, prediction), f1_score(true_outlier, prediction), accuracy_score(true_outlier, prediction), roc_auc_score(true_outlier, prediction)]]), axis = 0)
    #print(classification_report(true_outlier, prediction))
total = total.mean(axis = 0)   # 열끼리 평균
print(total)
save = pd.DataFrame({'Name':"IF_{}".format(args.num), 'Precision':"{:.2f}".format(total[0]), 'Recall':"{:.2f}".format(total[1]), 'F1':"{:.2f}".format(total[2]), 'ACC':"{:.2f}".format(total[3]), 'AUC':"{:.2f}".format(total[4])}, index = [0])
save.to_csv('C:/Users/yoona/Desktop/아침/비교실험.csv', index = False, mode='a', header = False)