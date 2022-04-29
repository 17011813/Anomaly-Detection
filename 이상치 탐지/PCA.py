from setting import bigyo_data
import pandas as pd
import numpy as np
from t2 import T2
from evaluating import evaluating_change_point, save_result
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import argparse
parser = argparse.ArgumentParser(description='PCA')
parser.add_argument('--num', help='몇번째 데이터 셋 비교 실험인지', type = int)
args = parser.parse_args()

t2 = T2(scaling=True, using_PCA=True)

train_set, test_set = bigyo_data(args.num) 
total, labels = np.empty((0,5), float), ['normal','anomaly']

X_train = train_set.drop(['anomaly'], axis=1)
true_outlier = np.array([test_set.anomaly]).reshape(-1,)

for i in range(5):
    t2.fit(X_train)
    t2.predict(test_set.drop(['anomaly'], axis=1), window_size = 5, plot_fig=False)
    prediction = np.array(pd.Series(((t2.T2>0.05) | (t2.Q>0.05)).astype(int)[0], index=test_set.index).fillna(0)).reshape(-1,)
    total = np.append(total, np.array([[precision_score(true_outlier, prediction), recall_score(true_outlier, prediction), f1_score(true_outlier, prediction), accuracy_score(true_outlier, prediction), roc_auc_score(true_outlier, prediction)]]), axis = 0)
    print(confusion_matrix(true_outlier, prediction))
total = total.mean(axis = 0)
print(total)
save = pd.DataFrame({'Name':"PCA_{}".format(args.num), 'Precision':"{:.2f}".format(total[0]), 'Recall':"{:.2f}".format(total[1]), 'F1':"{:.2f}".format(total[2]), 'ACC':"{:.2f}".format(total[3]), 'AUC':"{:.2f}".format(total[4])}, index = [0])
save.to_csv('C:/Users/yoona/Desktop/아침/비교실험.csv', index = False, mode='a', header = False)