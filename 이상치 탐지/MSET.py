from setting import bigyo_data
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import linalg as spla
import os, sys, warnings, math
from evaluating import evaluating_change_point, save_result
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import argparse
parser = argparse.ArgumentParser(description='MSET')
parser.add_argument('--num', help='몇번째 데이터 셋 비교 실험인지', type = int)
args = parser.parse_args()

def kernel(x,y):

    '''
    s(x,y) = 1 - ||x-y||/(||x|| + ||y||)
    '''

    if all(x==y):
        # Handling the case of x and y both being the zero vector.
        return 1.
    else:
        return 1. - np.linalg.norm(x-y)/(np.linalg.norm(x) + np.linalg.norm(y))

def otimes(X, Y):

    m1,n = np.shape(X)
    m2,p = np.shape(Y)

    if m1!=m2:
        raise Exception('dimensionality mismatch between X and Y.')

    Z = np.zeros( (n,p) )

    if n != p:
        for i in range(n):
            for j in range(p):
                Z[i,j] = kernel(X[:,i], Y[:,j])
    else:
        for i in range(n):     
            for j in range(i, p):
                Z[i,j] = kernel(X[:,i], Y[:,j])
                Z[j,i] = Z[i,j]

    return Z

def movmean(array, window):

    n = np.size(array)
    xx = array.copy()
    y = []
    for i in range(0, window):
        y.append(np.roll(xx.tolist() + [np.nan]*window, i))
    y = np.nanmean(y, axis=0)
    l = math.ceil(window/2)

    return y[l-1:n+l-1]

class MSET:
        
    def __init__(self):
        pass
    
    def calc_W(self, X_obs):
        
        DxX_obs = otimes(self.D, X_obs)
        try:
            W = spla.lu_solve(self.LU_factors, DxX_obs)
        except:
            W = np.linalg.solve(self.DxD, DxX_obs)
    
        return W

    def fit(self, df, train_start = None, train_stop = None):
        
        self.D = df[train_start:train_stop].values.T.copy() # memory matrix
        self.SS = MinMaxScaler()
        self.D = self.SS.fit_transform(self.D.T).T
        
        self.DxD = otimes(self.D, self.D)
        self.LU_factors = spla.lu_factor(self.DxD)
        
    def predict(self, df):
        
        X_obs = df.values.T.copy() 
        X_obs = self.SS.transform(X_obs.T).T
#         pred = pd.DataFrame(index=df.index, columns=df.columns)
        pred = np.zeros(X_obs.T.shape)
        
        for i in range(X_obs.shape[1]):
            pred[[i],:] = (self.D @ self.calc_W(X_obs[:,i].reshape([-1,1]))).T
            
        return pd.DataFrame(self.SS.inverse_transform(pred), index=df.index, columns=df.columns)

train_set, test_set = bigyo_data(args.num)
total, labels = np.empty((0,5), float), ['normal','anomaly']

X_train, X_test = train_set.drop(['anomaly'], axis=1), test_set.drop(['anomaly'], axis=1)
true_outlier = np.array([test_set.anomaly]).reshape(-1,)

for i in range(1):
    ms = MSET()
    ms.fit(X_train)

    # results predicting
    Y_pred = ms.predict(X_test)
    err = np.linalg.norm(X_test - Y_pred, axis=1)
    rel_err = movmean(err/np.linalg.norm(Y_pred, axis=1), window = 60)

    UCL = 0.05

    prediction = np.array(pd.DataFrame((rel_err>UCL), test_set.index).fillna(0).any(axis=1).astype(int)).reshape(-1,)
    total = np.append(total, np.array([[precision_score(true_outlier, prediction), recall_score(true_outlier, prediction), f1_score(true_outlier, prediction), accuracy_score(true_outlier, prediction), roc_auc_score(true_outlier, prediction)]]), axis = 0)
total = total.mean(axis = 0)   # 열끼리 평균
print(total)
save = pd.DataFrame({'Name':"MSET_{}".format(args.num), 'Precision':"{:.2f}".format(total[0]), 'Recall':"{:.2f}".format(total[1]), 'F1':"{:.2f}".format(total[2]), 'ACC':"{:.2f}".format(total[3]), 'AUC':"{:.2f}".format(total[4])}, index = [0])
save.to_csv('C:/Users/yoona/Desktop/아침/비교실험.csv', index = False, mode='a', header = False)
