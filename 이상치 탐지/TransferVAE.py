from setting import bigyo_data, train_data
import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from evaluating import evaluating_change_point, save_result
import argparse
scaler = MinMaxScaler()
tf.disable_v2_behavior()
parser = argparse.ArgumentParser(description='Transfer-VAE')
parser.add_argument('--num', help='몇번째 데이터 셋 비교 실험인지', type = int)
parser.add_argument('--train_support_batch', help='Meta train에 사용할 데이터 갯수', type = int)
args = parser.parse_args()

def lrelu(x, leak=0.2, name='lrelu'):
	return tf.maximum(x, leak*x)

def build_dense(input_vector,unit_no,activation):    
    return tf.layers.dense(input_vector,unit_no,activation=activation,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
            bias_initializer=tf.zeros_initializer())

class MLP_VAE:
    def __init__(self,input_dim,lat_dim, outliers_fraction):      
        self.outliers_fraction = outliers_fraction     
        self.input_dim = input_dim
        self.lat_dim = lat_dim # the lat_dim can exceed input_dim    
        
        self.input_X = tf.placeholder(tf.float32,shape=[None, self.input_dim],name='source_x')
        
        self.learning_rate = 0.0005
        self.batch_size =  32
        self.train_iter = 400
        self.fine_iter = 7
        self._build_VAE()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.pointer = 0
        
    def _encoder(self):
        with tf.variable_scope('encoder',reuse=tf.AUTO_REUSE):
            l1 = build_dense(self.input_X, 6, activation=lrelu)
            l2 = build_dense(l1, 4, activation=lrelu)     
            mu = tf.layers.dense(l2,self.lat_dim)
            sigma = tf.layers.dense(l2,self.lat_dim,activation=tf.nn.softplus)
            sole_z = mu + sigma *  tf.random_normal(tf.shape(mu),0,1,dtype=tf.float32)
        return mu,sigma,sole_z
        
    def _decoder(self,z):
        with tf.variable_scope('decoder',reuse=tf.AUTO_REUSE):
            l1 = build_dense(z, 4, activation=lrelu)
            l2 = build_dense(l1, 2, activation=lrelu)
            recons_X = tf.layers.dense(l2,self.input_dim)        # 다시 원래 차원으로 복원
        return recons_X           # 복원된 x

    def _build_VAE(self):
        self.mu_z,self.sigma_z,sole_z = self._encoder()
        self.recons_X = self._decoder(sole_z)
        
        with tf.variable_scope('loss'):
            KL_divergence = 0.5 * tf.reduce_sum(tf.square(self.mu_z) + tf.square(self.sigma_z) - tf.log(1e-8 + tf.square(self.sigma_z)) - 1, 1)
            mse_loss = tf.reduce_sum(tf.square(self.input_X-self.recons_X), 1)          
            self.all_loss =  mse_loss  
            self.loss = tf.reduce_mean(mse_loss + KL_divergence)
            
        with tf.variable_scope('train'):            
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            
    def _fecth_data(self,input_data):        
        if (self.pointer+1) * self.batch_size  >= input_data.shape[0]:
            return_data = input_data[self.pointer*self.batch_size:,:]
            self.pointer = 0
        else:
            return_data =  input_data[ self.pointer*self.batch_size:(self.pointer+1)*self.batch_size,:]
            self.pointer = self.pointer + 1
        return return_data

    def train(self):
        for i in range(self.train_iter):
            train_X = scaler.fit_transform(train_data(args.num, args.train_support_batch))  # 여기서 매 iter마다 random으로 13 * 100 개씩 불러온 후 스케일링
            this_X = self._fecth_data(train_X)
            self.sess.run([self.train_op], feed_dict={ self.input_X: this_X })

    def fine_tuning(self, fine_X):
        for i in range(self.fine_iter):
            this_X = self._fecth_data(fine_X)
            self.sess.run([self.train_op], feed_dict={ self.input_X: this_X })
        self.arrage_recons_loss(fine_X)
        
    def arrage_recons_loss(self,input_data):      # reconsturction error threshold 세팅
        all_losses =  self.sess.run(self.all_loss,feed_dict={self.input_X: input_data })
        #self.test_loss = np.percentile(all_losses,(1-self.outliers_fraction)*100)         # train을 통해 reconst loss 설정   
        self.test_loss = pd.Series(all_losses).quantile(self.outliers_fraction)*3/2

    def test(self,input_data):
        return_label = []
        #print("Reconstruction threshold : ", self.test_loss)
        for index in range(input_data.shape[0]):    # test 길이인 8957
            single_X = input_data[index].reshape(1,-1)
            this_loss = self.sess.run(self.loss,feed_dict={ self.input_X: single_X })
            if this_loss < self.test_loss:
                return_label.append(0)     
            else:
                return_label.append(1)      
        return return_label

total, labels = np.empty((0,5), float), ['normal','anomaly']
mlp_vae = MLP_VAE(8, 2, 0.05)

for i in range(5):
    mlp_vae.train()    # 데이터 매번 함수 안에서 iter 마다 random sampling으로 불러와서 Meta train 학습한다.
    tmp_train_set, tmp_test_set = bigyo_data(args.num)
    test_set = pd.concat([tmp_train_set, tmp_test_set])   # 여기서 앞의 5 + 5개는 fine tuning에 쓰일겨 -- 총 5 + 12400개

    test, test_label = test_set.iloc[:, :-1], test_set.iloc[:, -1]                     

    test = scaler.transform(test)
    fine_test, real_test = test[:10], test[10:]    # 여기 10 만약에 batch size 5아니고 더 작으면 꼭 바꿔야한다!!
    true_outlier = np.array(test_label[10:]).reshape(-1,)
    mlp_vae.fine_tuning(fine_test)
    mlp_vae_predict_label = np.array(mlp_vae.test(real_test)).reshape(-1,)
    
    total = np.append(total, np.array([[precision_score(true_outlier, mlp_vae_predict_label), recall_score(true_outlier, mlp_vae_predict_label), f1_score(true_outlier, mlp_vae_predict_label), accuracy_score(true_outlier, mlp_vae_predict_label), roc_auc_score(true_outlier, mlp_vae_predict_label)]]), axis = 0)
    print(classification_report(true_outlier, mlp_vae_predict_label))
    print(total)
total = total.mean(axis = 0)   # 열끼리 평균
save = pd.DataFrame({'Name':"Transfer-VAE_{}".format(args.num), 'Precision':"{:.2f}".format(total[0]), 'Recall':"{:.2f}".format(total[1]), 'F1':"{:.2f}".format(total[2]), 'ACC':"{:.2f}".format(total[3]), 'AUC':"{:.2f}".format(total[4])}, index = [0])
save.to_csv('C:/Users/yoona/Desktop/아침/비교실험.csv', index = False, mode='a', header = False)