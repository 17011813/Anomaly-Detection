import pandas as pd
import numpy as np
import random
from itertools import permutations

def bigyo_data(num):
    print("{} 번째 실험 중 ~~~".format(num))
    data = pd.read_csv('C:\\Users\\yoona\\Desktop\\아침\\data\\새로운실험\\{}\\{}_비교실험.csv'.format(num, num))
    anomaly_index, normal_index = data.index[(data['anomaly'] == 1)], data.index[(data['anomaly'] == 0)]
    train_anomaly, train_normal = data.loc[anomaly_index].iloc[:1], data.loc[normal_index].iloc[:9]
    test_anomaly, test_normal = data.loc[anomaly_index].iloc[1:], data.loc[normal_index].iloc[9:]
    
    train_set, test_set = pd.concat([train_normal, train_anomaly]), pd.concat([test_normal, test_anomaly])
    print(train_set.shape, test_set.shape)
    return train_set, test_set

def train_data(num, batch_size):
    meta_batch_size, tasks = 12, []
    if num == 4 : digits = list(set([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]))  # 여기서 훈련에 쓸 digit 제외해줘야한다.
    if num == 5 : digits = list(set([4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]))
    if num == 6 : digits = list(set([4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]))
    if num == 7 : digits = list(set([4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16]))
    if num == 8 : digits = list(set([4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16]))
    if num == 9 : digits = list(set([4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16]))
    if num == 10 : digits = list(set([4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16]))
    if num == 11 : digits = list(set([4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16]))
    if num == 12 : digits = list(set([4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16]))
    if num == 13 : digits = list(set([4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16]))
    if num == 14 : digits = list(set([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16]))
    if num == 15 : digits = list(set([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16]))
    if num == 16 : digits = list(set([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]))

    train_digits, normal_ids_per_label, anormal_ids_per_label, train_size = list(set(digits)), {}, {}, 0   # 라벨당 train input id니까 7,8,9 각 id에 대한 input들 모으려고
    df = pd.read_csv('C:/Users/yoona/Desktop/아침/data/새로운실험/{}/re_{}.csv'.format(num, num))
    train_inputs, train_labels, isit = df.iloc[:, :-2], np.array(df['label']).reshape(-1,), np.array(df['anomaly']).reshape(-1,)

    for digit in train_digits:                        # train label 내 train_digit인 7, 8, 9 각각에 대해
        normal_ids = np.where((digit == train_labels) & (isit == 0.0))[0]    
        anomal_ids = np.where((digit == train_labels) & (isit == 1.0))[0]
        train_size = train_size + len(normal_ids) + len(anomal_ids)                         # 해당 class(digit)랑 같은 데이터 길이만큼 train이 됨..
        random.shuffle(normal_ids)
        random.shuffle(anomal_ids)
        normal_ids_per_label.update({digit: normal_ids})      # {0:25,36,57 ...} 이런식으로 각 task 숫자의 index ids위치
        anormal_ids_per_label.update({digit: anomal_ids})

    while True:   
        tasks_remaining = meta_batch_size - len(tasks) 
        if tasks_remaining <= 0: break                       
        tasks_to_add = list(permutations(digits, 1))               
        n_tasks_to_add = min(len(tasks_to_add), tasks_remaining)    
        tasks.extend(tasks_to_add[:n_tasks_to_add])    

    num_inputs_per_meta_batch = (batch_size * meta_batch_size) *2    # (10 * 13) * 2  -- 여기서는 *2를 자체적으로 해줘야함

    for i in range(min(train_size // num_inputs_per_meta_batch, 1000)):  # 트레인 (23875 // 600) __ 39 테스트 (1090 // 600) __ 1 - batch에 따라 만들 수 있는 세트
        all_indexs, all_labels = np.array([], dtype=np.int32), np.array([], dtype=np.int32) 
        for task in tasks:   # 3번 도는데 tasks는 [(7,), (8,), (9,)] 3개 (train set)         [(0,), (1,)] 2개 (test set)
            labels = np.empty(batch_size, dtype=np.int32)
            label_indexs = np.append(np.random.choice(normal_ids_per_label[task[0]], 9), np.random.choice(anormal_ids_per_label[task[0]], 1))    # a를 위해 중복을 허용해서 정상 데이터 중 9개, 비정상 데이터 중 1개 뽑아주고
            label_indexs = np.append(label_indexs, np.random.choice(normal_ids_per_label[task[0]], 9))           # b를 위해 중복을 허용해서 정상 데이터 중 9개                  
            label_indexs = np.append(label_indexs, np.random.choice(anormal_ids_per_label[task[0]], 1))          # b를 위해 중복을 허용해서 비정상 데이터 중 1개
            #print("{}에 대한 정상이랑 비정상 잘 뽑혔나~?".format(task), isit[label_indexs])
            labels.fill(task[0])                                       # labels에는 7 label 어떤 클래스인지 라벨값  --  (6200,)  각 task 0또는 1에 대해 각 6200개씩
            all_labels, all_indexs = np.append(all_labels, labels), np.append(all_indexs, label_indexs) # all_ids에는 7에 해당하는 idx 진짜 위치 인덱스

    return train_inputs.loc[all_indexs]   # indexs에 해당하는 데이터만 가져오기