# base로 돌려야하는 애들

import pandas as pd
import ConvAE
import MSET
import LSTMAE
import PCA
import LSTM
import LSTMVAE
import Tsquared
import AE
import IF

test_support_batch = 100    # Meta test시 test finetuning을 위한 support set batch 크기
test_query_batch = [1260, 310, 302, 247, 303, 301, 486, 351, 209, 165, 202, 6209, 1417]   # 매 비교실험마다 test query 개수 달라서

for i, real_test in enumerate(test_query_batch, start = 4):                               # 4부터 16까지 비교 실험
    print("{}번째 비교실험 중~~~~".format(i))
    AE.main(num = i, test_batch = test_support_batch, test_size = real_test)     # 매번 몇번째 데이터로 비교실험할지 i를 건네준다.
    ConvAE.main(num = i, test_batch = test_support_batch, test_size = real_test)
    IF.main(num = i, test_batch = test_support_batch, test_size = real_test)
    LSTM.main(num = i, test_batch = test_support_batch, test_size = real_test)
    LSTMAE.main(num = i, test_batch = test_support_batch, test_size = real_test)
    LSTMVAE.main(num = i, test_batch = test_support_batch, test_size = real_test)
    MSET.main(num = i, test_batch = test_support_batch, test_size = real_test)
    PCA.main(num = i, test_batch = test_support_batch, test_size = real_test)
    Tsquared.main(num = i, test_batch = test_support_batch, test_size = real_test)
    save = pd.DataFrame({'Name':" ", 'Precision':" ", 'Recall':" ", 'F1':" "})
    save.to_csv('C:/Users/yoona/Desktop/아침/예측성능.csv', index = False, mode='a', header = False)
