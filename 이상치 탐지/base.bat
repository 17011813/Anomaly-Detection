@echo off

FOR /L %%x IN (4, 1, 16) DO (
python AE.py --num=%%x
python IF.py --num=%%x
python LSTM.py --num=%%x
python LSTMAE.py --num=%%x
python LSTMVAE.py --num=%%x
python MSET.py --num=%%x
python PCA.py --num=%%x
python Tsquared.py --num=%%x
python VAE.py --num=%%x
python TransferVAE.py --num=%%x --train_support_batch=10
)