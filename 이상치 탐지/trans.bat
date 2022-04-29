@echo off

FOR /L %%x IN (4, 1, 16) DO (
python TransferAE.py --num=%%x --train_support_batch=10
python TransferVAE.py --num=%%x --train_support_batch=10
)

