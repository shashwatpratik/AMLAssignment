main.py \
--root test \
-a M3T \
--optim adam \
--lr 0.0003 \
--max-epoch 20 \
--eval-freq 5 \
--train-batch-size 4 \
--test-batch-size 4 \
--save-dir $ENV(PWD)/logs/B4_M3T_LR0003