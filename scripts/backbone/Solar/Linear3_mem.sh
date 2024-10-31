if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/backbone" ]; then
    mkdir ./logs/backbone
fi

itr=2
seq_len=336
tau=1.0
data=Solar
model_name=Linear3_mem

learning_rate=0.005
for pred_len in 24 48
do
  leader_num=4
  state_num=16
  python -u run_longExp.py \
    --dataset $data --model $model_name  --seq_len $seq_len --pred_len $pred_len \
    --leader_num $leader_num --state_num $state_num --temperature $tau \
    --learning_rate $learning_rate --use_mem --cl --mem_sele  --gpu 3 --itr $itr > logs/backbone/$model_name'_cl_mem_sele'$data'_'$pred_len'_K'$leader_num'_tau'$tau'_state'$state_num'_lr'$learning_rate.log 2>&1
done
for pred_len in 96
do
  leader_num=4
  state_num=12
  python -u run_longExp.py \
    --dataset $data --model $model_name  --seq_len $seq_len --pred_len $pred_len \
    --leader_num $leader_num --state_num $state_num --temperature $tau \
    --learning_rate $learning_rate --use_mem --cl --mem_sele  --gpu 3 --itr $itr > logs/backbone/$model_name'_cl_mem_sele'$data'_'$pred_len'_K'$leader_num'_tau'$tau'_state'$state_num'_lr'$learning_rate.log 2>&1
done
for pred_len in 192 336
do
  leader_num=8
  state_num=16
  python -u run_longExp.py \
    --dataset $data --model $model_name  --seq_len $seq_len --pred_len $pred_len \
    --leader_num $leader_num --state_num $state_num --temperature $tau \
    --learning_rate $learning_rate --use_mem --cl --mem_sele  --gpu 3 --itr $itr > logs/backbone/$model_name'_cl_mem_sele'$data'_'$pred_len'_K'$leader_num'_tau'$tau'_state'$state_num'_lr'$learning_rate.log 2>&1
done
for pred_len in 720
do
  leader_num=8
  state_num=16
  learning_rate=0.001
  python -u run_longExp.py \
    --dataset $data --model $model_name  --seq_len $seq_len --pred_len $pred_len \
    --leader_num $leader_num --state_num $state_num --temperature $tau \
    --learning_rate $learning_rate --use_mem --cl --mem_sele  --gpu 3 --itr $itr > logs/backbone/$model_name'_cl_mem_sele'$data'_'$pred_len'_K'$leader_num'_tau'$tau'_state'$state_num'_lr'$learning_rate.log 2>&1
done