if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/backbone" ]; then
    mkdir ./logs/backbone
fi

itr=2
seq_len=336
tau=1.0
data=Weather
model_name=Linear3_mem

for pred_len in 24
do
  learning_rate=0.001
  leader_num=8
  state_num=20
  python -u run_longExp.py \
    --dataset $data --model $model_name  --seq_len $seq_len --pred_len $pred_len \
    --leader_num $leader_num --state_num $state_num --temperature $tau \
    --learning_rate $learning_rate --cl --itr $itr --use_mem  > logs/backbone/$model_name'_cl_mem_'$data'_'$pred_len'_K'$leader_num'_tau'$tau'_state'$state_num'_lr'$learning_rate.log 2>&1
done
for pred_len in 48 96
do
  learning_rate=0.001
  leader_num=4
  state_num=12
  python -u run_longExp.py \
    --dataset $data --model $model_name  --seq_len $seq_len --pred_len $pred_len \
    --leader_num $leader_num --state_num $state_num --temperature $tau \
    --learning_rate $learning_rate --cl --itr $itr --use_mem  > logs/backbone/$model_name'_cl_mem_'$data'_'$pred_len'_K'$leader_num'_tau'$tau'_state'$state_num'_lr'$learning_rate.log 2>&1
done
for pred_len in 192
do
  learning_rate=0.001
  leader_num=8
  state_num=8
  python -u run_longExp.py \
    --dataset $data --model $model_name --seq_len $seq_len --pred_len $pred_len \
    --leader_num $leader_num --state_num $state_num --temperature $tau \
    --learning_rate $learning_rate --cl --itr $itr --use_mem  > logs/backbone/$model_name'_cl_mem_'$data'_'$pred_len'_K'$leader_num'_tau'$tau'_state'$state_num'_lr'$learning_rate.log 2>&1
done
for pred_len in 336
do
  learning_rate=0.0005
  leader_num=4
  state_num=8
  python -u run_longExp.py \
    --dataset $data --model $model_name  --seq_len $seq_len --pred_len $pred_len \
    --leader_num $leader_num --state_num $state_num --temperature $tau \
    --learning_rate $learning_rate --cl --itr $itr --use_mem  > logs/backbone/$model_name'_cl_mem_'$data'_'$pred_len'_K'$leader_num'_tau'$tau'_state'$state_num'_lr'$learning_rate.log 2>&1
done
for pred_len in 720
do
  learning_rate=0.0005
  leader_num=2
  state_num=8
  python -u run_longExp.py \
    --dataset $data --model $model_name  --seq_len $seq_len --pred_len $pred_len \
    --leader_num $leader_num --state_num $state_num --temperature $tau \
    --learning_rate $learning_rate --cl --itr $itr --use_mem  > logs/backbone/$model_name'_cl_mem_'$data'_'$pred_len'_K'$leader_num'_tau'$tau'_state'$state_num'_lr'$learning_rate.log 2>&1
done