if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/backbone" ]; then
    mkdir ./logs/backbone
fi

itr=1
seq_len=336
data=Weather
model_name=MTGNN_mem2

for pred_len in 24 48 96 192 336 720
do
for learning_rate in 0.0001
do
  python -u run_longExp.py \
    --dataset $data --model $model_name --seq_len $seq_len --pred_len $pred_len \
    --itr $itr \
    --learning_rate $learning_rate --use_mem > logs/backbone/$model_name'_'$data'_mem_'$pred_len'_lr'$learning_rate.log 2>&1
done
done