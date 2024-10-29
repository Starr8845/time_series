if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/backbone" ]; then
    mkdir ./logs/backbone
fi

itr=1
seq_len=336
data=PeMSD8
model_name=MTGNN_mem3

for pred_len in 192 336 720
do
for learning_rate in 0.001
do
  python -u run_longExp.py \
    --dataset $data --model $model_name --seq_len $seq_len --pred_len $pred_len \
    --learning_rate $learning_rate --gpu 3 --itr 1 > logs/backbone/$model_name'_'$data'_mem_'$pred_len'_lr'$learning_rate.log 2>&1
  wait
done
done