python -u run_longExp.py --dataset Solar  --model MTGNN3 --seq_len 336 --pred_len 24 --itr 1 --learning_rate 0.005 --backbone --gpu 0
wait
python -u run_longExp.py --dataset Solar  --model MTGNN3 --seq_len 336 --pred_len 48 --itr 1 --learning_rate 0.005 --backbone --gpu 0
wait
python -u run_longExp.py --dataset Solar  --model MTGNN3 --seq_len 336 --pred_len 96 --itr 1 --learning_rate 0.005 --backbone --gpu 0
wait
python -u run_longExp.py --dataset Solar  --model MTGNN3 --seq_len 336 --pred_len 192 --itr 1 --learning_rate 0.005 --backbone --gpu 0
wait
python -u run_longExp.py --dataset Solar  --model MTGNN3 --seq_len 336 --pred_len 336 --itr 1 --learning_rate 0.005 --backbone --gpu 0
wait
python -u run_longExp.py --dataset Solar  --model MTGNN3 --seq_len 336 --pred_len 720 --itr 1 --learning_rate 0.005 --backbone --gpu 0
wait