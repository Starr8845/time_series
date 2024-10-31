python -u run_longExp.py --dataset Solar  --model MTGNN_mem3 --seq_len 336 --pred_len 24 --itr 1 --learning_rate 0.001 --gpu 1 --batch_size=32 --use_mem
wait 
python -u run_longExp.py --dataset Solar  --model MTGNN_mem3 --seq_len 336 --pred_len 48 --itr 1 --learning_rate 0.001 --gpu 1 --batch_size=32 --use_mem
wait 
python -u run_longExp.py --dataset Solar  --model MTGNN_mem3 --seq_len 336 --pred_len 96 --itr 1 --learning_rate 0.001 --gpu 1 --batch_size=32 --use_mem
wait 
python -u run_longExp.py --dataset Solar  --model MTGNN_mem3 --seq_len 336 --pred_len 192 --itr 1 --learning_rate 0.001 --gpu 1 --batch_size=32 --use_mem
wait 
python -u run_longExp.py --dataset Solar  --model MTGNN_mem3 --seq_len 336 --pred_len 336 --itr 1 --learning_rate 0.001 --gpu 1 --batch_size=32 --use_mem
wait 
python -u run_longExp.py --dataset Solar  --model MTGNN_mem3 --seq_len 336 --pred_len 720 --itr 1 --learning_rate 0.001 --gpu 1 --batch_size=32 --use_mem
wait 