python -u run_longExp.py --dataset PeMSD8 --model MTGNN_mem3 --seq_len 336 --pred_len 48 --itr 1 --learning_rate 0.001  --gpu 2 --use_mem
wait
python -u run_longExp.py --dataset PeMSD8 --model MTGNN_mem3 --seq_len 336 --pred_len 96 --itr 1 --learning_rate 0.001  --gpu 2 --use_mem
wait
python -u run_longExp.py --dataset PeMSD8 --model MTGNN_mem3 --seq_len 336 --pred_len 192 --itr 1 --learning_rate 0.001  --gpu 2 --use_mem
wait
python -u run_longExp.py --dataset PeMSD8 --model MTGNN_mem3 --seq_len 336 --pred_len 336 --itr 1 --learning_rate 0.001  --gpu 2 --use_mem
wait
python -u run_longExp.py --dataset PeMSD8 --model MTGNN_mem3 --seq_len 336 --pred_len 720 --itr 1 --learning_rate 0.001  --gpu 2 --use_mem
wait
python -u run_longExp.py --dataset PeMSD8 --model MTGNN --seq_len 336 --pred_len 192 --itr 1 --learning_rate 0.001 --backbone --gpu 2 
wait
python -u run_longExp.py --dataset PeMSD8 --model MTGNN --seq_len 336 --pred_len 336 --itr 1 --learning_rate 0.001 --backbone --gpu 2 
wait
python -u run_longExp.py --dataset PeMSD8 --model MTGNN --seq_len 336 --pred_len 720 --itr 1 --learning_rate 0.001 --backbone --gpu 2 
wait
python -u run_longExp.py --dataset PeMSD8 --model MTGNN_mem3 --seq_len 336 --pred_len 48 --itr 1 --learning_rate 0.001  --gpu 2 --use_mem --cl --mem_sele
wait
python -u run_longExp.py --dataset PeMSD8 --model MTGNN_mem3 --seq_len 336 --pred_len 48 --itr 1 --learning_rate 0.001  --gpu 2 --use_mem --cl --mem_sele
wait
python -u run_longExp.py --dataset PeMSD8 --model MTGNN_mem3 --seq_len 336 --pred_len 96 --itr 1 --learning_rate 0.001  --gpu 2 --use_mem --cl --mem_sele
wait
python -u run_longExp.py --dataset PeMSD8 --model MTGNN_mem3 --seq_len 336 --pred_len 192 --itr 1 --learning_rate 0.001  --gpu 2 --use_mem --cl --mem_sele
wait
python -u run_longExp.py --dataset PeMSD8 --model MTGNN_mem3 --seq_len 336 --pred_len 336 --itr 1 --learning_rate 0.001  --gpu 2 --use_mem --cl --mem_sele
wait
python -u run_longExp.py --dataset PeMSD8 --model MTGNN_mem3 --seq_len 336 --pred_len 720 --itr 1 --learning_rate 0.001  --gpu 2 --use_mem --cl --mem_sele
wait
