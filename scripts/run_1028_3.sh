
python -u run_longExp.py --dataset Weather --model PatchTST2 --seq_len 336 --pred_len 96 --itr 1 --learning_rate 0.0001 --backbone --gpu 1
wait
python -u run_longExp.py --dataset Weather --model PatchTST2 --seq_len 336 --pred_len 192 --itr 1 --learning_rate 0.0001 --backbone --gpu 1
wait
python -u run_longExp.py --dataset Weather --model PatchTST2 --seq_len 336 --pred_len 336 --itr 1 --learning_rate 0.0001 --backbone --gpu 1
wait
python -u run_longExp.py --dataset Weather --model PatchTST2 --seq_len 336 --pred_len 720 --itr 1 --learning_rate 0.0001 --backbone --gpu 1
wait
python -u run_longExp.py --dataset wind --model PatchTST2 --seq_len 336 --pred_len 24 --itr 1 --learning_rate 0.0001 --backbone --gpu 1
wait
python -u run_longExp.py --dataset wind --model PatchTST2 --seq_len 336 --pred_len 48 --itr 1 --learning_rate 0.0001 --backbone --gpu 1
wait
python -u run_longExp.py --dataset wind --model PatchTST2 --seq_len 336 --pred_len 96 --itr 1 --learning_rate 0.0001 --backbone --gpu 1
wait
python -u run_longExp.py --dataset wind --model PatchTST2 --seq_len 336 --pred_len 192 --itr 1 --learning_rate 0.0001 --backbone --gpu 1
wait
python -u run_longExp.py --dataset Weather --model PatchTST2 --seq_len 336 --pred_len 336 --itr 1 --learning_rate 0.0001 --backbone --gpu 1
wait
python -u run_longExp.py --dataset wind --model PatchTST2 --seq_len 336 --pred_len 720 --itr 1 --learning_rate 0.0001 --backbone --gpu 1
wait