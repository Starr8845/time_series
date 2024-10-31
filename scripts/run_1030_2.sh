python -u run_longExp.py --dataset Solar  --model PatchTST2 --seq_len 336 --pred_len 24 --itr 1 --learning_rate 0.001 --pct_start 0.1 --backbone --gpu 2
wait 
python -u run_longExp.py --dataset Solar  --model PatchTST2 --seq_len 336 --pred_len 48 --itr 1 --learning_rate 0.001 --pct_start 0.1 --backbone --gpu 2
wait 
python -u run_longExp.py --dataset Solar  --model PatchTST2 --seq_len 336 --pred_len 96 --itr 1 --learning_rate 0.001 --pct_start 0.1 --backbone --gpu 2
wait 
python -u run_longExp.py --dataset Solar  --model PatchTST2 --seq_len 336 --pred_len 192 --itr 1 --learning_rate 0.001 --pct_start 0.1 --backbone --gpu 2
wait 
python -u run_longExp.py --dataset Solar  --model PatchTST2 --seq_len 336 --pred_len 336 --itr 1 --learning_rate 0.001 --pct_start 0.1 --backbone --gpu 2
wait 
python -u run_longExp.py --dataset Solar  --model PatchTST2 --seq_len 336 --pred_len 720 --itr 1 --learning_rate 0.001 --pct_start 0.1 --backbone --gpu 2
wait 