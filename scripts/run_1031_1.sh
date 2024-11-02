python -u run_longExp.py --dataset PeMSD8 --model PatchTST --seq_len 336 --pred_len 96 --itr 1 --learning_rate 0.001 --backbone --gpu 1 --batch_size 16 --pct_start 0.2
wait
python -u run_longExp.py --dataset PeMSD8 --model PatchTST2_mem --seq_len 336 --pred_len 24 --itr 1 --learning_rate 0.001 --backbone --gpu 1 --batch_size 4 --pct_start 0.2 --use_mem
wait
python -u run_longExp.py --dataset PeMSD8 --model PatchTST2_mem --seq_len 336 --pred_len 48 --itr 1 --learning_rate 0.001 --backbone --gpu 1 --batch_size 4 --pct_start 0.2 --use_mem
wait
python -u run_longExp.py --dataset PeMSD8 --model PatchTST2_mem --seq_len 336 --pred_len 72 --itr 1 --learning_rate 0.001 --backbone --gpu 1 --batch_size 4 --pct_start 0.2 --use_mem
wait