python -u run_longExp.py --dataset PeMSD8 --model PatchTST2_mem --seq_len 336 --pred_len 192 --itr 1 --learning_rate 0.001 --backbone --gpu 2 --batch_size 4 --pct_start 0.2 --use_mem
wait
python -u run_longExp.py --dataset PeMSD8 --model PatchTST2_mem --seq_len 336 --pred_len 336 --itr 1 --learning_rate 0.001 --backbone --gpu 2 --batch_size 4 --pct_start 0.2 --use_mem
wait
python -u run_longExp.py --dataset PeMSD8 --model PatchTST2_mem --seq_len 336 --pred_len 720 --itr 1 --learning_rate 0.001 --backbone --gpu 2 --batch_size 4 --pct_start 0.2 --use_mem
wait