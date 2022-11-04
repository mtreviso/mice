CUDA_VISIBLE_DEVICES=2 python3 run_stage_one.py -task snli -stage1_exp mice_gold
CUDA_VISIBLE_DEVICES=2 python3 run_stage_two.py -task snli -stage2_exp mice_binary_no_search -editor_path results/snli/editors/mice_gold/checkpoints/best.pth
