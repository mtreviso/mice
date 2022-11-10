# Commands for running MiCE without minimality

FRAC=$1
# this is with: do_sample = False
for GENERATE in beam sample
do
	CUDA_VISIBLE_DEVICES=6 python3 run_stage_two.py -task imdb -stage2_exp mice_no_binary/1110/${FRAC}_${GENERATE} -editor_path results/imdb/editors/mice/imdb_editor.pth -max_search_levels 1 -search_method linear -max_mask_frac ${FRAC} -max_edit_rounds 1 -no_filter_by_validity -generate_type ${GENERATE} -top_k 50 -num_generations 1 -generation_num_beams 15 -no_repeat_ngram_size 2
done
