CUDA_VISIBLE_DEVICES=6

PREDICTOR_DIR=trained_predictors/0105

allennlp train src/predictors/newsgroups/newsgroups_t5.json \
        --include-package src.predictors.newsgroups.newsgroups_dataset_reader \
        --include-package src.predictors.t5_classifier \
        -s trained_predictors/0105/newsgroups/model \
	--overrides '{"trainer": {"cuda_device": '${CUDA_VISIBLE_DEVICES}'}}'
#	--overrides '{"trainer": {"cuda_device": '${CUDA_VISIBLE_DEVICES}', "num_epochs": 10}}'


################################################################
########################## STAGE ONE ###########################
################################################################

: '
STAGE1EXP=t5_mice_gold
#STAGE1EXP=NEW_t5_mice_gold

CUDA_VISIBLE_DEVICES=6 python run_stage_one.py -task newsgroups \
	-stage1_exp ${STAGE1EXP} \
	-predictor_dir ${PREDICTOR_DIR} \
	-predictor_name t5_predictor \
	-editor_model_name t5-small \
	-model_max_length=512 
#	-train_batch_size=64 \
#	-val_batch_size=32

################################################################
########################## STAGE ONE ###########################
################################################################

########### NO BINARY SEARCH ###########

for FRAC in 0.3 0.5
do
	for GENERATE in sample beam
	do
		CUDA_VISIBLE_DEVICES=7 python3 run_stage_two.py -task newsgroups \
			-stage2_exp mice_no_binary/0105_gold_contrast/${FRAC}_${GENERATE} \
			-predictor_dir ${PREDICTOR_DIR} \
			-predictor_name t5_predictor \
			-editor_path results/newsgroups/editors/${STAGE1EXP}/checkpoints/best.pth \
			-editor_model_name t5-small \
			-max_search_levels 1 \
			-search_method linear \
			-max_mask_frac ${FRAC} \
			-max_edit_rounds 1 \
			-no_filter_by_validity \
			-generate_type ${GENERATE} \
			-top_k 50 \
			-num_generations 1 \
			-generation_num_beams 15 \
			-no_repeat_ngram_size 2 \
			-contrast_pred_idx None 
	done
done


############ BINARY SEARCH ############
: '
Same hyperparams except binary-search-specific params: max_edit_rounds, max_mask_frac, max_search_levels, search_method
Setting generation-specific (like top_k, beam vs. sample, etc.) to be same as current set-up
TODO: what about num_generations and no_filter_by_validity? -> set as same as current set-up
'

for GENERATE in sample beam
do
	CUDA_VISIBLE_DEVICES=7 python3 run_stage_two.py -task newsgroups \
		-stage2_exp mice_binary/0105_gold_contrast/${GENERATE} \
		-predictor_dir ${PREDICTOR_DIR} \
		-predictor_name t5_predictor \
		-editor_path results/newsgroups/editors/${STAGE1EXP}/checkpoints/best.pth \
		-editor_model_name t5-small \
		-generate_type ${GENERATE} \
		-top_k 50 \
		-num_generations 1 \
		-generation_num_beams 15 \
		-no_filter_by_validity \
		-no_repeat_ngram_size 2 \
		-contrast_pred_idx None 
done
 '
