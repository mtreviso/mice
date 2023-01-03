CUDA_VISIBLE_DEVICES=6

PREDICTOR_DIR=trained_predictors/t5_imdb

: '
allennlp train src/predictors/imdb/imdb_t5.json \
        --include-package src.predictors.imdb.imdb_dataset_reader \
        --include-package src.predictors.t5_classifier \
        -s ${PREDICTOR_DIR}/model \
	--overrides '{"trainer": {"cuda_device": '${CUDA_VISIBLE_DEVICES}'}}'
'

################################################################
########################## STAGE ONE ###########################
################################################################

STAGE1EXP=t5_mice_gold

#python run_stage_one.py -task imdb \
#	-stage1_exp ${STAGE1EXP} \
#	-predictor_dir ${PREDICTOR_DIR} \
#	-predictor_name t5_predictor \
#	-editor_model_name t5-small

################################################################
########################## STAGE ONE ###########################
################################################################

GENERATE=sample

########### NO BINARY SEARCH ###########
: '
FRAC=$1
python3 run_stage_two.py -task imdb \
	-stage2_exp mice_no_binary/1202/${FRAC}_${GENERATE} \
	-predictor_dir ${PREDICTOR_DIR} \
	-predictor_name t5_predictor \
	-editor_path results/imdb/editors/${STAGE1EXP}/checkpoints/best.pth \
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
	-no_repeat_ngram_size 2
'

############ BINARY SEARCH ############
: '
Same hyperparams except binary-search-specific params: max_edit_rounds, max_mask_frac, max_search_levels, search_method
Setting generation-specific (like top_k, beam vs. sample, etc.) to be same as current set-up
TODO: what about num_generations and no_filter_by_validity? -> set as same as current set-up
'

python3 run_stage_two.py -task imdb \
	-stage2_exp mice_binary/1228/${GENERATE} \
	-predictor_dir ${PREDICTOR_DIR} \
	-predictor_name t5_predictor \
	-editor_path results/imdb/editors/${STAGE1EXP}/checkpoints/best.pth \
	-editor_model_name t5-small \
	-generate_type ${GENERATE} \
	-top_k 50 \
	-num_generations 1 \
	-generation_num_beams 15 \
	-no_filter_by_validity \
	-no_repeat_ngram_size 2
