## CUDA_VISIBLE_DEVICES=2 allennlp train src/predictors/snli/snli_roberta.json --include-package src.predictors.snli.snli_dataset_reader -s trained_predictors/models/snli/
## CUDA_VISIBLE_DEVICES=2 python3 run_stage_one.py -task snli -stage1_exp mice_gold
## CUDA_VISIBLE_DECIES=0 python3 run_stage_two.py -task imdb -stage2_exp mice_binary -editor_path results/imdb/editors/mice/imdb_editor.pth
## CUDA_VISIBLE_DECIES=0 python3 run_stage_two.py -task imdb -stage2_exp mice_binary_no_search -editor_path results/imdb/editors/mice/imdb_editor.pth -max_edit_rounds 0


################################################################
####################### TRAIN PREDICTORS #######################
################################################################

# Train RACE Predictor
allennlp train src/predictors/race/race_roberta.json \
	--include-package src.predictors.race.race_dataset_reader \
	-s trained_predictors/models/race/

# Train IMDB Predictor
allennlp train src/predictors/imdb/imdb_roberta.json \
	--include-package src.predictors.imdb.imdb_dataset_reader \
	-s trained_predictors/models/imdb/ 

# Train Newsgroups Predictor
allennlp train src/predictors/newsgroups/newsgroups_roberta.json \
	--include-package src.predictors.newsgroups.newsgroups_dataset_reader \
	-s trained_predictors/models/newsgroups/

################################################################
########################## STAGE ONE ###########################
################################################################

STAGE1EXP=mice_gold

python run_stage_one.py -task imdb -stage1_exp ${STAGE1EXP} 
python run_stage_one.py -task newsgroups -stage1_exp ${STAGE1EXP} 
python run_stage_one.py -task race -stage1_exp ${STAGE1EXP} 

################################################################
########################## STAGE TWO ###########################
################################################################

STAGE2EXP=mice_binary

python run_stage_two.py -task imdb \
	-editor_path results/imdb/editors/${STAGE1EXP}/checkpoints/ \
	-stage2_exp ${STAGE2EXP} 

python run_stage_two.py -task newsgroups \
	-editor_path results/newsgroups/editors/${STAGE1EXP}/checkpoints/ \
	-stage2_exp ${STAGE2EXP}

python run_stage_two.py -task race \
	-editor_path results/race/editors/${STAGE1EXP}/checkpoints/ \
	-stage2_exp ${STAGE2EXP} 
