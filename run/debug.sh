gpu=1
run_name='llama'
output_dir='/opt/data/private/wtc_beifen/bird/exp_res/output/llama2_7b_bf16_debug'



export WANDB_API_KEY=ea2ecdf4a828acb51d4ccf063512edd1f2f389d7
export WANDB_PROJECT=llama2_debug


echo '''flying'''

# CUDA_VISIBLE_DEVICES=1  python train_bird.py --seed 1 --cfg /opt/data/private/wtc_beifen/bird/finetuning/configure/experiment/T5_large_finetune_bird_kg.cfg \
# --run_name ${run_name} --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps \
# --eval_steps 2000 --metric_for_best_model avr --greater_is_better true --save_strategy steps --save_steps 2000 \
# --save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 4 --num_train_epochs 20 \
# --adafactor true --learning_rate 5e-5 --do_train true --do_eval true --do_predict false --predict_with_generate true \
# --output_dir ${output_dir} --per_device_train_batch_size 1 --per_device_eval_batch_size 4 \
# --generation_num_beams 1 --generation_max_length 128 --input_max_length 1024 --ddp_find_unused_parameters true \
# --report_to wandb --overwrite_output_dir true --load_weights_from /opt/data/private/wtc_beifen/bird/finetuning/output/T5_large_bird_kg/checkpoint-12000 \

CUDA_VISIBLE_DEVICES=0,1 python train_bird.py --seed 1 --cfg /opt/data/private/wtc_beifen/bird/finetuning/configure/experiment/llama2.cfg \
--run_name ${run_name} --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps \
--eval_steps 2000 --metric_for_best_model avr --greater_is_better true --save_strategy steps --save_steps 2000 \
--save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 1 --num_train_epochs 20 \
--adafactor true --learning_rate 5e-5 --do_train false --do_eval true --do_predict false --predict_with_generate true \
--output_dir ${output_dir} --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
--generation_num_beams 1 --generation_max_length 128 --input_max_length 1024 --ddp_find_unused_parameters true \
--report_to wandb --overwrite_output_dir true --is_decoder_only true\
