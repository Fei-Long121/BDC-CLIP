export NCCL_TIMEOUT=900
export NCCL_P2P_DISABLE=1
export NCCL_P2P_LEVEL=NVL

cd ../../

model_path=./output_fewshot_pretrain/ckpt_epoch_9.pth

VAL_FILE=./configs/base2novel/pretrained_on_k400/hmdb51/bdc_clip_s1.yaml
VAL_FILES=($VAL_FILE)
output_path=./hmdb51_base2novel_finetuned
llm_json_path=./prompts/HMDB51/hmdb51.json

for split_path in "${VAL_FILES[@]}"; do
    PYTHONWARNINGS="ignore" python -m torch.distributed.launch --nproc_per_node=2 main.py \
     -cfg $split_path \
     --output $output_path \
     --llm_json_path $llm_json_path \
     --finetune_fewshot $model_path
done

best_model_path="${output_path}/best.pth"

#Evaluate on base set
VAL_FILE_NOVEL=./configs/base2novel/pretrained_on_k400/hmdb51/bdc_clip_s1.yaml
VAL_FILES=($VAL_FILE_NOVEL)
for split_path in "${VAL_FILES[@]}"; do
    PYTHONWARNINGS="ignore" python -m torch.distributed.launch --nproc_per_node=2 main.py \
     -cfg $split_path \
     --llm_json_path $llm_json_path \
     --finetune_fewshot $best_model_path \
     --only_test
done

#Evaluate on novel set
VAL_FILE_NOVEL=./configs/base2novel/pretrained_on_k400/hmdb51/bdc_clip_novel_eval.yaml
VAL_FILES=($VAL_FILE_NOVEL)
for split_path in "${VAL_FILES[@]}"; do
    PYTHONWARNINGS="ignore" python -m torch.distributed.launch --nproc_per_node=2 main.py \
     -cfg $split_path \
     --llm_json_path $llm_json_path \
     --finetune_fewshot $best_model_path \
     --only_test
done