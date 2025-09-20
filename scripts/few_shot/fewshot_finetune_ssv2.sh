export NCCL_TIMEOUT=900
export NCCL_P2P_DISABLE=1
export NCCL_P2P_LEVEL=NVL

cd ../../

model_path=./output_fewshot_pretrain/ckpt_epoch_9.pth

VAL_FILE1=./configs/few_shot/pretrained_on_k400/ssv2/bdc_clip_fs_ssv2_2_shot.yaml
VAL_FILE2=./configs/few_shot/pretrained_on_k400/ssv2/bdc_clip_fs_ssv2_4_shot.yaml
VAL_FILE3=./configs/few_shot/pretrained_on_k400/ssv2/bdc_clip_fs_ssv2_8_shot.yaml
VAL_FILE4=./configs/few_shot/pretrained_on_k400/ssv2/bdc_clip_fs_ssv2_16_shot.yaml
VAL_FILES=($VAL_FILE1 $VAL_FILE2 $VAL_FILE3 $VAL_FILE4)

base_output_path=./ssv2_few_shot_finetuned
llm_json_path=./prompts/SSv2/ssv2.json

for split_path in "${VAL_FILES[@]}"; do
    shot=$(echo "$split_path" | grep -oP '\d+(?=_shot\.yaml)')
    output_path="${base_output_path}_${shot}_shot"
    
    PYTHONWARNINGS="ignore" python -m torch.distributed.launch --nproc_per_node=2 main.py \
     -cfg $split_path \
     --output $output_path \
     --llm_json_path $llm_json_path \
     --finetune_fewshot $model_path
done