export NCCL_TIMEOUT=900
export NCCL_P2P_DISABLE=1
export NCCL_P2P_LEVEL=NVL

cd ../../

output_path=./ucf101_zeroshot

llm_json_path=./prompts/UCF101/ucf101.json
model_path=./output_zeroshot_pretrain/ckpt_epoch_9.pth

VAL_FILE1=./configs/zero_shot/ucf101/bdc_clip_zs_ucf101_split1.yaml
VAL_FILE2=./configs/zero_shot/ucf101/bdc_clip_zs_ucf101_split2.yaml
VAL_FILE3=./configs/zero_shot/ucf101/bdc_clip_zs_ucf101_split3.yaml
VAL_FILES=($VAL_FILE1 $VAL_FILE2 $VAL_FILE3)


for split_path in "${VAL_FILES[@]}"; do
    PYTHONWARNINGS="ignore" python -m torch.distributed.launch --nproc_per_node=2 main.py \
    -cfg $split_path \
    --output $output_path \
    --llm_json_path $llm_json_path \
    --only_test \
    --wise_ft 0.0 \
    --resume $model_path
done