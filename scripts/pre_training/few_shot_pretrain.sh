export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export PYTORCH_LOG_LEVEL=ERROR

cd ../../

llm_json_path=./prompts/K400/k400.json
output_path=./output_fewshot_pretrain
pre_train_config=configs/pre_training/few_shot/bdc_clip_fs_pre_k400.yaml

PYTHONWARNINGS="ignore" python -m torch.distributed.launch \
              --nproc_per_node=2 main.py \
              -cfg $pre_train_config \
              --llm_json_path $llm_json_path \
              --output $output_path
