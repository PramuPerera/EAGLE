import json

import torch
from safetensors.torch import load_file, save_file

ckpt = torch.load("EAGLE-Qwen2-7B-Instruct-run67-int8-aug7-llama/pytorch_model.bin")
ref_ckpt = load_file("/mnt/efs/people/yawenwuu/projects/Qwen2.5-Coder/finetuning/sft/checkpoints_efs/7B_hunks/lr5e-5-wr100-wd0.0-bsz512-maxlen8192/run67/int8/model-00003-of-00003.safetensors")

ckpt['lm_head.weight'] = ref_ckpt['lm_head.weight']

save_file(ckpt, "EAGLE-Qwen2-7B-Instruct-run67-int8-aug7-llama-converted/model.safetensors")

with open("EAGLE-Qwen2-7B-Instruct-run67-int8-aug7-llama/config.json") as rf:
    cfg = json.load(rf)

cfg = {"model_type": "eagle", "model": cfg}

with open("EAGLE-Qwen2-7B-Instruct-run67-int8-aug7-llama-converted/config.json", "w") as wf:
    json.dump(cfg, wf)
