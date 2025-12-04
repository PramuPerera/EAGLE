import json
from safetensors import safe_open
import torch
def load_tensors(file):
    tensors = {}
    with safe_open(file, framework="pt", device=1) as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors
#/mnt/efs/people/yawenwuu/projects/Qwen2.5-Coder/finetuning/sft/checkpoints_efs/7B_hunks/lr5e-5-wr100-wd0.0-bsz512-maxlen8192/run67/int8
import torch
from safetensors.torch import load_file, save_file
ckpt = load_file("checkpoints/state_12/model.safetensors", device='cpu')
import IPython
IPython.embed()
ref_ckpt =  torch.load("/mnt/efs/people/yawenwuu/sft/checkpoints/7B_hunks/lr5e-5-wr100-wd0.0-bsz256-maxlen3270/run13/pytorch_model-00004-of-00004.bin")#, device='cpu')
#ckpt = load_tensors("checkpoints/state_12/model_1.safetensors")
#ref_ckpt = load_tensors("/mnt/efs/people/yawenwuu/sft/checkpoints/7B_hunks/lr5e-5-wr100-wd0.0-bsz256-maxlen3270/run13/pytorch_model-00004-of-00004.bin")
ckpt['lm_head.weight'] = ref_ckpt['lm_head.weight']

save_file(ckpt, "checkpoints/model_6/model.safetensors")

#with open("checkpoints/config.json") as rf:
#    cfg = json.load(rf)

#cfg = {"model_type": "eagle", "model": cfg}

#with open("checkpoints/config.json", "w") as wf:
#    json.dump(cfg, wf)
