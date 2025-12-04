import json
from safetensors import safe_open
# from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModelForSequenceClassification
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
from eagle.model.cnets import Model
from eagle.model.configs import EConfig
data = torch.load('mockdata/0/data_999.ckpt', weights_only=False)
with open("llama_config.json", "r") as f:
        config = json.load(f)
config = EConfig(**config)
model = Model(config, load_emb=True, path='/mnt/efs/people/yawenwuu/projects/Qwen2.5-Coder/finetuning/sft/checkpoints_efs/7B_hunks/lr5e-5-wr100-wd0.0-bsz512-maxlen8192/run67/int8')
#model = Model.from_pretrained('EAGLE-Qwen2-7B-Instruct-run67-int8-aug7-llama')
a = torch.load('pytorch_model.bin')#checkpoints_run67int8_aug13_vllm_feats_with_spec_llama_continue_mask_cosine/last-v1.ckpt')
model.embed_tokens.weight =  torch.nn.Parameter(a['state_dict']['model.embed_tokens.weight'].to(torch.bfloat16)).cuda()
model.layers[0].self_attn.q_proj.weight= torch.nn.Parameter(a['state_dict']['model.layers.0.self_attn.q_proj.weight'].to(torch.bfloat16)).cuda()
model.layers[0].self_attn.k_proj.weight= torch.nn.Parameter(a['state_dict']['model.layers.0.self_attn.k_proj.weight'].to(torch.bfloat16)).cuda()
model.layers[0].self_attn.v_proj.weight= torch.nn.Parameter(a['state_dict']['model.layers.0.self_attn.v_proj.weight'].to(torch.bfloat16)).cuda()
model.layers[0].self_attn.o_proj.weight= torch.nn.Parameter(a['state_dict']['model.layers.0.self_attn.o_proj.weight'].to(torch.bfloat16)).cuda()
model.layers[0].mlp.gate_proj.weight= torch.nn.Parameter(a['state_dict']['model.layers.0.mlp.gate_proj.weight'].to(torch.bfloat16)).cuda()
model.layers[0].mlp.up_proj.weight= torch.nn.Parameter(a['state_dict']['model.layers.0.mlp.up_proj.weight'].to(torch.bfloat16)).cuda()
model.layers[0].mlp.down_proj.weight= torch.nn.Parameter(a['state_dict']['model.layers.0.mlp.down_proj.weight'].to(torch.bfloat16)).cuda()
model.layers[0].post_attention_layernorm.weight= torch.nn.Parameter(a['state_dict']['model.layers.0.post_attention_layernorm.weight'].to(torch.bfloat16)).cuda()
model.fc.weight= torch.nn.Parameter(a['state_dict']['model.fc.weight'].to(torch.bfloat16)).cuda()
model.fc.bias= torch.nn.Parameter(a['state_dict']['model.fc.bias'].to(torch.bfloat16)).cuda()
model = model.to(torch.bfloat16).cuda()
print(data["hidden_state"][:-1,:].shape, data["input_ids"][:,1:].shape)



def paddingtensor( intensors, N):
    B, n, S = intensors.shape
    padding_tensor = torch.zeros(B, N - n, S, device=intensors.device)
    outtensors = torch.cat((intensors, padding_tensor), dim=1)
    return outtensors

def paddingtensor2D( intensors, N):
    B, n = intensors.shape
    padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype, device=intensors.device)
    outtensors = torch.cat((intensors, padding_tensor), dim=1)
    return outtensors


#predict = model(torch.roll(data["hidden_state"][:,:],1,0).unsqueeze(0).cuda(), input_ids=data["input_ids"][:,:].cuda())
hs = paddingtensor(torch.roll(data["hidden_state"],1,0).unsqueeze(0),4096).cuda().to(torch.bfloat16)
ii = paddingtensor2D(data["input_ids"],4096).to(torch.int64)
with torch.no_grad():
    predict = model(hs, ii)
fc = torch.nn.Linear(3584,152064, bias=False)
basepath = '/mnt/efs/people/yawenwuu/projects/Qwen2.5-Coder/finetuning/sft/checkpoints_efs/7B_hunks/lr5e-5-wr100-wd0.0-bsz512-maxlen8192/run67/int8'
with open(os.path.join(basepath, "model.safetensors.index.json"), "r") as f:
    index_json = json.loads(f.read())
    head_path = index_json["weight_map"]["lm_head.weight"]
with safe_open(os.path.join(basepath, head_path),
              framework="pt",
              device="cpu") as f:
    tensor_slice = f.get_slice("lm_head.weight")
    vocab_size, hidden_dim = tensor_slice.get_shape()
    tensor = tensor_slice[:, :hidden_dim].float()
fc.data = tensor#a['state_dict']['head.weight']
fc.cuda()
fc.weight.data = tensor
out = fc(predict.to(torch.float32).cpu())
res = torch.argmax(out, -1)
import IPython
IPython.embed()
