import argparse
import copy
import tqdm
parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=100)
parser.add_argument('--index', type=int, default=1)
parser.add_argument('--gpu_index', type=int, nargs='+', default=[0])
parser.add_argument('--outdir', type=str, default='outdir0')
args = parser.parse_args()
import os
import numpy as np
#os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
from datasets import load_dataset, concatenate_datasets, Dataset
import json
from fastchat.model.model_adapter import get_conversation_template
from torch.utils.data import DataLoader
bigname="/mnt/efs/people/yawenwuu/sft/checkpoints/7B_hunks/lr5e-5-wr100-wd0.0-bsz512-maxlen8192/run21/"
bigname="/mnt/efs/people/yawenwuu/projects/Qwen2.5-Coder/finetuning/sft/checkpoints_efs/7B_hunks/lr5e-5-wr100-wd0.0-bsz512-maxlen8192/run67/int8"#"/mnt/efs/people/yawenwuu/sft/checkpoints/7B_hunks/lr5e-5-wr100-wd0.0-bsz256-maxlen3270/run13"
# smallname = "/home/lyh/weights/hf/llama/7B/"
#bigname='/mnt/efs/people/pramudi/model/models/1b'


def longest_common_prefix(list1, list2):
    prefix_length = 0
    min_length = min(len(list1), len(list2))

    for i in range(min_length):
        if list1[i] == list2[i]:
            prefix_length += 1
        else:
            break

    common_prefix = list1[:prefix_length]
    return common_prefix, prefix_length


def build_dataset_rank(
        tokenizer, split="train",
        select=None,
):
    datasets = []
    for i in ['python','java','typescript','javascript']:
        datasets.append(load_dataset("json", data_files=f"/mnt/efs/people/eyuyu/projects/genie/project_context_ablation/v13_project_context_data/train_data/{i}/combined.jsonl"))
    ds = concatenate_datasets([datasets])
    #ds = load_dataset('json', data_files="/home/hongyanz/scratch/data/ShareGPT_V4.3_unfiltered_cleaned_split.json")
    #ds = ds['train']
    ds = ds.shuffle(seed=100)
    # ds1 = ds.select(range(100,200))
    # dst=ds.select(range(200,300))
    # ds2=ds.select(range(300,len(ds)))
    original_columns1 = ds1.column_names
    # original_columns2 = ds2.column_names
    num_proc = 92

    '''def preprocess_function(examples):
        new_examples = {
            "input_ids": [],
            "label": []
        }
        for i in range(len(examples['id'])):
            new_examples["input_ids"].append(i[''])
            new_examples["label"].append(loss_mask[None,:])

        return new_examples

    ds1 = ds1.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns1,
        load_from_cache_file=False
    )'''

    # ds1 = ds1.filter(lambda x: len(x["input_ids"]) < 1024, batched=False)
    # ds1 = ds1.filter(lambda x: x['queryf'] not in gqs, batched=False)
    # ds1 = ds1.filter(lambda x: "Are there any tips in regards to teaching" in x['queryf'], batched=False)

    #ds1.set_format(type="torch")
    # ds2.set_format(type="torch")
    # dst.set_format(type="torch")
    return ds1
tokenizer = AutoTokenizer.from_pretrained('/mnt/efs/people/pramudi/HF_HOME/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct-FP8/snapshots/e8ab3f2db9e388999a004eea5a31c16a8b517bc0',use_fast=False) #features: ['test_input_ids', 'test_output_ids', 'input_ids', 'label', 'metadata', 'category', 'thinking_or_reasoning', 'new_category_recommendation']
#ds  = concatenate_datasets([Dataset.from_file(f'/mnt/efs/people/pramudi/Eagle/data-base/data-000{"{:02d}".format(i)}-of-00013.arrow') for i in range(0,13)]) #build_dataset_rank(bigtokenizer)     
datasets = []
for i in ['python','java','typescript','javascript']:
    datasets.append(load_dataset("json", data_files=f"/mnt/efs/people/eyuyu/projects/genie/project_context_ablation/v13_project_context_data/train_data/{i}/combined.jsonl")['train'])
ds = concatenate_datasets(datasets)

#ds  = concatenate_datasets([Dataset.from_file(f'/mnt/efs/people/yawenwuu/projects/Qwen2.5-Coder/finetuning/sft/dataset/code_edit_efs/context/hunks_v2/merged_output_part_0_39_mixed_pc_non_pc_dedup_threshold_09_chat_tokenized_v2_user_masked_with_codeediteval_bugbash/train_packed_max_len_8192/data-000{"{:02d}".format(i)}-of-00015.arrow') for i in range(1,14)]) #build_dataset_rank(bigtokenizer)

'''def list_files(path):
    datapath = []
    for root, directories, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            datapath.append(file_path)
    return datapath

datapath = list_files('/mnt/efs/people/pramudi/Eagle/data_aug_6')'''




# quantization_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#     )
# bigmodel = AutoModelForCausalLM.from_pretrained(bigname, load_in_4bit=True, device_map={"": 0}, )
# smallmodel = AutoModelForCausalLM.from_pretrained(smallname, load_in_4bit=True, device_map={"": 1}, )
#bigmodel = AutoModelForCausalLM.from_pretrained(bigname,  device_map="auto",torch_dtype=torch.float16)
#bigmodel = AutoModelForCausalLM.from_pretrained(bigname,  device_map="auto",load_in_8bit=True)
#bigmodel.eval()


@torch.no_grad()
def ge(data, LLM, tokenizer):
    with torch.no_grad():
                input_ids0=data["left_context"]
                input_ids0 = torch.tensor(tokenizer.encode(input_ids0))
                input_ids0 = input_ids0.cuda()
                print(input_ids0.shape)
                input_ids0 = input_ids0[:4097]
                input_ids = input_ids0[:-1]
                labels = input_ids0[1:]
                #outs_big =  bigmodel(input_ids, output_hidden_states=True
                outs_big = llm.generate( prompt_token_ids=input_ids.detach().cpu().numpy().tolist(), sampling_params=sampling_params) # bigmodel(data)#, output_hidden_states=True)
                hidden_state_big = [] # outs_big.hidden_states[-1]
                max_prob_tokens_big = 0 #torch.argmax(outs_big.logits, dim=-1)
                #probs = torch.softmax(outs_big.logits, dim=-1)
                #maxp=probs[0].max(dim=1).values
                td={"input_ids":input_ids.detach().cpu().numpy(),"hidden_state":None,'label':labels }
                return td

'''outdir = f'{args.outdir}/{args.index}'
if not os.path.exists(outdir):
    os.makedirs(outdir)'''

def writedata(name,data_point):
    if not os.path.exists(name):
        os.makedirs(name)
    current_length=len(os.listdir(name))
    idx=current_length
    torch.save(data_point, f'{name}/data_{idx}.ckpt')
train_dataloader=DataLoader(ds, batch_size=1)
#for id,data in tqdm(enumerate(train_dataloader)):
#    print(id)
i = 0
from vllm import LLM, SamplingParams
import gc
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1)
llm = LLM(enforce_eager=True, 
    enable_prefix_caching=False,
    model='/mnt/efs/people/pramudi/HF_HOME/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct-FP8/snapshots/e8ab3f2db9e388999a004eea5a31c16a8b517bc0',#'/mnt/efs/people/yawenwuu/projects/Qwen2.5-Coder/finetuning/sft/checkpoints_efs/7B_hunks/lr5e-5-wr100-wd0.0-bsz512-maxlen8192/run67/int8',#'/mnt/efs/people/yawenwuu/sft/checkpoints/7B_hunks/lr5e-5-wr100-wd0.0-bsz256-maxlen3270/run13',#'Qwen/Qwen2-7B-Instruct',#'lmsys/vicuna-7b-v1.3',# '/mnt/efs/people/yawenwuu/sft/checkpoints/7B_hunks/lr5e-5-wr100-wd0.0-bsz256-maxlen3270/run13',#'lmsys/vicuna-7b-v1.3',#"meta-llama/Meta-Llama-3-8B-Instruct",
    tensor_parallel_size=2,
    speculative_config= None#{
#        "method": "eagle3",
#    "model": 'EAGLE3-Qwen2-7B-Instruct-run67-abation',# 'EAGLE-Qwen2-7B-Instruct-run67-int8-q28kv4_30k_n2',#'EAGLE-Qwen2-7B-Instruct-run67-int8-aug16',#'EAGLE-Qwen2-7B-Instruct',#'/mnt/efs/people/pramudi/EAGLE-Qwen2-7B-Instruct',#'yuhuili/EAGLE-Qwen2-7B-Instruct',#'/mnt/efs/people/pramudi/draft3',#'yuhuili/EAGLE-Vicuna-7B-v1.3',# 'state_0',#'yuhuili/EAGLE-Vicuna-7B-v1.3',#"yuhuili/EAGLE-LLaMA3-Instruct-8B",
#     "draft_tensor_parallel_size": 1,
#    "num_speculative_tokens": 8,
#    }
)
import shutil
if os.path.exists(args.outdir):
    shutil.rmtree(args.outdir)
    print('removing ', args.outdir)
    os.makedirs(f'{args.outdir}/0')
'''for d in train_dataloader:
    i+=1
    print(i)
    with torch.no_grad():
        data = torch.tensor(d["input_ids"]).cuda().unsqueeze(0)
        outs_big = llm.generate(, sampling_params) # bigmodel(data)#, output_hidden_states=True)
        del outs_big
        del data
        with torch.no_grad():
            #print(torch.cat((torch.tensor(d["input_ids"]).cuda().unsqueeze(0), torch.tensor(d["label"]).cuda().unsqueeze(0) ), dim=-1).shape)
            #data = torch.cat((torch.tensor(d["input_ids"]).cuda().unsqueeze(0), torch.tensor(d["label"]).cuda().unsqueeze(0) ), dim=-1)
            #loss_mask = [0]*len(d["input_ids"])+[1]*len(d["label"])
            data = torch.tensor(d["input_ids"]).cuda().unsqueeze(0)
            outs_big = bigmodel(data, output_hidden_states=True)
            hidden_state_big = outs_big.hidden_states[-1]
            td = {"input_ids": data.detach().cpu()[0], "target": d["label"],  "hidden_state": hidden_state_big.detach().cpu()[0]}
            writedata(outdir, td)
            del td
            del outs_big
            del hidden_state_big
            del data
            torch.cuda.empty_cache()
            gc.collect()'''
for id,data in tqdm(enumerate(ds)):
    if id%1==100:
        print(id,end="\t")
    if id % 1000 == 0:
        print("")
    try:
        with torch.no_grad():
            name = outdir = f'{args.outdir}/0'
            outdata = ge(data, LLM, tokenizer)
            current_length=len(os.listdir(name))
            idx=current_length-1
            data_point = torch.load(f'{name}/data_{idx}.ckpt')
            assert all((data_point['input_ids'].cpu() == outdata['input_ids'])[0])
            data_point['label'] = outdata['label']
            torch.save(data_point, f'{name}/data_{idx}.ckpt')
            del data_point
    except:
        print('error')
