from vllm import LLM, SamplingParams
import torch
import os

def list_files(path):
    datapath = []
    for root, directories, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            datapath.append(file_path)
    return datapath

datapath = list_files('/mnt/efs/people/pramudi/Eagle/data_aug_6')
print(f"Found {len(datapath)} files")

# Create dataset and dataloader
traindatapath = datapath[:int(len(datapath) * 0.95)]
testdatapath = datapath[ int(len(datapath) * 0.95):int(len(datapath) * 0.95)+500]
data = torch.load(testdatapath[0], weights_only=False)
prompts = torch.tensor(data['input_ids'][...,:2048][None, :]).cuda()
#prompts = ['write a python prog']
print('PROMPT')
EOS = [
        "<|endoftext|>",
        "<|endofmask|>",
        "<|/after_change|><|endoftext|>",
        '<|/after_change|>\n```\n<|endoftext|>',
        '<|/after_change|>\n```<|endoftext|>',
        '<|/after_change|>\n```<|im_end|>',
        "<|extra_0|>",
    ]
sampling_params = SamplingParams(n=1,
        temperature=0,
        top_p=1,
        repetition_penalty=1.0,
        max_tokens=8192,
        include_stop_str_in_output=True,
        skip_special_tokens=False,
        logprobs = 1,
        stop = EOS,
    )
llm = LLM(
    model='/mnt/efs/people/yawenwuu/projects/Qwen2.5-Coder/finetuning/sft/checkpoints_efs/7B_hunks/lr5e-5-wr100-wd0.0-bsz512-maxlen8192/run67/int8',#'EAGLE-Qwen2-7B-Instruct-run67-q28kv4_30k_n1_0820_multitoken',#'/mnt/efs/people/yawenwuu/projects/Qwen2.5-Coder/finetuning/sft/checkpoints_efs/7B_hunks/lr5e-5-wr100-wd0.0-bsz512-maxlen8192/run67/int8',#'/mnt/efs/people/yawenwuu/sft/checkpoints/7B_hunks/lr5e-5-wr100-wd0.0-bsz256-maxlen3270/run13',#'Qwen/Qwen2-7B-Instruct',#'lmsys/vicuna-7b-v1.3',# '/mnt/efs/people/yawenwuu/sft/checkpoints/7B_hunks/lr5e-5-wr100-wd0.0-bsz256-maxlen3270/run13',#'lmsys/vicuna-7b-v1.3',#"meta-llama/Meta-Llama-3-8B-Instruct",
    tensor_parallel_size=1, enforce_eager=True,
    speculative_config= {
        "method": "eagle",
    "model": 'EAGLE-Qwen2-7B-Instruct-run67-q28kv4_30k_n1_0820_multitoken',#'EAGLE3-Qwen2-7B-Instruct-run67-abation',# 'EAGLE-Qwen2-7B-Instruct-run67-int8-q28kv4_30k_n2',#'EAGLE-Qwen2-7B-Instruct-run67-int8-aug16',#'EAGLE-Qwen2-7B-Instruct',#'/mnt/efs/people/pramudi/EAGLE-Qwen2-7B-Instruct',#'yuhuili/EAGLE-Qwen2-7B-Instruct',#'/mnt/efs/people/pramudi/draft3',#'yuhuili/EAGLE-Vicuna-7B-v1.3',# 'state_0',#'yuhuili/EAGLE-Vicuna-7B-v1.3',#"yuhuili/EAGLE-LLaMA3-Instruct-8B",
     "draft_tensor_parallel_size": 1,
    "num_speculative_tokens": 8,
    }, disable_log_stats=False
)

import time
time.sleep(5)  # Wait to ensure stats are printed

#prompts = 'write a hellp world program'
outputs =  llm.generate(prompt_token_ids=prompts.cpu().numpy().tolist()[0],  sampling_params=sampling_params)
for metric in llm.get_metrics():
    if metric.name == 'vllm:spec_decode_num_accepted_tokens_per_pos':
        print(metric.values)
        print('AR')
        print( sum([(i+1)*v for i,v in enumerate(metric.values)])/sum(metric.values ))
#outputs = llm.generate(prompt_token_ids=prompts[0], sampling_params=sampling_params)
#outputs = llm.generate(prompts, sampling_params=sampling_params)
#a = torch.load('data_aug_6/0/data_999.ckpt', weights_only=False)
#outputs =llm.generate( prompt_token_ids=list(a['input_ids'].tolist()[0]))
#print(outputs)
'''for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    import IPython
    IPython.embed()
    print(f"{generated_text!r}")'''
