from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import vllm
import json
import time
import numpy as np
# Prepare your prompts
benchmark_prompts = []
model_path = '/mnt/efs/people/yawenwuu/sft/checkpoints/7B_hunks/lr5e-5-wr100-wd0.0-bsz256-maxlen3270/run13'
#model_path = '/mnt/efs/people/pramudi/editpred/W8A8'
with open('/mnt/efs/people/pramudi/edit_pred_test_segments-4K.jsonl', "r") as f:
      for line in f.readlines():
          benchmark_prompts.append(json.loads(line.strip()))

# Initialize the tokenizer

tokenizer = AutoTokenizer.from_pretrained(model_path)

# Pass the default decoding hyperparameters of Qwen2.5-7B-Instruct
# max_tokens is for the maximum length for generation.

sampling_params = SamplingParams(
        temperature=0.0, 
        top_p=1.0, 
        top_k=1,
        repetition_penalty=1.0, 
        max_tokens=2048,
        include_stop_str_in_output=True,
        skip_special_tokens=False,
    )

# Input the model name or path. Can be GPTQ or AWQ models.
llm = LLM(
        model=model_path,
        tokenizer=model_path,
        skip_tokenizer_init=False,
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
        #speculative_config={"method": "ngram", "num_speculative_tokens":64, "prompt_lookup_max":8, "enable_DFS":True},
        #speculative_config={
        #"model": "/mnt/efs/people/pramudi/editpred/qwen7b_draft_330",
        #"num_speculative_tokens": 16,
        #},
        speculative_config={
            "method": 'Eagle',
            "model": '/mnt/efs/people/pramudi/Eagle/checkpoints/model_6',
            "num_speculative_tokens": 16,
            "draft_tensor_parallel_size": 1,
            "max_model_len": 4096,
        } ,
        max_seq_len_to_capture=35000,
        #ispeculative_model="/mnt/efs/people/pramudi/editpred/qwen7b_draft_330",
        #num_speculative_tokens=,
        #speculative_model="[ngram]",
        #num_speculative_tokens=16,
        #ngram_prompt_lookup_max=7,
    )

print('inference')
import IPython
IPython.embed()
