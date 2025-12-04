# vllm serve <model-name> --speculative-config '("method": "eagle", "model": "<draft-model>", "num_speculative_tokens": 4)'
# curl http://localhost:8000/metrics
# vllm:spec_decode_num_accepted_tokens_total{model_name="your-model"} 12345
# vllm:spec_decode_num_draft_tokens_total{model_name="your-model"} 23456
# acceptance_rate = accepted_tokens / draft_tokens
# CUDA_VISIBLE_DEVICES=0  python3 -m eagle.train.main_lightning_eagle3 --tmpdir data_aug_6 
# --cpdir checkpoints_run67int8_sep6_vllm_feats_llama3_cont 
# --configpath llama_config_05B.json 
# --basepath /mnt/efs/people/yawenwuu/projects/Qwen2.5-Coder/finetuning/sft/checkpoints_efs/7B_hunks/lr5e-5-wr100-wd0.0-bsz512-maxlen8192/run67/int8 
# --load EAGLE3-Qwen2-7B-Instruct-run67-sep9/pytorch_model.bin


# vllm serve /mnt/efs/people/yawenwuu/projects/Qwen2.5-Coder/finetuning/sft/checkpoints_efs/7B_hunks/lr5e-5-wr100-wd0.0-bsz512-maxlen8192/run67/int8 --speculative-config '("method": "eagle3", "model": "/mnt/efs/people/pramudi/Eagle/EAGLE3-Qwen2-7B-Instruct-run67-sep9", "num_speculative_tokens": 4)'

from openai import OpenAI
client = OpenAI(
    api_key="EMPTY",  # or your API key if set
    base_url="http://localhost:8000/v1"
)
response = client.completions.create(
    model="<your-model-name>",
    prompt="Your custom prompt here.",
    max_tokens=32
)
print(response.choices[0].text)