from vllm import LLM, SamplingParams
import torch
prompts = [
''''<|im_start|>system\nYou are an intelligent programming assistant to make code edits based on the recent changes made by the user.<|im_end|>\n<|im_start|>user\nYour task is to predict the edit span and the new content for the edit. The input code is augmented with special tokens to indicate file name, cursor location, and recent changes. Follow the instructions below to make your code edit prediction.\nNotes of special tokens:\n1. <|current_edit|> is the focused file to edit.\n2. <|code_to_edit|> is the user's cursor location. It is the starting location for you to make the edit for.\n3. <|edit_begin|> is at the end of the file and is the start of your prediction.\n4. <|before_change|><|/before_change|> and <|after_change|></after_change> marks the recent changes made by the user. <|before_change|> marks the content before the change, and <|after_change|> marks the content after the change.\n5. The input ends with <|edit_begin|><|before_change|>, following which you make span prediction ending with <|/before_change|> and new content prediction wrapping with <|after_change|><|/after_change|> for the location specified by <|code_to_edit|>.\n\nCode for edit span and new content prediction:\n\n```\n<|repo_name|>untitled\n<|current_edit|><|file_sep|>file:///Volumes/workplace/q_toy/nep_test_2.py\n# Calculator\nclass Calculator:\n    def __init__(self):\n        \"\"\"\"\"\"\n        pass\n\n    def add(self, num1, num2):\n<|before_change|><|/before_change|><|after_change|>        \"\"\"Add Function\"\"\"\n<|/after_change|>        print(f\"Adding {num1} and {num2}\")\n        return num1 + num2\n\n    def subtract(self, num1, num2):\n<|before_change|><|/before_change|><|after_change|>        \"\"\"Subtraction Function\"\"\"\n<|/after_change|>        print(f\"Subtracting {num2} from {num1}\")\n        return num1 - num2\n\n    def multiply(self, num1, num2):\n<|before_change|><|/before_change|><|after_change|>        \"\"\"Multiplication Function\"\"\"\n<|/after_change|><|code_to_edit|>        print(f\"Multiplying {num1} and {num2}\")\n        return num1 * num2\n\n    def divide(self, num1, num2):\n        if num2 == 0:\n            raise ValueError(\"Cannot divide by zero\")\n        print(f\"Dividing {num1} by {num2}\")\n        return num1 / num2\n    \n<|edit_begin|><|before_change|>\n```\n<|im_end|>\n<|im_start|>assistant\n\nThe edit span and new content prediction:\n\n```\n''']
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
        stop = EOS
    )
llm = LLM(
    model='/mnt/efs/people/yawenwuu/projects/Qwen2.5-Coder/finetuning/sft/checkpoints_efs/7B_hunks/lr5e-5-wr100-wd0.0-bsz512-maxlen8192/run67/int8',#'/mnt/efs/people/yawenwuu/sft/checkpoints/7B_hunks/lr5e-5-wr100-wd0.0-bsz256-maxlen3270/run13',#'Qwen/Qwen2-7B-Instruct',#'lmsys/vicuna-7b-v1.3',# '/mnt/efs/people/yawenwuu/sft/checkpoints/7B_hunks/lr5e-5-wr100-wd0.0-bsz256-maxlen3270/run13',#'lmsys/vicuna-7b-v1.3',#"meta-llama/Meta-Llama-3-8B-Instruct",
    tensor_parallel_size=1, enforce_eager=True,
    speculative_config= None,#{
    #"model":  'EAGLE-Qwen2-7B-Instruct-run67-int8-aug13-llama-mask-annheal',#'EAGLE-Qwen2-7B-Instruct',#'/mnt/efs/people/pramudi/EAGLE-Qwen2-7B-Instruct',#'yuhuili/EAGLE-Qwen2-7B-Instruct',#'/mnt/efs/people/pramudi/draft3',#'yuhuili/EAGLE-Vicuna-7B-v1.3',# 'state_0',#'yuhuili/EAGLE-Vicuna-7B-v1.3',#"yuhuili/EAGLE-LLaMA3-Instruct-8B",
    # "draft_tensor_parallel_size": 1,
    #"num_speculative_tokens": 8,
    #}
)
#prompts = 'write a hellp world program'
#outputs = llm.generate(prompts, sampling_params)
#print(outputs)
a = torch.load('data_aug_6/0/data_999.ckpt', weights_only=False)
outputs =llm.generate( prompt_token_ids=list(a['input_ids'].tolist()[0]))
print(outputs)
'''for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    import IPython
    IPython.embed()
    print(f"{generated_text!r}")'''
