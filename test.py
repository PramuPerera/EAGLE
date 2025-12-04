from eagle.model.ea_model import EaModel
from fastchat.model import get_conversation_template
import torch
model = EaModel.from_pretrained(
    use_eagle3 = False,
    base_model_path='/mnt/efs/people/yawenwuu/projects/Qwen2.5-Coder/finetuning/sft/checkpoints_efs/7B_hunks/lr5e-5-wr100-wd0.0-bsz512-maxlen8192/run67/int8',#'/mnt/efs/people/yawenwuu/sft/checkpoints/7B_hunks/lr5e-5-wr100-wd0.0-bsz256-maxlen3270/run13',
    ea_model_path='EAGLE-Qwen2-7B-Instruct-run67-int8-aug7-llama-converted',#'checkpoints_run67int8_aug3',#'checkpoints/Eagle',
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    total_token=-1
)
model.cuda()
model.eval()
your_message="Hello"
prompt = "# hello world program \n"
prompts = [
''''<|im_start|>system\nYou are an intelligent programming assistant to make code edits based on the recent changes made by the user.<|im_end|>\n<|im_start|>user\nYour task is to predict the edit span and the new content for the edit. The input code is augmented with special tokens to indicate file name, cursor location, and recent changes. Follow the instructions below to make your code edit prediction.\nNotes of special tokens:\n1. <|current_edit|> is the focused file to edit.\n2. <|code_to_edit|> is the user's cursor location. It is the starting location for you to make the edit for.\n3. <|edit_begin|> is at the end of the file and is the start of your prediction.\n4. <|before_change|><|/before_change|> and <|after_change|></after_change> marks the recent changes made by the user. <|before_change|> marks the content before the change, and <|after_change|> marks the content after the change.\n5. The input ends with <|edit_begin|><|before_change|>, following which you make span prediction ending with <|/before_change|> and new content prediction wrapping with <|after_change|><|/after_change|> for the location specified by <|code_to_edit|>.\n\nCode for edit span and new content prediction:\n\n```\n<|repo_name|>untitled\n<|current_edit|><|file_sep|>file:///Volumes/workplace/q_toy/nep_test_2.py\n# Calculator\nclass Calculator:\n    def __init__(self):\n        \"\"\"\"\"\"\n        pass\n\n    def add(self, num1, num2):\n<|before_change|><|/before_change|><|after_change|>        \"\"\"Add Function\"\"\"\n<|/after_change|>        print(f\"Adding {num1} and {num2}\")\n        return num1 + num2\n\n    def subtract(self, num1, num2):\n<|before_change|><|/before_change|><|after_change|>        \"\"\"Subtraction Function\"\"\"\n<|/after_change|>        print(f\"Subtracting {num2} from {num1}\")\n        return num1 - num2\n\n    def multiply(self, num1, num2):\n<|before_change|><|/before_change|><|after_change|>        \"\"\"Multiplication Function\"\"\"\n<|/after_change|><|code_to_edit|>        print(f\"Multiplying {num1} and {num2}\")\n        return num1 * num2\n\n    def divide(self, num1, num2):\n        if num2 == 0:\n            raise ValueError(\"Cannot divide by zero\")\n        print(f\"Dividing {num1} by {num2}\")\n        return num1 / num2\n    \n<|edit_begin|><|before_change|>\n```\n<|im_end|>\n<|im_start|>assistant\n\nThe edit span and new content prediction:\n\n```\n''']
#conv = get_conversation_template("vicuna")
#conv.append_message(conv.roles[0], your_message)
#conv.append_message(conv.roles[1], None)
#prompt = conv.get_prompt()
input_ids=model.tokenizer.encode(prompts[0])
#input_ids=model.tokenizer([prompts]).input_ids
input_ids = torch.as_tensor([input_ids]).cuda()
output_ids=model.ea_generate(input_ids,temperature=0.5,max_new_tokens=512)
#output=model.tokenizer.decode(output_ids[0])
