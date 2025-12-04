from datasets import load_dataset, concatenate_datasets, Dataset
import json
for i in ['python','java','typescript','javascript']:
    out = []
    data = (load_dataset("json", data_files=f"/mnt/efs/people/eyuyu/projects/genie/project_context_ablation/v13_project_context_data/train_data/{i}/combined.jsonl"))
    for id, d in enumerate(data['train']):
        instruct = f'<|fim_prefix|>{d["left_context"]}<|fim_suffix|>{d["right_context"]}<|fim_middle|>' # d["left_context"]
        out.append({"id": id, "conversations": [{"from": "human", "value": instruct}, {"from": "gpt", "value": ""}]})
        if id == 50:
            with open(f'fim_data/test/{i}/data.json', 'w') as f:
                json.dump(out, f)
            out = []
        if id == 15000:
            with open(f'fim_data/train/{i}/data.json', 'w') as f:
                json.dump(out, f)
            break
