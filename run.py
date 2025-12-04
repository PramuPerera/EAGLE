import torch
from transformers import AutoTokenizer, AutoModel
import os
from torch.utils.data import DataLoader
def extract_features(text, model, tokenizer, device):
    # Tokenize the input text
    #inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    #inputs = {k: v.to(device) for k, v in inputs.items()}

    # Extract features
    with torch.no_grad():
        outputs = model(text.cuda(), output_hidden_states=True)

    # Get the last hidden state
    last_hidden_state = outputs.last_hidden_state

    # Get all hidden states
    all_hidden_states = outputs.hidden_states

    return last_hidden_state, all_hidden_states

def writedata(name,data_point):
    if not os.path.exists(name):
        os.makedirs(name)
    current_length=len(os.listdir(name))
    idx=current_length
    torch.save(data_point, f'{name}/data_{idx}.ckpt')

def main():
    import gc
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = f'out-data/0'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # Load the Qwen-2B model and tokenizer
    model_name = "/mnt/efs/people/yawenwuu/sft/checkpoints/7B_hunks/lr5e-5-wr100-wd0.0-bsz512-maxlen8192/run21/"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    from datasets import load_dataset, concatenate_datasets, Dataset
    #ds = Dataset.from_file('train.hf')
    ds = concatenate_datasets([Dataset.from_file(f'/mnt/efs/people/pramudi/Eagle/data/data-000{"{:02d}".format(i)}-of-00025.arrow') for i in range(0,1)])
    ds = [d['input_ids'] for d in ds]
    train_dataloader = DataLoader(ds, batch_size=1)
    it = 0
    for d in train_dataloader:
        print(it)
        it+=1
        text =d# torch.tensor(d["input_ids"]).unsqueeze(0)
        # Extract features
        with torch.no_grad():
            last_hidden_state, all_hidden_states = extract_features(text, model, tokenizer, device)
        #td = {"input_ids": text.detach().cpu()[0],"label":d["label"], "hidden_state": last_hidden_state.detach().cpu()[0]}
        #writedata(outdir, td)
        del last_hidden_state
        del all_hidden_states
        del text
        #del td
        torch.cuda.empty_cache()
        gc.collect()

    # You can further process or analyze these features as needed

if __name__ == "__main__":
    main()
