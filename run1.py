import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets, Dataset
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(text, 
                                  return_tensors='pt', 
                                  padding='max_length', 
                                  truncation=True, 
                                  max_length=self.max_length)
        return {key: val.squeeze(0) for key, val in encoding.items()}

def extract_features(model, dataloader, device):
    features = []
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            batch = {k: torch.tensor(v).unsqueeze(0).to(device) for k, v in batch.items()}
            #batch = {'input_ids': torch.tensor(batch['input_ids']).unsqueeze(0).to(device)}
            outputs = model(**batch, output_hidden_states=True)
            
            # Get the last hidden state
            last_hidden_state = outputs.last_hidden_state
            
            # You can choose to use the last hidden state or any other representation
            # Here, we're using mean pooling over the last hidden state
            features.append(last_hidden_state.mean(dim=1))
    
    return torch.cat(features, dim=0)

def main():
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the Qwen-2B model and tokenizer
    model_name = "/mnt/efs/people/yawenwuu/sft/checkpoints/7B_hunks/lr5e-5-wr100-wd0.0-bsz256-maxlen3270/run13"#"Qwen/Qwen-2B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    # Example texts
    texts = [
        "Hello, how are you today?",
        "The weather is nice.",
        "I love programming in Python!",
        # Add more texts as needed
    ]*100

    # Create dataset and dataloader
    #dataset = TextDataset(texts, tokenizer)
    #dataset = concatenate_datasets([Dataset.from_file(f'data/data-00000-of-000{"{:02d}".format(i)}.arrow') for i in range(1,15)])
    dataset = concatenate_datasets([Dataset.from_file(f'/mnt/efs/people/pramudi/Eagle/data/data-000{"{:02d}".format(i)}-of-00025.arrow') for i in range(0,1)])
    dataset = [{'input_ids':torch.tensor(i['input_ids'][:4096])} for i in dataset]
    #dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Extract features
    features = extract_features(model, dataset, device)

    # Print the shape of the extracted features
    print(f"Shape of extracted features: {features.shape}")

    # You can further process or analyze these features as needed

if __name__ == "__main__":
    main()
