import argparse
import json
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from safetensors import safe_open
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup, AutoConfig
from typing import Any, Dict, List
import random

from ..model.cnets import Model
from ..model.configs import EConfig

torch.backends.cuda.matmul.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser(description='EAGLE Training with PyTorch Lightning')
    parser.add_argument('--basepath', type=str, default='/home/lyh/weights/hf/vicuna_v13/7B/')
    parser.add_argument('--configpath', type=str, default="config.json")
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=8)
    parser.add_argument('--tmpdir', type=str, default='0')
    parser.add_argument('--cpdir', type=str, default='0')
    parser.add_argument('--precision', type=str, default='bf16')
    parser.add_argument('--devices', type=int, default=-1)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--strategy', type=str, default='auto')
    parser.add_argument('--load', type=str, default=None)
    return parser.parse_args()


def get_train_config(args):
    return {
        "lr": args.lr,
        "bs": args.bs,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "datapath": f"{args.tmpdir}",
        "is_warmup": False,
        "num_epochs": 100,
        "num_warmup_steps": 2000,
        "total_steps": 100000,
        "p_w": 0.1,
        "v_w": 1.0,
        "head_w": 0.1,
        "num_workers": 0,  # Changed from 2 to 0 to avoid multiprocessing issues
        "embeding": True,
        "act": "No",
        "data_noise": False,
        "noise": "uniform",
        "mean": 0.0,
        "std": 0.2,
        "residual": "true,norm",
        "max_len": 4096,
        "config_path": args.configpath,
        "b1": 0.9,
        "b2": 0.95,
        "grad_clip": 0.5,
        "save_freq": 1
        }


def load_head(basepath):
    head = torch.nn.Linear(AutoConfig.from_pretrained(basepath).hidden_size, 
                          AutoConfig.from_pretrained(basepath).vocab_size, bias=False)

    try:
        with open(os.path.join(basepath, "model.safetensors.index.json"), "r") as f:
            index_json = json.loads(f.read())
            head_path = index_json["weight_map"]["lm_head.weight"]
        with safe_open(os.path.join(basepath, head_path),
                      framework="pt",
                      device="cpu") as f:
            tensor_slice = f.get_slice("lm_head.weight")
            vocab_size, hidden_dim = tensor_slice.get_shape()
            tensor = tensor_slice[:, :hidden_dim].float()
    except:
        with open(os.path.join(basepath, "pytorch_model.bin.index.json"), "r") as f:
            index_json = json.loads(f.read())
            head_path = index_json["weight_map"]["lm_head.weight"]
        tensor = torch.load(os.path.join(basepath, head_path))
        tensor = tensor["lm_head.weight"].float()

    head.weight.data = tensor
    head.eval()

    for param in head.parameters():
        param.requires_grad = False
    return head


def list_files(path):
    datapath = []
    for root, directories, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            datapath.append(file_path)
    return datapath


class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.0):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


class AddUniformNoise:
    def __init__(self, std=0.0):
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = (torch.rand_like(tensor) - 0.5) * self.std * 512 / tensor.shape[1]
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


class CustomDataset(Dataset):
    def __init__(self, datapath, max_len, transform=None):
        self.data = datapath
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            data = torch.load(self.data[index], weights_only=False)
            
            new_data = {}
            hidden_state = torch.tensor(data['hidden_state'][:self.max_len][None, :]).cpu()
            input_ids = torch.tensor(data['input_ids'][:self.max_len][None, :]).cpu()
            label = torch.tensor(data["label"][:self.max_len][None, :]).cpu()
            loss_mask = list(torch.where(torch.tensor(label.squeeze())!=-1000, 1, 0).numpy())
            #torch.Size([1, 4096, 3584]) torch.Size([1, 1, 7529]) torch.Size([1, 4096])
            
            input_ids = input_ids[0][:,:self.max_len]
            label = label[0]
            hidden_state= hidden_state[:,:,:3584]

            length = hidden_state.shape[1]
            attention_mask = [1] * length
            #loss_mask[-1] = 0
            input_ids_target = input_ids[:, 1:]
            zeropadding = torch.tensor([[0]])
            input_ids_target = torch.cat((input_ids_target, zeropadding), dim=1)

            target = hidden_state[:, 1:, :]
            zeropadding = torch.zeros(1, 1, target.shape[2])
            target = torch.cat((target, zeropadding), dim=1)
            loss_mask[-1] = 0
            new_data["attention_mask"] = attention_mask
            new_data["loss_mask"] = loss_mask
            new_data["target"] = hidden_state#target
            new_data["hidden_state_big"] = torch.roll(hidden_state, 1, 1) #torch.roll(hidden_state, 1, 0) #hidden_state
            new_data["input_ids"] = input_ids.to(torch.int64) #input_ids_target
            if self.transform:
                new_data = self.transform(new_data)

            return new_data
        except Exception as e:
            print(f"Error loading file {self.data[index]}: {e}")
            # Return a simple dummy sample as fallback
            hidden_dim = 3584  # Adjust based on your model
            dummy_data = {
                "attention_mask": [1],
                "loss_mask": [0],
                "target": torch.zeros(1, 1, hidden_dim),
                "hidden_state_big": torch.zeros(1, 1, hidden_dim),
                "input_ids": torch.zeros(1, 1, dtype=torch.long)
            }
            return dummy_data


class DataCollatorWithPadding:
    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        padding_tensor = torch.zeros(B, N - n, S, device=intensors.device)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype, device=intensors.device)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            if not features:
                return {}
                
            max_length = 4096#max(item['hidden_state_big'].shape[1] for item in features)
            batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
            batch_hidden_states = torch.cat([self.paddingtensor(item['hidden_state_big'], max_length) for item in features])
            batch_target = torch.cat([self.paddingtensor(item['target'], max_length) for item in features])
            batch_loss_mask = torch.tensor(
                [item['loss_mask'] + [0] * (max_length - len(item['loss_mask'])) for item in features])

            batch = {
                "input_ids": batch_input_ids,
                "hidden_states": batch_hidden_states,
                "target": batch_target,
                "loss_mask": batch_loss_mask,
            }
            return batch
        except Exception as e:
            print(f"Error in collator: {e}")
            # Return empty batch as fallback
            return {}


def top_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res


class EAGLEModule(pl.LightningModule):
    def __init__(self, config, train_config, head, path=None):
        super().__init__()
        self.model = Model(config, load_emb=True, path=path)
        self.head = head
        self.train_config = train_config
        self.criterion = nn.SmoothL1Loss(reduction="none") #nn.MSELoss(reduction='none')
        # Don't save hyperparameters to avoid serialization issues
        
    def forward(self, hidden_states, input_ids=None):
        return self.model(hidden_states, input_ids=input_ids)
    
    def compute_loss(self, target, target_p, predict, loss_mask):
        out_head = self.head(predict)
        out_logp = nn.LogSoftmax(dim=2)(out_head)
        plogp = target_p * out_logp
        ploss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / (loss_mask.sum() + 1e-5)
        vloss = self.criterion(predict, target)
        vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (loss_mask.sum() + 1e-5)
        return vloss, ploss, out_head
    
    def training_step(self, batch, batch_idx):
        if not batch:  # Handle empty batch case
            return None
            
        hidden_states = batch["hidden_states"]
        target = batch["target"]
        loss_mask = batch["loss_mask"].unsqueeze(-1)
        input_ids = batch["input_ids"]
        
        # Simple forward pass without input_ids
        max_steps = 5
        steps = random.randint(1,max_steps)
        with torch.no_grad():
            for i in range(steps):
                hidden_states = self(hidden_states, input_ids)
                hidden_states = torch.roll(hidden_states, 1, 1)

        predict = self(hidden_states, input_ids)
        with torch.no_grad():    
            target_head = self.head(target)
            target_p = nn.Softmax(dim=2)(target_head)
            target_p = target_p.detach()
        vloss, ploss, out_head = self.compute_loss(target, target_p, predict, loss_mask)
        loss = self.train_config["v_w"] * vloss + self.train_config["p_w"] * ploss


        


        with torch.no_grad():
            _, predicted = torch.max(out_head, 2)
            _, target = torch.max(target_head, 2)
            ct = loss_mask.sum().item()
            cc = ((predicted == target) * loss_mask.squeeze()).sum().item()
        # Log metrics - use on_step=True, on_epoch=False to avoid synchronization issues
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log('vloss', vloss, prog_bar=True, on_step=True, on_epoch=False)
        self.log('ploss', ploss, prog_bar=True, on_step=True, on_epoch=False)
        self.log('acc1', cc/(loss_mask.squeeze().sum().item()+0.00000001), prog_bar=True, on_step=True, on_epoch=False) 
        return loss
    
    def validation_step(self, batch, batch_idx):
        hidden_states = batch["hidden_states"]
        target = batch["target"]
        loss_mask = batch["loss_mask"].unsqueeze(-1)
        input_ids = batch["input_ids"]

        # Simple forward pass without input_ids
        with torch.no_grad():
            hidden_states = self(hidden_states, input_ids)
            hidden_states = torch.roll(hidden_states, 1, 1)
        predict = self(hidden_states, input_ids)
        target_head = self.head(target)
        target_p = nn.Softmax(dim=2)(target_head)
        target_p = target_p.detach()
        vloss, ploss, out_head = self.compute_loss(target, target_p, predict, loss_mask)
        loss = self.train_config["v_w"] * vloss + self.train_config["p_w"] * ploss

        with torch.no_grad():
            _, predicted = torch.max(out_head, 2)
            _, target = torch.max(target_head, 2)
            ct = loss_mask.sum().item()
            cc = ((predicted == target) * loss_mask.squeeze()).sum().item()
        self.log('valacc1', cc/(loss_mask.squeeze().sum().item()+0.00000001))


    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.train_config["lr"],
            betas=(self.train_config["b1"], self.train_config["b2"]),
            weight_decay=0.0,
        )
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.train_config["num_warmup_steps"],
            num_training_steps=self.train_config["total_steps"],
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


def main():
    args = parse_args()
    pl.seed_everything(0)
    
    train_config = get_train_config(args)
    
    # Load model head
    head = load_head(args.basepath)
    
    # Load config
    with open(train_config["config_path"], "r") as f:
        config = json.load(f)
    config = EConfig(**config)
    
    # Setup data
    datapath = list_files(train_config["datapath"])
    print(f"Found {len(datapath)} files")
    
    # Setup transform
    transform = None
    if train_config["data_noise"]:
        if train_config["noise"] == "gaussian":
            transform = AddGaussianNoise(mean=train_config["mean"], std=train_config["std"])
        elif train_config["noise"] == "uniform":
            transform = AddUniformNoise(std=train_config["std"])
    
    # Create dataset and dataloader
    traindatapath = datapath[:int(len(datapath) * 0.95)]
    testdatapath = datapath[ int(len(datapath) * 0.95):int(len(datapath) * 0.95)+500]
    traindataset = CustomDataset(traindatapath, train_config["max_len"], transform=transform)
    testdataset = CustomDataset(testdatapath, train_config["max_len"])
    data_collator = DataCollatorWithPadding()
    
    train_loader = DataLoader(
        traindataset,
        batch_size=train_config["bs"],
        shuffle=True,
        num_workers=train_config["num_workers"],  # Set to 0 to avoid multiprocessing issues
        collate_fn=data_collator,
        pin_memory=False,  # Changed to False to avoid CUDA issues
    )
    test_loader = DataLoader(
        testdataset,
        batch_size=train_config["bs"],
        shuffle=False,
        num_workers=train_config["num_workers"],  # Set to 0 to avoid multiprocessing issues
        collate_fn=data_collator,
        pin_memory=False,  # Changed to False to avoid CUDA issues
    )
    # Create model
    model = EAGLEModule(config, train_config, head, path=args.basepath)
    if args.load:
        a = torch.load(args.load)
        if 'state_dict' not in a.keys():
            a = {'state_dict':a}
            prefix= ''
        else:
            prefix = 'model.'
        model.model.embed_tokens.weight.data =  a['state_dict'][f'{prefix}embed_tokens.weight'].to(torch.bfloat16).cuda()
        model.model.layers[0].self_attn.q_proj.weight.data= a['state_dict'][f'{prefix}layers.0.self_attn.q_proj.weight'].to(torch.bfloat16).cuda()
        model.model.layers[0].self_attn.k_proj.weight.data= a['state_dict'][f'{prefix}layers.0.self_attn.k_proj.weight'].to(torch.bfloat16).cuda()
        model.model.layers[0].self_attn.v_proj.weight.data= a['state_dict'][f'{prefix}layers.0.self_attn.v_proj.weight'].to(torch.bfloat16).cuda()
        model.model.layers[0].self_attn.o_proj.weight.data= a['state_dict'][f'{prefix}layers.0.self_attn.o_proj.weight'].to(torch.bfloat16).cuda()
        model.model.layers[0].mlp.gate_proj.weight.data= a['state_dict'][f'{prefix}layers.0.mlp.gate_proj.weight'].to(torch.bfloat16).cuda()
        model.model.layers[0].mlp.up_proj.weight.data= a['state_dict'][f'{prefix}layers.0.mlp.up_proj.weight'].to(torch.bfloat16).cuda()
        model.model.layers[0].mlp.down_proj.weight.data=a['state_dict'][f'{prefix}layers.0.mlp.down_proj.weight'].to(torch.bfloat16).cuda()
        model.model.layers[0].post_attention_layernorm.weight.data= a['state_dict'][f'{prefix}layers.0.post_attention_layernorm.weight'].to(torch.bfloat16).cuda()
        model.model.fc.weight.data= a['state_dict'][f'{prefix}fc.weight'].to(torch.bfloat16).cuda()
        model.model.fc.bias.data= a['state_dict'][f'{prefix}fc.bias'].to(torch.bfloat16).cuda()
        del a
        torch.cuda.empty_cache()
    # Setup callbacksi
    callbacks = [
        ModelCheckpoint(
            dirpath=args.cpdir,
            filename='{epoch}-{step}',
            save_top_k=3,
            monitor='valacc1',
            mode='max',
            save_last=True,
            every_n_train_steps=100,  # Save every 100 steps instead of epochs
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    # Setup logger
    logger = None
    if os.environ.get('RANK', '0') == '0':
        logger = WandbLogger(project="ess", entity="pramudi", config=train_config)
    
    # Setup trainer with minimal options
    trainer = pl.Trainer(
        max_epochs=train_config["num_epochs"],
        precision=args.precision,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy="auto",  # Simplified strategy
        gradient_clip_val=train_config["grad_clip"],
        accumulate_grad_batches=train_config["gradient_accumulation_steps"],
        logger=logger,
        callbacks=callbacks,
        detect_anomaly=False,
        enable_checkpointing=True,
        enable_progress_bar=True,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )
    
    # Train model
    #trainer.validate(model, test_loader)
    trainer.fit(model, train_loader, test_loader)


if __name__ == "__main__":
    main()
