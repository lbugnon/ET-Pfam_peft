"""Load a pretrained model and finetune it with LoRa"""

import argparse
import json 

args = argparse.ArgumentParser(description="Finetune a pretrained model with LoRa")
args.add_argument("model_id", type=str, help="Pretrained model identifier")
args.add_argument("--device", type=str, default="cuda:0", help="Device to use for training")

args = args.parse_args()

model_path = f"models/mini/model{args.model_id}/"
config = json.load(open(f"{model_path}/config.json"))
config.update({
    "use_lora": True,
    "lr_lora": 1e-4, 
    "lr_cnn": 1e-5,
    "lr_fc": 1e-5,
    "batch_size": 2,
    "use_embeddings": False,
    "train_fraction": 1.0
})
json.dump(config, open(f"config/config_finetuned_{args.model_id}.json", "w"), indent=4)

env = json.load(open("config/env.json"))
env["device"] = args.device
json.dump(env, open(f"{model_path}/env.json", "w"), indent=4)

# calling training 
import subprocess
subprocess.run(["python", "train_basemodel_lora.py", 
    "-p", model_path+"weights.pk",
    "-c", f"config/config_finetuned_{args.model_id}.json",
    "-o", f"output_finetuned_lora_mini_{args.model_id}/",
])