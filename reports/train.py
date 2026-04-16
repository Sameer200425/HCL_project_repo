import yaml
import torch
import torch.nn as nn
import torch.optim as optim

# Example: Vision Transformer from timm
import timm

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    print("Loaded project:", config["project"]["name"])

    # Example: create a ViT model
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=config["vit"]["num_classes"])
    print(model)

    # Example optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])

    # Example loss
    criterion = nn.CrossEntropyLoss(label_smoothing=config["training"]["label_smoothing"])

    # TODO: load dataset from config["data"]["processed_dir"]
    # TODO: implement training loop

    print("Training loop would start here...")

if __name__ == "__main__":
    main()
