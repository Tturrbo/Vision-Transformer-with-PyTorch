import torch
import torch.nn as nn
from vit import ViT
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary
from datasets import load_dataset
from torchvision import transforms

torch.manual_seed(42)
data = load_dataset("timm/oxford-iiit-pet")

img_process = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def vit_collate_fn(batch):
    images = [img_process(example["image"].convert("RGB")) for example in batch]
    labels = [example["label"] for example in batch]
    return {"image": torch.stack(images), 
            "labels": torch.tensor(labels)}

def train(model, optimizer, loss_func, train_loader, n_epochs):
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.
        for batch in train_loader:
            inputs = batch["image"]
            labels = batch["labels"]
            logits = vit_model(inputs)
            loss = loss_func(logits, labels)
            print(loss)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            optimizer.step()
            optimizer.zero_grad()
        mean_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {mean_loss:.4f}")


def evaluate(model, loss_func, test_loader):
    model.eval()
    total_loss = 0.
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["image"]
            labels = batch["labels"]
            logits = vit_model(inputs)
            loss = loss_func(logits, labels)
            print(loss)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total +=labels.size(0)
    avg_loss = total_loss / len(test_loader)
    accuracy = 100* correct / total
    return print(f"Average loss: {avg_loss}| Accuracy: {accuracy}")

train_loader = DataLoader(data["train"], batch_size=4, collate_fn=vit_collate_fn)
test_loader = DataLoader(data["test"], batch_size=4, collate_fn=vit_collate_fn)

vit_model = ViT(image_size=224, patch_size=16, in_channels=3, num_classes=1000, embed_dim=768,
 depth=12, num_heads=12, ff_dim=3072, dropout=0.1)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.AdamW(vit_model.parameters(), lr=1e-5)
n_epochs = 3
train(vit_model, optimizer, loss_func, train_loader, n_epochs)
#summary(vit_model, input_size=batch.shape)
evaluate(vit_model, loss_func, test_loader)



