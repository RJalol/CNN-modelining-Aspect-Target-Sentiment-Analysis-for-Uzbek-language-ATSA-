import torch
from torch.utils.data import DataLoader
from cnn_model import CNN_Text
from dataset import UzABSA_Dataset
from utils import simple_tokenizer, build_vocab
import json  

# Dataset
train_path = "uzabsa_train.json"
tokenizer = simple_tokenizer

# Vocab
with open(train_path, 'r', encoding='utf-8') as f:
    raw_data = [json.loads(line) for line in f if line.strip()]
vocab = build_vocab(raw_data, tokenizer)

# Hyperparams
embed_dim = 100
num_classes = 4
batch_size = 32
epochs = 10
lr = 0.001

# Dataset
train_dataset = UzABSA_Dataset(train_path, tokenizer, vocab)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model
model = CNN_Text(vocab_size=len(vocab), embed_dim=embed_dim, class_num=num_classes)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss, correct = 0, 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        loss.backward()
        optimizer.step()
    acc = correct / len(train_dataset)
    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Accuracy: {acc:.4f}")
