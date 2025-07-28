import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Text(nn.Module):
    def __init__(self, vocab_size, embed_dim, class_num, kernel_sizes=[3,4,5], num_channels=100, dropout=0.5, pretrained_embeddings=None):
        super(CNN_Text, self).__init__()
        if pretrained_embeddings is not None:
            self.embed = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        else:
            self.embed = nn.Embedding(vocab_size, embed_dim)

        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_channels, (k, embed_dim)) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_channels * len(kernel_sizes), class_num)

    def forward(self, x):
        x = self.embed(x)  # [batch_size, seq_len, embed_dim]
        x = x.unsqueeze(1)  # [batch_size, 1, seq_len, embed_dim]
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(B, C, seq_len-kernel+1), ...]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
