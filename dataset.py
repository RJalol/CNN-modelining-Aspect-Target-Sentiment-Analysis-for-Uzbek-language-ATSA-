import json
import torch
from torch.utils.data import Dataset
from utils import tokenize_and_pad, label_to_id

class UzABSA_Dataset(Dataset):
    def __init__(self, path, tokenizer, vocab, max_len=100):
        with open(path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f if line.strip()]

        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_len = max_len
        self.samples = self._process()

    def _process(self):
        samples = []
        for item in self.data:
            text = item["text"]
            for asp in item["aspect_terms"]:
                term = asp["term"]
                polarity = asp["polarity"]
                new_text = f"{text} [ASP] {term}"
                input_ids = tokenize_and_pad(new_text, self.vocab, self.tokenizer, self.max_len)
                label = label_to_id(polarity)
                samples.append((input_ids, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_ids, label = self.samples[idx]
        return torch.tensor(input_ids), torch.tensor(label)
