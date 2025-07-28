from collections import Counter
import numpy as np

def simple_tokenizer(text):
    return text.lower().split()

def build_vocab(dataset, tokenizer, min_freq=1):
    counter = Counter()
    for item in dataset:
        tokens = tokenizer(item["text"])
        for asp in item["aspect_terms"]:
            tokens += tokenizer(asp["term"])
        counter.update(tokens)

    vocab = {"<pad>": 0, "<unk>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

def tokenize_and_pad(text, vocab, tokenizer, max_len):
    tokens = tokenizer(text)
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
    if len(ids) < max_len:
        ids += [vocab["<pad>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

def label_to_id(label):
    mapping = {"positive": 0, "neutral": 1, "negative": 2, "conflict": 3}
    return mapping[label]

def id_to_label(idx):
    mapping = ["positive", "neutral", "negative", "conflict"]
    return mapping[idx]
