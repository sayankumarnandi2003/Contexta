import torch
import os
import re

# Hyperparameters
device = 'cpu'
block_size = 128
batch_size = 64

# We'll initialize these when load_data() is called
tokens_list = []
vocab_size = 0
stoi = {}
itos = {}
train_data = None
val_data = None

def clean_text(text):
    replacements = {
        '’': "'",
        '‘': "'",
        '“': '"',
        '”': '"',
        '—': '-',
        '•': '*',
        '™': 'TM',
        'æ': 'ae',
        'á': 'a',
        'â': 'a',
        'é': 'e',
        'ê': 'e',
        'ô': 'o',
        '£': 'pound',
        '§': 'section',
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

def tokenize(text):
    return re.findall(r'[A-Za-z]+|\d+|[^\w\s]|\s+', text)

def encode(s):
    if isinstance(s, str):
        s = tokenize(s)
    return [stoi[t] for t in s if t in stoi]

def decode(l):
    return ''.join([itos[i] for i in l])

def load_data(file_path='input.txt'):
    global tokens_list, vocab_size, stoi, itos, train_data, val_data
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    text = clean_text(text)
    
    tokens = tokenize(text)
    tokens_list = sorted(list(set(tokens)))
    vocab_size = len(tokens_list)
    stoi = {t: i for i, t in enumerate(tokens_list)}
    itos = {i: t for i, t in enumerate(tokens_list)}

    data = torch.tensor(encode(tokens), dtype=torch.long)
    n = int(0.9 * len(data))  # first 90% train, rest val
    train_data = data[:n]
    val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y