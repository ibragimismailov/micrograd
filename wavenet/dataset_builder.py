import torch
import random

def build_train_valid_test(words, block_size, stoi, train_ratio, valid_ratio, test_ratio):
    random.seed(42)
    random.shuffle(words)
    n1 = int(train_ratio*len(words))
    n2 = int((train_ratio + valid_ratio)*len(words))

    X_train, Y_train = build_dataset(words[:n1], block_size, stoi)
    X_valid, Y_valid = build_dataset(words[n1:n2], block_size, stoi)
    X_test, Y_test = build_dataset(words[n2:], block_size, stoi)

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test

def build_dataset(words, block_size, stoi):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix] # crop and append

    return torch.tensor(X), torch.tensor(Y)