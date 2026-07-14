import torch
import torch.nn.functional as F
import random
import math


def train(X_train, Y_train, block_size, embedding_dim, W1, b1, W2, b2, C, parameters, g):
    iterations = 10000
    batch_size = 64

    lrs = [0.3, 0.1, 0.1, 0.03, 0.03, 0.01, 0.01]
    epochs = len(lrs)

    steps, losses = [], []
    lri, lsi = [], []
    for epoch in range(epochs):
        learning_rate = lrs[epoch]

        for iter in range(iterations):

            # batch construction
            ix = torch.randint(0, X_train.shape[0], (batch_size,), generator=g)

            # forward pass
            emb = C[X_train[ix]]
            h = torch.tanh(emb.view(-1, block_size * embedding_dim) @ W1 + b1)
            logits = h @ W2 + b2
            loss = F.cross_entropy(logits, Y_train[ix])

            # backward pass
            for p in parameters:
                p.grad = None
            loss.backward()

            # update
            for p in parameters:
                p.data += -learning_rate * p.grad

            # to plot iteration vs loss
            steps.append(iter + epoch * iterations)
            losses.append(loss.log10().item())

            # to plot learning rate vs loss
            lri.append(math.log10(learning_rate))
            lsi.append(loss.log10().item())

        print('epoch', epoch, 'learning_rate', learning_rate, 'loss', loss.item())

    return W1, b1, W2, b2, C


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

        #print(w)
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix] # crop and append

    return torch.tensor(X), torch.tensor(Y)


def get_loss(X, Y, block_size, embedding_dim, W1, b1, W2, b2, C):
    emb = C[X]
    h = torch.tanh(emb.view(-1, block_size * embedding_dim) @ W1 + b1)
    logits = h @ W2 + b2
    return F.cross_entropy(logits, Y)


def sample(count, block_size, embedding_dim, W1, b1, W2, b2, C, itos, g):
    for i in range(count):

        out = []
        context = [0] * block_size
        
        while True:
            emb = C[torch.tensor([context])]
            h = torch.tanh(emb.view(1, block_size * embedding_dim) @ W1 + b1)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
            context = context[1:] + [ix]
            out.append(itos[ix])
            if ix == 0:
                break

        print(''.join(out))


def main():
    words = open('names.txt', 'r').read().splitlines()

    chars = sorted(list(set(''.join(words))))
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i:s for s,i in stoi.items()}

    block_size = 3 # context length - how many charachters do we take to predict the next one
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = build_train_valid_test(words, block_size, stoi, 0.8, 0.1, 0.1)

    hidden_layer_size = 200
    embedding_dim = 10
    g = torch.Generator().manual_seed(2147483647)
    C = torch.randn((27, embedding_dim), generator=g)
    W1 = torch.randn((block_size * embedding_dim, hidden_layer_size), generator=g)
    b1 = torch.randn(hidden_layer_size, generator=g)
    W2 = torch.randn((hidden_layer_size, 27), generator=g)
    b2 = torch.randn(27, generator=g)
    parameters = [C, W1, b1, W2, b2]

    print('parameters count', sum(p.nelement() for p in parameters))

    for p in parameters:
        p.requires_grad = True

    W1, b1, W2, b2, C = train(X_train, Y_train, block_size, embedding_dim, W1, b1, W2, b2, C, parameters, g)

    loss_valid = get_loss(X_valid, Y_valid, block_size, embedding_dim, W1, b1, W2, b2, C)    
    print('loss vslid', loss_valid.item())
    
    sample(25, block_size, embedding_dim, W1, b1, W2, b2, C, itos, g)


if __name__ == "__main__":
    main()