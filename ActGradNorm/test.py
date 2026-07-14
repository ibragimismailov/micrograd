import torch
import torch.nn.functional as F

from layers import Linear, BatchNorm1d, Tanh
from dataset_builder import build_train_valid_test


@torch.no_grad() # to let know pytorch that this code requires no gradients calculations
def split_loss(split, datasets, C, layers):
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = datasets
    x, y = {
        'train': (X_train, Y_train),
        'valid': (X_valid, Y_valid),
        'test': (X_test, Y_test),
    }[split]

    # forward pass
    emb = C[x]
    x = emb.view(emb.shape[0], -1)
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, y)

    print(split, loss.item())

def sample(count, block_size, layers, C, itos):
    for _ in range(count):

        out = []
        context = [0] * block_size # initialize with all ...
        while True:
            # forward pass
            emb = C[torch.tensor([context])] # (1,block_size,n_embed)
            x = emb.view(emb.shape[0], -1) # concatinate the vectors
            for layer in layers:
                x = layer(x)
            logits = x
            probs = F.softmax(logits, dim=1)
            # sample from the distribution
            ix = torch.multinomial(probs, num_samples=1).item()
            # shift the context window and track the samples
            context = context[1:] + [ix]
            out.append(ix)
            # break if we sampled the end-of-word character
            if ix == 0:
                break
            
        print(''.join(itos[i] for i in out))


def main():
    words = open('names.txt', 'r').read().splitlines()

    # build a vocabulary of characters and mappings to/from integers
    chars = sorted(list(set(''.join(words))))
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i:s for s,i in stoi.items()}
    vocab_size = len(itos)

    block_size = 3 # context length - how many charachters do we take to predict the next one
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = build_train_valid_test(words, block_size, stoi, 0.8, 0.1, 0.1)

    n_embed = 10 # the dimensionality of the character embedding vectors
    n_hidden = 100 # the number of neurons in the hidden layer of the MLP
    g = torch.Generator().manual_seed(2147483647) # for reproducibility

    C = torch.randn((vocab_size, n_embed), generator=g)
    layers = [
        Linear(n_embed * block_size, n_hidden,   generator=g), BatchNorm1d(n_hidden), Tanh(),
        Linear(n_hidden,             n_hidden,   generator=g), BatchNorm1d(n_hidden), Tanh(),
        Linear(n_hidden,             n_hidden,   generator=g), BatchNorm1d(n_hidden), Tanh(),
        Linear(n_hidden,             n_hidden,   generator=g), BatchNorm1d(n_hidden), Tanh(),
        Linear(n_hidden,             n_hidden,   generator=g), BatchNorm1d(n_hidden), Tanh(),
        Linear(n_hidden,             n_hidden,   generator=g), BatchNorm1d(n_hidden), Tanh(),
        Linear(n_hidden,             vocab_size, generator=g), BatchNorm1d(vocab_size), 
    ]

    with torch.no_grad():
        # last layer: make less confident
        layers[-1].gamma *= 0.1

    parameters = [C] + [p for layer in layers for p in layer.parameters()]
    print('number of parameters in the model:', sum(p.nelement() for p in parameters))
    for p in parameters:
        p.requires_grad = True

    batch_size = 64
    iterations = 25000

    lrs = [0.03, 0.01, 0.003, 0.001]
    epochs = len(lrs)
    iterations_per_epoch = iterations // epochs

    losses = []
    ud = []

    for epoch in range(epochs):
        learning_rate = lrs[epoch]

        for iter in range(iterations_per_epoch):

            # minibatch construct
            ix = torch.randint(0, X_train.shape[0], (batch_size,), generator=g)
            Xb, Yb = X_train[ix], Y_train[ix]

            # forward pass
            emb = C[Xb]
            x = emb.view(emb.shape[0], -1)
            for layer in layers:
                x = layer(x)
            loss = F.cross_entropy(x, Yb) # loss function

            # backward pass
            for p in parameters:
                p.grad = None
            loss.backward()

            # update
            for p in parameters:
                p.data += -learning_rate * p.grad

            #track stats
            step = iter + epoch * iterations_per_epoch
            if step % (iterations // 20) == 0:
                print(f'{step:7d}/{iterations:7d}: {loss.item():.4f} lr={learning_rate}')
            losses.append(loss.log10().item())

            with torch.no_grad():
                ud.append([(learning_rate*p.grad.std() / p.data.std()).log10().item() for p in parameters])

    # put layers into eval mode (needed for batchnorm espessially)
    for layer in layers:
        layer.training = False

    split_loss('train', (X_train, Y_train, X_valid, Y_valid, X_test, Y_test), C, layers)
    split_loss('valid', (X_train, Y_train, X_valid, Y_valid, X_test, Y_test), C, layers)
    split_loss('test',  (X_train, Y_train, X_valid, Y_valid, X_test, Y_test), C, layers)

    sample(25, block_size, layers, C, itos)

if __name__ == "__main__":
    main()