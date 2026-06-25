import torch
import torch.nn.functional as F


def train(words, stoi):
    # create a training set
    xs, ys = [], []

    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            xs.append(ix1)
            ys.append(ix2)

    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    num = xs.nelement()
    print('number of examples', num)

    # randomly initialize W matrix
    g = torch.Generator().manual_seed(2147483647)
    W = torch.randn((27, 27), generator=g, requires_grad=True)

    iterations = 100
    reg_ratio = 0.01
    learning_rate = 50

    for run in range(iterations):
        # forward pass
        xenc = F.one_hot(xs, num_classes=27).float() # input to the network - one_hot encoded
        logits = xenc @ W # predicted log-counts
        # next 2 lines is a 'softmax'
        counts = logits.exp() # counts - quivalent to N
        probs = counts / counts.sum(1, keepdim=True) # probabilities for the next charachter
        loss = -probs[torch.arange(num), ys].log().mean() + reg_ratio * (W**2).mean()
        print('loss', loss.item())

        # backward pass
        W.grad = None # set 0 to gradient
        loss.backward()
        
        # update
        W.data += -learning_rate * W.grad

    return W

def run(n, W, itos):
    # sampling from the trained neural net
    g = torch.Generator().manual_seed(2147483647)

    for i in range(n):

        out = []
        ix = 0
        while True:
            xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
            logits = xenc @ W # predict log-counts
            # next 2 lines is a 'softmax'
            counts = logits.exp() # counts - quivalent to N
            p = counts / counts.sum(1, keepdim=True) # probabilities for the next charachter
            
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
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

    W = train(words, stoi)
    run(25, W, itos)


if __name__ == "__main__":
    main()