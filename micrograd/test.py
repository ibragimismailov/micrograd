from engine import Value
from nn import Neuron, Layer, MLP

def main():
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0] # desired targets

    n = MLP(3, [4, 4, 1])

    for i in range(100):
    
        ypred = [n(x) for x in xs]
        loss = sum([(yout-ygt)**2 for ygt, yout in zip(ys, ypred)]) / len(ys)
        print('loss', loss.data)

        n.zero_grad()
        loss.backward()

        alpha = 0.05
        for p in n.parameters():
            p.data += -alpha * p.grad
        
        print([y.data for y in ypred])


if __name__ == "__main__":
    main()