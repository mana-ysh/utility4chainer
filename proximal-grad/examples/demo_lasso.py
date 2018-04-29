
import chainer
from chainer import links as L, functions as F, optimizers
import numpy as np

# for import prox_optimizers
import sys
sys.path.append('../')

from prox_optimizers import L1ProximalOptimizer


class LinearRegressor(chainer.Chain):
    def __init__(self, in_dim, out_dim):
        super(LinearRegressor, self).__init__(
            W = L.Linear(in_dim, out_dim, nobias=True)
        )

    def __call__(self, X, Y):
        pred_Y = self.predict(X)
        loss = F.sum(F.squared_error(pred_Y, Y))
        return loss

    def predict(self, X):
        return self.W(X)


# generate synthetic data with sparse matrix
def generate_synthetic(in_dim, out_dim, num, threshold=0.5):
    X = np.random.randn(num, in_dim).astype(np.float32)
    true_W = np.random.randn(in_dim, out_dim).astype(np.float32)
    true_W[np.abs(true_W) < threshold] = 0
    Y = X.dot(true_W) + np.random.randn(num, out_dim)
    return X, Y.astype(np.float32), true_W


def main():
    in_dim = 200
    out_dim = 100
    num = 1000
    n_iter = 500
    strength = 0.1
    X, Y, true_W = generate_synthetic(in_dim, out_dim, num)

    m = LinearRegressor(in_dim, out_dim)
    prox_opt = L1ProximalOptimizer(optimizers.Adam(), 0.001)
    prox_opt.setup(m)

    for i in range(n_iter):
        print('{}th Iteration...'.format(i+1))
        m.cleargrads()
        loss = m(X, Y)
        print(loss.data.mean())
        loss.backward()
        prox_opt.update()

    print('leant parameter')
    print(m.W.W.data)

    print('gold parameter')
    print(true_W)









if __name__ == '__main__':
    main()
