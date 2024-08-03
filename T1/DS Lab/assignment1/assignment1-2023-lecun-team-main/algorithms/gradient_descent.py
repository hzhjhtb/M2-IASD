import numpy as np
import time

'''
The Gradient Descent algorithm for matrix factorization
'''

class GradientDescent:
    def __init__(self, R, test_ds=None, target_matrix_rank=1, lam=1e-3, mu=1e-3, learning_rate=1e-4):
        
        self.R = np.nan_to_num(R)  # shape: (m, n)
        self.test_ds = np.nan_to_num(test_ds)
        self.target_matrix_rank = target_matrix_rank  # k
        self.lam = lam
        self.mu = mu
        self.learning_rate = learning_rate

        #self.I = np.random.rand(R.shape[0], target_matrix_rank)  # shape: (m, k)
        #self.U = np.random.rand(target_matrix_rank, R.shape[1])  # shape: (k, n)
        self.I = np.sqrt(self.R.sum(1) / (self.R > 0).sum(1))[..., None]
        self.U = np.sqrt(self.R.sum(0) / (self.R > 0).sum(0))[..., None].T

        self.R_output = None

    def train(self, train_steps):
        start_time = time.time()
        for step in range(train_steps):
            # compute loss
            loss = self.compute_loss()

            # update I and U
            self.optimize()

            # compute test loss ONLY if test_ds is provided
            test_loss_str = ""
            if self.test_ds is not None:
                test_loss = self.compute_test_loss()
                test_loss_str = f", testing loss {test_loss}"
            # compute SPS, step per second
            SPS = int(step / (time.time() - start_time))

            print(f"step {step}: training loss {loss}{test_loss_str}, SPS: {SPS}")
        self.R_output = np.where(self.R > 0, self.R, self.I @ self.U)

    def compute_loss(self):
        # loss = ||R - I @ U||^2 + lam * ||I||^2 + mu * ||U||^2
        return np.linalg.norm(self.R - (self.I @ self.U) * (self.R > 0)) ** 2 + self.lam * np.linalg.norm(
            self.I) ** 2 + self.mu * np.linalg.norm(self.U) ** 2

    def optimize(self):
        # gradient of I: -2 * ((R - I @ U) * (R > 0)) @ U.T + 2 * lam * I
        grad_U = (2*((self.R - self.I@self.U)[..., None]*np.expand_dims(-self.I, 1)*(self.R>0)[..., None]).sum(0)+2*self.mu*self.U.T).T
        # gradient of U: -2 * I.T @ ((R - I @ U) * (R > 0)) + 2 * mu * U
        grad_I = (2*(((self.R - self.I@self.U)[..., None]*(-self.U.T[None, ...]))*(self.R>0)[..., None]).sum(1)+2*self.lam*self.I)

        # update I
        self.I -= self.learning_rate * grad_I
        # update U
        self.U -= self.learning_rate * grad_U

    def compute_test_loss(self):
        if self.test_ds is None:
            return "Test dataset not provided"
        # test_loss = ||test_ds - I @ U||^2
        return np.linalg.norm(self.test_ds - (self.I @ self.U) * (self.test_ds > 0)) ** 2


if __name__ == '__main__':
    train_ds = np.load("../data/ratings_train.npy")
    train_ds = np.nan_to_num(train_ds)

    # Optionally, load the test dataset
    try:
        test_ds = np.load("../data/ratings_test.npy")
        test_ds = np.nan_to_num(test_ds)
    except FileNotFoundError:
        test_ds = None

    users_nb = train_ds.shape[0]
    movies_nb = train_ds.shape[1]

    R = train_ds

    model = GradientDescent(R, test_ds)

    model.train(train_steps=60)
    test_rmse = np.sqrt(
        ((test_ds - model.R_output * (test_ds > 0)) ** 2).sum() / (test_ds > 0).sum())
    print(f"test_rmse = {test_rmse}")
