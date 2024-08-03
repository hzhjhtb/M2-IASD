import numpy as np
import time

'''
The Gradient Descent algorithm with bias for matrix factorization

NOT TESTED COMPREHENSIVELY
'''


class GradientDescent:
    def __init__(self, R, test_ds=None, target_matrix_rank=1, lam=1e-3, mu=1e-3, gamma=1e-3, learning_rate=1e-4):

        self.R = np.nan_to_num(R)  # shape: (m, n)
        self.test_ds = np.nan_to_num(test_ds)
        self.target_matrix_rank = target_matrix_rank  # k
        self.lam = lam
        self.mu = mu
        self.gamma = gamma

        self.learning_rate = learning_rate

        # self.I = np.random.rand(R.shape[0], target_matrix_rank)  # shape: (m, k)
        # self.U = np.random.rand(target_matrix_rank, R.shape[1])  # shape: (k, n)
        self.I = np.sqrt(self.R.sum(1) / (self.R > 0).sum(1))[..., None]
        self.U = np.sqrt(self.R.sum(0) / (self.R > 0).sum(0))[..., None].T

        # self.I = np.zeros((self.R.shape[0], target_matrix_rank))  # shape: (m, k)
        # # self.U = np.zeros((target_matrix_rank, self.R.shape[1]))
        # self.U = np.random.rand(target_matrix_rank, self.R.shape[1])  # shape: (k, n)

        self.B = np.zeros_like(self.R)


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
        self.R_output = np.where(self.R > 0, self.R, self.I @ self.U + self.B)

    def compute_loss(self):
        # loss = ||R - I @ U||^2 + lam * ||I||^2 + mu * ||U||^2
        return np.linalg.norm(self.R - (self.I @ self.U + self.B) * (self.R > 0)) ** 2 + self.lam * np.linalg.norm(
            self.I) ** 2 + self.mu * np.linalg.norm(self.U) ** 2 + self.gamma * np.linalg.norm(self.B) ** 2

    def optimize(self):
        # gradient of U: -2 * I.T @ ((R - I @ U) * (R > 0)) + 2 * mu * U
        grad_U = (2 * ((self.R - self.I @ self.U - self.B)[..., None] * np.expand_dims(-self.I, 1) * (self.R > 0)[
            ..., None]).sum(
            0) + 2 * self.mu * self.U.T).T
        # gradient of I: -2 * ((R - I @ U - B) * (R > 0)) @ U.T + 2 * lam * I
        grad_I = (2 * (((self.R - self.I @ self.U - self.B)[..., None] * (-self.U.T[None, ...])) * (self.R > 0)[
            ..., None]).sum(
            1) + 2 * self.lam * self.I)
        # gradient of B: -2 * ((R - I @ U - B) * (R > 0)) + 2 * gamma * B
        grad_B = -2 * ((self.R - self.I @ self.U - self.B) * (self.R > 0)) + 2 * gamma * self.B


        # update I
        self.I -= self.learning_rate * grad_I
        # update U
        self.U -= self.learning_rate * grad_U
        # update B
        self.B -= self.learning_rate * grad_B

    def compute_test_loss(self):
        if self.test_ds is None:
            return "Test dataset not provided"
        # test_loss = ||test_ds - I @ U||^2
        return np.linalg.norm(self.test_ds - (self.I @ self.U + self.B) * (self.test_ds > 0)) ** 2




if __name__ == '__main__':
    train_ds = np.load("../data/ratings_train.npy")
    train_ds = np.nan_to_num(train_ds)

    try:
        test_ds = np.load("../data/ratings_test.npy")
        test_ds = np.nan_to_num(test_ds)
    except FileNotFoundError:
        test_ds = None

    users_nb = train_ds.shape[0]
    movies_nb = train_ds.shape[1]

    R = train_ds

    # hyperparams scale
    ranks = [1]
    lambdas = [0.3]
    mus = [0.3]
    gammas = [0.8]
    learning_rates = [0.0005]
    train_steps = [100]

    best_rmse = float('inf')
    best_params = None

    for rank in ranks:
        for lam in lambdas:
            for mu in mus:
                for gamma in gammas:
                    for learning_rate in learning_rates:
                        for train_step in train_steps:
                            model = GradientDescent(R, test_ds, target_matrix_rank=rank, lam=lam, mu=mu, gamma=gamma, learning_rate=learning_rate)
                            model.train(train_step)

                            test_rmse = np.sqrt(((test_ds - model.R_output * (test_ds > 0)) ** 2).sum() / (test_ds > 0).sum())
                            print(f"Rank: {rank}, Lambda: {lam}, Mu: {mu}, Gamma:{gamma}, lr:{learning_rate} Test RMSE: {test_rmse}")

                        # update best hyperparams
                            if test_rmse < best_rmse:
                                best_rmse = test_rmse
                                best_params = (rank, lam, mu, gamma, learning_rate, train_step)

    print(
        f"Best parameters: Rank {best_params[0]}, Lambda {best_params[1]}, Mu {best_params[2]}, Gamma {best_params[3]}, "
        f"lr {best_params[4]},train_step {best_params[5]} with Test RMSE: {best_rmse}")