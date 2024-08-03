import numpy as np
import time

'''
The Alternating Least Squares algorithm for matrix factorization.
'''


class AlternatedLeastSquare:
    def __init__(self, R, test_ds=None, target_matrix_rank=1, lam=0.3, mu=0.35):
        self.R = np.nan_to_num(R) # shape: (m, n)
        self.test_ds = np.nan_to_num(test_ds)
        self.target_matrix_rank = target_matrix_rank  # k
        self.lam = lam
        self.mu = mu

        # self.I = np.random.rand(R.shape[0], target_matrix_rank)  # shape: (m, k)
        # self.U = np.random.rand(target_matrix_rank, R.shape[1])  # shape: (k, n)

        # Initialize I and U with original method
        I_init = np.sqrt(self.R.sum(1) / (self.R > 0).sum(1))[..., None]  # shape: (m, 1)
        U_init = np.sqrt(self.R.sum(0) / (self.R > 0).sum(0))[..., None].T  # shape: (1, n)

        if target_matrix_rank == 1:
            self.I = I_init
            self.U = U_init
        else:
            # Randomly initialize the remaining columns/rows
            I_rand = np.random.rand(R.shape[0], target_matrix_rank - 1)  # shape: (m, k-1)
            U_rand = np.random.rand(target_matrix_rank - 1, R.shape[1])  # shape: (k-1, n)

            # Concatenate to form I and U with desired shape: (m, k) and (k, n)
            self.I = np.hstack((I_init, I_rand))
            self.U = np.vstack((U_init, U_rand))

        self.R_output = None

    def train(self, train_steps):
        start_time = time.time()
        for step in range(train_steps):
            # compute loss
            loss = self.compute_loss()

            # update I and U alternatively
            self.optimize_I()
            self.optimize_U()

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
        mask = self.R > 0  # mask for known ratings

        # loss = ||R  - I @ U * (R>0)||^2 + lam * ||I||^2 + mu * ||U||^2
        loss = np.linalg.norm(self.R - self.I @ self.U * mask) ** 2
        loss += self.lam * np.linalg.norm(self.I) ** 2
        loss += self.mu * np.linalg.norm(self.U) ** 2
        return loss

    def optimize_I(self):
        mask = self.R > 0  # mask for known ratings

        for i in range(self.I.shape[0]):
            # Transpose the boolean mask to align dimensions
            relevant_mask = mask[i, :]
            relevant_U = self.U[:, relevant_mask]
            relevant_R = self.R[i, relevant_mask]

            # Update I matrix for the current item i
            self.I[i, :] = np.linalg.solve(relevant_U @ relevant_U.T + self.lam * np.eye(self.target_matrix_rank),
                                           relevant_U @ relevant_R)

    def optimize_U(self):
        mask = self.R > 0  # mask for known ratings

        for j in range(self.U.shape[1]):
            # Transpose the boolean mask to align dimensions
            relevant_mask = mask[:, j]
            relevant_I = self.I[relevant_mask, :]
            relevant_R = self.R[relevant_mask, j]

            # Update U matrix for the current user j
            self.U[:, j] = np.linalg.solve(relevant_I.T @ relevant_I + self.mu * np.eye(self.target_matrix_rank),
                                           relevant_I.T @ relevant_R)

    def compute_test_loss(self):
        mask = self.test_ds > 0  # mask for known ratings

        # test_loss = ||test_ds - I @ U * (test_ds>0)||^2
        test_loss = np.linalg.norm(self.test_ds - self.I @ self.U * mask) ** 2
        return test_loss


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
    lambdas = [0.3, 0.35, 0.4]
    mus = [0.3, 0.35, 0.4]
    train_steps = [20, 25, 30]

    best_rmse = float('inf')
    best_params = None

    for rank in ranks:
        for lam in lambdas:
            for mu in mus:
                for train_step in train_steps:
                    model = AlternatedLeastSquare(R, test_ds, target_matrix_rank=rank, lam=lam, mu=mu)
                    model.train(train_step)

                    test_rmse = np.sqrt(((test_ds - model.R_output * (test_ds > 0)) ** 2).sum() / (test_ds > 0).sum())
                    print(f"Rank: {rank}, Lambda: {lam}, Mu: {mu}, Test RMSE: {test_rmse}")

                    # update best hyperparams
                    if test_rmse < best_rmse:
                        best_rmse = test_rmse
                        best_params = (rank, lam, mu, train_step)

    print(
        f"Best parameters: Rank {best_params[0]}, Lambda {best_params[1]}, Mu {best_params[2]}, train_step {best_params[3]} with Test RMSE: {best_rmse}")
# if __name__ == '__main__':
#     train_ds = np.load("../data/ratings_train.npy")
#     train_ds = np.nan_to_num(train_ds)
#
#     # Optionally, load the test dataset
#     try:
#         test_ds = np.load("../data/ratings_test.npy")
#         test_ds = np.nan_to_num(test_ds)
#     except FileNotFoundError:
#         test_ds = None
#
#     users_nb = train_ds.shape[0]
#     movies_nb = train_ds.shape[1]
#
#     R = train_ds
#
#     model = AlternatedLeastSquare(R, test_ds, target_matrix_rank=2, lam=1e-1*2, mu=1e-1*2)
#     model.train(train_steps=60)
#
#     test_rmse = np.sqrt(
#         ((test_ds - model.R_output * (test_ds > 0)) ** 2).sum() / (test_ds > 0).sum())
#     print(f"test_rmse = {test_rmse}")

