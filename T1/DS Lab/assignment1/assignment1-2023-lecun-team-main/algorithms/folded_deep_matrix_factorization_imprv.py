import torch
import numpy as np
import time
import re
import torch.nn as nn
from torch.nn import functional as F
import math
import tqdm
import os
#from torch.utils.tensorboard import SummaryWriter

'''
Deep Matrix Factorization Algorithm with our team's improvements over paper and using folding
'''

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class SimilarityModel(nn.Module):
    def __init__(self, users_nb, movies_nb, counter, cls_counter):
        super().__init__()
        self.user_linear_1 = nn.Linear(movies_nb, 512)
        self.user_linear_2 = nn.Linear(512, 512)
        self.user_linear_3 = nn.Linear(512, 256)

        self.movie_linear_1 = nn.Linear(users_nb + counter + cls_counter - 1, 1024)
        #self.movie_linear_1 = nn.Linear(users_nb + cls_counter - 1, 1024)
        self.movie_linear_2 = nn.Linear(1024, 256)

        #self.common_projector = torch.nn.Parameter(torch.empty((256, 256)))
        #nn.init.kaiming_uniform_(self.common_projector, a = math.sqrt(5))

    def forward(self, users_features, movies_features):
        users_features = F.dropout(users_features, p = 0.4, training = self.training)
        movies_features = F.dropout(movies_features, p = 0.4, training = self.training)

        users_h = F.relu(self.user_linear_1(users_features))
        users_h = F.relu(self.user_linear_2(users_h))
        users_h_transf = self.user_linear_3(users_h)

        movies_h = F.relu(self.movie_linear_1(movies_features))
        movies_h = F.relu(self.movie_linear_2(movies_h))

        #scalar_product =  (users_h.unsqueeze(-2) @ ((self.common_projector @ self.common_projector.T) @ movies_h.T).T.unsqueeze(-1)).squeeze()
        #norm_users_h = (users_h.unsqueeze(-2) @ ((self.common_projector @ self.common_projector.T) @ users_h.T).T.unsqueeze(-1)).squeeze()
        #norm_movies_h = (movies_h.unsqueeze(-2) @ ((self.common_projector @ self.common_projector.T) @ movies_h.T).T.unsqueeze(-1)).squeeze()
        #cosine_simiarity = scalar_product / (torch.sqrt(norm_users_h * norm_movies_h))
        cosine_simiarity = F.cosine_similarity(users_h_transf, movies_h, dim = 1)

        return cosine_simiarity, users_h, movies_h

class FullModel(nn.Module):
    def __init__(self, users_nb, movies_nb, counter, cls_counter):
        super().__init__()
        self.similarity_model = SimilarityModel(users_nb, movies_nb, counter, cls_counter)
        self.user_decoder_1 = nn.Linear(512, 512)
        self.user_decoder_2 = nn.Linear(512, movies_nb)

        self.movie_decoder_1 = nn.Linear(256, 1024)
        #self.movie_decoder_2 = nn.Linear(1024, users_nb + cls_counter - 1)
        self.movie_decoder_2 = nn.Linear(1024, users_nb + counter + cls_counter - 1)

    def forward(self, users_features, movies_features):
        cosine_simiarity, users_h, movies_h = self.similarity_model(users_features, movies_features)

        users_decoded = F.relu(self.user_decoder_1(users_h))
        users_decoded = self.user_decoder_2(users_decoded)

        movies_decoded = F.relu(self.movie_decoder_1(movies_h))
        movies_decoded = self.movie_decoder_2(movies_decoded)

        return cosine_simiarity, users_decoded, movies_decoded

class FoldedDeepMatrixFactorization:
    def __init__(self, R = None, test_ds = None, learning_rate = 1e-3, batch_size = 256, folds_nb = 5, eval = False, mode = 'train'):
        R = np.nan_to_num(R)
        bincounts, classes = np.histogram(R)
        bincounts, classes = bincounts[1:], classes[1:]

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.eval = eval

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.R = R = torch.tensor(R).to(self.device)
        classes, bincounts = torch.unique(self.R, return_counts = True)
        classes, bincounts = bincounts[1:], classes[1:]
        self.class_weights = bincounts.sum() / (classes.shape[0] * bincounts)

        self.users_nb, self.movies_nb = R.shape[0], R.shape[1]

        script_dir = os.path.dirname(__file__)
        rel_path = "../metadata/namesngenre.npy"
        abs_file_path = os.path.join(script_dir, rel_path)
        self.movies_date, self.movies_classes_matrix, self.counter, self.cls_counter = self.make_metadata(np.load(abs_file_path))

        self.gpus = [0]
        self.folds_nb = folds_nb
        if mode == 'train':
            self.folds_train, self.folds_test = self.make_folds(R, folds_nb)
            self.models = [FullModel(self.users_nb, self.movies_nb, self.counter, self.cls_counter).to(self.device) for _ in range(folds_nb)]

        self.best_models = [SimilarityModel(self.users_nb, self.movies_nb, self.counter, self.cls_counter).to(self.device) for _ in range(folds_nb)]

        self.test_ds = test_ds
        self.learning_rate = learning_rate

        self.R_output = None

        self.set_up_tensorboard("AdamW_lr1e-3_12epochs_asym_transf_dec_512")

    def set_up_tensorboard(self, spec = "vanilla"):
        run_name = f"{int(time.time())}"
        #self.writer = SummaryWriter(f"runs/{spec}_{run_name}")
        #writer.add_text(
        #    "hyperparameters",
        #    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        #)

    def make_metadata(self, meta_datas):
        meta_datas[3674][0] += ' (1994)'
        meta_datas[4920][0] += ' (2016)'
        meta_datas[4940][0] += ' (2016)'
        
        dates_to_token_map = {}
        counter = 0
        dates = torch.zeros(meta_datas.shape[0], dtype=torch.int64, device = self.device)
        regex = r'.*\(([0-9]{4})\)'
        for i, m in enumerate(meta_datas):
            date = re.findall(regex, m[0])[0]
            if date not in dates_to_token_map:
                dates_to_token_map[date] = counter
                counter += 1

            dates[i] = dates_to_token_map[date]

        movies_date = F.one_hot(dates).float()
        classes = {}
        cls_counter = 0
        for m in meta_datas:
            classes_ = m[1].split('|')
            for c in classes_:
                if c not in classes:
                    classes[c] = cls_counter
                    cls_counter += 1
        movies_classes_matrix = torch.zeros((meta_datas.shape[0], cls_counter - 1), device = self.device)
        for i, m in enumerate(meta_datas):
            classes_ = m[1].split('|')
            for c in classes_:
                if classes[c] != 19:
                    movies_classes_matrix[i, classes[c]] = 1

        return movies_date, movies_classes_matrix, counter, cls_counter

    def make_folds(self, R, folds_nb):
        non_zero_idx = (R > 0).argwhere()
        perm_inds = torch.randperm(non_zero_idx.shape[0])
        non_zero_idx = non_zero_idx[perm_inds]

        item_per_fold = non_zero_idx.shape[0] // folds_nb
        folds_train = R.unsqueeze(0).repeat(folds_nb, 1, 1)
        folds_test = torch.zeros_like(folds_train)
        for f in range(folds_nb):
            start = item_per_fold * f
            end = item_per_fold * (f + 1) if f != folds_nb - 1 else -1
            fold_idx = non_zero_idx[start : end]

            for x, y in fold_idx:
                folds_train[f, x, y] = 0
                folds_test[f, x, y] = R[x, y]

        return folds_train, folds_test

    def make_test_train_ds(self, fold_nb):
        train_ds, test_ds = self.folds_train[fold_nb], self.folds_test[fold_nb]

        # Train dataset
        non_zero_idx = (train_ds > 0).argwhere()
        users_explicit_ds = train_ds[non_zero_idx[:, 0]].float()
        for idx, (x, y) in enumerate(non_zero_idx):
            users_explicit_ds[idx, y] = 0

        movies_explicit_ds = torch.cat([
            train_ds[:, non_zero_idx[:, 1]].T / 5,
            self.movies_date[non_zero_idx[:, 1]],
            self.movies_classes_matrix[non_zero_idx[:, 1]]
        ], dim = 1).float()
        for idx, (x, y) in enumerate(non_zero_idx):
            movies_explicit_ds[idx, x] = 0
        explicit_ratings_ds = torch.cat([train_ds[x, y].unsqueeze(0) for x, y in non_zero_idx]).float()

        # Test dataset
        test_sparse_idx = (test_ds > 0).argwhere()
        test_users_explicit_ds = train_ds[test_sparse_idx[:, 0]].float()
        for idx, (x, y) in enumerate(test_sparse_idx):
            test_users_explicit_ds[idx, y] = 0

        test_movies_explicit_ds = torch.cat([
            train_ds[:, test_sparse_idx[:, 1]].T / 5,
            self.movies_date[test_sparse_idx[:, 1]],
            self.movies_classes_matrix[test_sparse_idx[:, 1]]
        ], dim = 1).float()
        for idx, (x, y) in enumerate(test_sparse_idx):
            test_movies_explicit_ds[idx, x] = 0

        return users_explicit_ds, movies_explicit_ds, explicit_ratings_ds, test_users_explicit_ds, test_movies_explicit_ds

    def train(self, train_steps):
        folds_best_rmse_loss = torch.zeros(self.folds_nb)
        improvements = torch.zeros(self.folds_nb, train_steps)
        for f in range(self.folds_nb):
            print(f"\nTraining fold {f + 1}/{self.folds_nb}")
            t = tqdm.trange(1, train_steps + 1)
            users_explicit_ds, movies_explicit_ds, explicit_ratings_ds, test_users_explicit_ds, test_movies_explicit_ds = self.make_test_train_ds(f)
            test_ds_torch = self.folds_test[f]
            test_sparse_idx = (test_ds_torch > 0).argwhere()

            sim_model = self.models[f]
            optimizer = torch.optim.AdamW(sim_model.parameters(), lr = self.learning_rate)

            best_test_loss = torch.inf
            rmse_best_test_loss = torch.inf

            batch_number = users_explicit_ds.shape[0] // self.batch_size
            global_step = 0
            for step in t:
                b_inds = torch.randperm(users_explicit_ds.shape[0])
                total_loss = torch.zeros(1, device = self.device)

                batch_counter = 0
                for start in range(0, batch_number * self.batch_size, self.batch_size):
                    end = start + self.batch_size
                    mb_inds = b_inds[start : end]

                    # For the dropout
                    sim_model.train()

                    users_features_batch = users_explicit_ds[mb_inds] / 5
                    movies_features_batch = movies_explicit_ds[mb_inds]

                    class_indices = (explicit_ratings_ds[mb_inds] * 2 - 1).long()
                    target_class_weights = self.class_weights[class_indices]

                    ratings_batch = explicit_ratings_ds[mb_inds] / 5
                    similarities, users_decoded, movies_decoded = sim_model(users_features_batch, movies_features_batch)
                    similarities = (similarities + 1) / 2

                    users_recons_loss = F.binary_cross_entropy_with_logits(users_decoded, users_features_batch, reduction = "mean")
                    movies_recons_loss = F.binary_cross_entropy_with_logits(movies_decoded, movies_features_batch, reduction = "mean")
                    loss = F.binary_cross_entropy(similarities, ratings_batch, reduction = "mean") + 0.5 * (users_recons_loss + movies_recons_loss)
                    #loss = F.binary_cross_entropy(similarities, ratings_batch)

                    optimizer.zero_grad()
                    loss.backward()

                    grad_norm = nn.utils.clip_grad_norm_(sim_model.parameters(), max_norm=float('inf'))

                    optimizer.step()

                    total_loss += loss.item()

                    sim_model.eval()

                    with torch.no_grad():
                        test_predictions = torch.zeros(test_users_explicit_ds.shape[0], device = self.device)
                        test_b_inds = torch.randperm(test_users_explicit_ds.shape[0])
                        for test_start in range(0, test_users_explicit_ds.shape[0], self.batch_size):
                            test_end = test_start + self.batch_size
                            test_mb_inds = test_b_inds[test_start:test_end]

                            test_users_features_batch = test_users_explicit_ds[test_mb_inds] / 5
                            test_movies_features_batch = test_movies_explicit_ds[test_mb_inds]
                            test_similarities, _, _ = sim_model(test_users_features_batch, test_movies_features_batch)
                            test_similarities = (test_similarities + 1) / 2

                            test_predictions[test_mb_inds] = test_similarities * 5

                        predicted_matrix = torch.sparse_coo_tensor(test_sparse_idx.T, test_predictions, (self.users_nb, self.movies_nb))

                        test_rmse = torch.sqrt(((test_ds_torch - predicted_matrix.to_dense() * (test_ds_torch>0)) ** 2).sum() / (test_ds_torch>0).sum())

                        if test_rmse < rmse_best_test_loss:
                            improvements[f, step - 1] += 1
                            test_loss = torch.linalg.norm(test_ds_torch - predicted_matrix.to_dense() * (test_ds_torch>0))**2
                            self.best_models[f].load_state_dict(sim_model.similarity_model.state_dict())
                            best_test_loss = test_loss
                            rmse_best_test_loss = test_rmse

                    batch_counter += 1
                    global_step += 1
                    #self.writer.add_scalar(f"losses/train_fold_{f}", loss.item(), global_step)
                    #self.writer.add_scalar(f"scalars/grad_norm_fold_{f}", grad_norm.cpu().item(), global_step)
                    #self.writer.add_scalar(f"losses/best_test_rmse_fold_{f}", rmse_best_test_loss.item(), global_step)
                    t.set_postfix(train_loss = loss.item(), best_test_loss = best_test_loss.item(), rmse_best_test_loss = rmse_best_test_loss.item())
                folds_best_rmse_loss[f] = rmse_best_test_loss

            del users_explicit_ds, movies_explicit_ds, explicit_ratings_ds, test_users_explicit_ds, test_movies_explicit_ds
            torch.cuda.empty_cache()
            #self.writer.add_histogram(f"histograms/improvements_per_fold_dist{f}", improvements[f].numpy(), global_step)
            print(improvements[f])

        print(improvements.sum(0))
        print(f"Cross-validation mean: {folds_best_rmse_loss.mean().item()}")
        #self.writer.add_scalar(f"final_results/cross_validation_mean", folds_best_rmse_loss.mean().item(), 1)

        if self.eval:
            self.evaluation()

        self.save_best_models()

    def save_best_models(self):
        script_dir = os.path.dirname(__file__)
        rel_path = "../weights/dmf/"

        for i, m in enumerate(self.best_models):
            abs_file_path = os.path.join(script_dir, f"{rel_path}fold_{i}.pt")
            torch.save(m.state_dict(), abs_file_path)

    def load_best_models(self):
        script_dir = os.path.dirname(__file__)
        rel_path = "../weights/dmf/"

        for i, m in enumerate(self.best_models):
            abs_file_path = os.path.join(script_dir, f"{rel_path}fold_{i}.pt")
            m.load_state_dict(torch.load(abs_file_path))

    def make_output(self, batch_size = 256, tqdm_on = True):
        print("\nGenerating output")
        test_sparse_idx = (self.R == 0).argwhere()

        if tqdm_on:
            t = tqdm.trange(1, test_sparse_idx.shape[0])

        test_b_inds = torch.arange(test_sparse_idx.shape[0])
        test_predictions = torch.zeros(self.folds_nb, test_sparse_idx.shape[0], device = self.device)
        for test_start in range(0, test_sparse_idx.shape[0], batch_size):
            test_end = test_start + batch_size
            test_mb_inds = test_b_inds[test_start : test_end]

            test_users_idx = test_sparse_idx[test_mb_inds, 0]
            test_users_explicit_ds = self.R[test_users_idx].float() / 5

            test_movies_idx = test_sparse_idx[test_mb_inds, 1]
            test_movies_explicit_ds = torch.cat([self.R[:, test_movies_idx].T / 5, self.movies_date[test_movies_idx], self.movies_classes_matrix[test_movies_idx]], dim = 1).float()

            for f in range(self.folds_nb):
                sim_model = self.best_models[f]
                sim_model.eval()
                
                with torch.no_grad():
                    test_similarities, _, _ = sim_model(test_users_explicit_ds, test_movies_explicit_ds)
                    test_similarities = (test_similarities + 1) / 2
    
                    test_predictions[f, test_mb_inds] = test_similarities * 5

            if tqdm_on:
                t.update(batch_size)

            del test_users_explicit_ds, test_movies_explicit_ds, test_users_idx, test_movies_idx, test_similarities
            torch.cuda.empty_cache()

        test_predictions = test_predictions.mean(0)
        predicted_matrix = torch.sparse_coo_tensor(test_sparse_idx.T, test_predictions, (self.users_nb, self.movies_nb))
        self.R_output = predicted_matrix.to_dense().cpu().numpy()
        print("DONE!")

    def evaluation(self):
        train_ds = np.load("./ratings_train.npy")
        test_ds = np.load("./ratings_test.npy")
        train_ds_torch = torch.tensor(np.nan_to_num(train_ds), device = self.device)
        test_ds_torch = torch.tensor(np.nan_to_num(test_ds), device = self.device)

        test_sparse_idx = (test_ds_torch > 0).argwhere()
        test_users_explicit_ds = torch.cat([
            train_ds_torch[x].unsqueeze(0) for x, y in test_sparse_idx
        ]).float()
        for idx, (x, y) in enumerate(test_sparse_idx):
            test_users_explicit_ds[idx, y] = 0
        
        test_movies_explicit_ds = torch.cat([
            torch.cat([train_ds_torch[:, y].unsqueeze(0) / 5, self.movies_date[y].unsqueeze(0), self.movies_classes_matrix[y].unsqueeze(0)], dim = 1) for x, y in test_sparse_idx
        ]).float()
        for idx, (x, y) in enumerate(test_sparse_idx):
            test_movies_explicit_ds[idx, x] = 0

        test_predictions = torch.zeros(self.folds_nb, test_users_explicit_ds.shape[0], device = self.device)

        for f in range(self.folds_nb):
            sim_model = self.best_models[f]
            sim_model.eval()
            test_b_inds = torch.randperm(test_users_explicit_ds.shape[0])
            with torch.no_grad():
                for test_start in range(0, test_users_explicit_ds.shape[0], self.batch_size):
                    test_end = test_start + self.batch_size
                    test_mb_inds = test_b_inds[test_start : test_end]
    
                    test_users_features_batch = test_users_explicit_ds[test_mb_inds] / 5
                    test_movies_features_batch = test_movies_explicit_ds[test_mb_inds]
                    test_similarities, _, _ = sim_model(test_users_features_batch, test_movies_features_batch)
                    test_similarities = (test_similarities + 1) / 2
    
                    test_predictions[f, test_mb_inds] = test_similarities * 5

        test_predictions = test_predictions.mean(0)
        predicted_matrix = torch.sparse_coo_tensor(test_sparse_idx.T, test_predictions, (self.users_nb, self.movies_nb))
        test_loss = torch.linalg.norm(test_ds_torch - predicted_matrix.to_dense() * (test_ds_torch>0))**2
        test_rmse = torch.sqrt(((test_ds_torch - predicted_matrix.to_dense() * (test_ds_torch>0)) ** 2).sum() / (test_ds_torch>0).sum())
        print(f"Eval loss: {test_loss.item()}, eval rmse loss: {test_rmse.item()}")

