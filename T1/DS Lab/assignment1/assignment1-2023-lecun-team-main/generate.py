import numpy as np
import os
from tqdm import tqdm, trange
import argparse

from algorithms.folded_deep_matrix_factorization_imprv import FoldedDeepMatrixFactorization
from algorithms.gradient_descent import GradientDescent
from algorithms.alternated_least_square import AlternatedLeastSquare

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a completed ratings table.')
    parser.add_argument("--name", type=str, default="ratings_eval.npy",
                      help="Name of the npy of the ratings table to complete")

    args = parser.parse_args()



    # Open Ratings table
    print('Ratings loading...') 
    table = np.load(args.name) ## DO NOT CHANGE THIS LINE
    print('Ratings Loaded.')

    # Training our deep matrix factorization
    #model = FoldedDeepMatrixFactorization(table, folds_nb = 20)
    #model.train(train_steps=12)

    # Training our GD matrix factorization
    # model = GradientDescent(table)
    # model.train(60)

    # Training our ALS matrix factorization
    #model = AlternatedLeastSquare(table)
    #model.train(25)

    # Inference for deep matrix factorization
    model = FoldedDeepMatrixFactorization(table, folds_nb = 20, mode = 'infer')
    model.load_best_models()
    model.make_output(batch_size = 65536, tqdm_on = False)

    # Save the completed table
    np.save("output.npy", model.R_output) ## DO NOT CHANGE THIS LINE
