import torch 
import os
from tqdm import trange
import argparse
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim


from model import Generator, Discriminator
from utils import D_train, G_train, save_models


############################### Vanilla GAN ####################################
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Normalizing Flow.')
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.0001,
                      help="The learning rate to use for training.")
    parser.add_argument("--batch_size", type=int, default=128, 
                        help="Size of mini-batches for SGD")

    args = parser.parse_args()


    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Data Pipeline
    print('Dataset loading...')
    # MNIST Dataset
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5))])

    train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='data/MNIST/', train=False, transform=transform, download=False)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=args.batch_size, shuffle=False)
    print('Dataset Loaded.')


    print('Model Loading...')
    mnist_dim = 784
    G = torch.nn.DataParallel(Generator(g_output_dim = mnist_dim)).cuda()
    D = torch.nn.DataParallel(Discriminator(mnist_dim)).cuda()


    # model = DataParallel(model).cuda()
    print('Model loaded.')
    # Optimizer 



    # define loss
    criterion = nn.BCELoss() 

    # define optimizers
    G_optimizer = optim.Adam(G.parameters(), lr = args.lr)
    D_optimizer = optim.Adam(D.parameters(), lr = args.lr)

    print('Start Training :')
    
    n_epoch = args.epochs
    for epoch in trange(1, n_epoch+1, leave=True):
        print('-------------epoch: ', epoch, '-------------')    
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, mnist_dim)
            D_loss = D_train(x, G, D, D_optimizer, criterion)
            G_loss = G_train(x, G, D, G_optimizer, criterion)

        D_loss = D_loss / len(train_loader)
        G_loss = G_loss / len(train_loader)
        print('D_loss: ', D_loss, 'G_loss: ', G_loss)

        if epoch % 10 == 0:
            save_models(G, D, 'checkpoints')

        # generate sample to see the improvement of the model
        z = torch.randn(args.batch_size, 100).cuda()
        x = G(z)
        x = x.reshape(args.batch_size, 28, 28)
        # visualize generated samples
        torchvision.utils.save_image(x[0], os.path.join('samples', f'{epoch}.png'))         
                
    print('Training done')
'''
################################################################################




############################# Wasserstein GAN ##################################
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Wasserstein GAN.')
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.00015,  # Lower learning rate
                      help="The learning rate to use for training.")
    parser.add_argument("--batch_size", type=int, default=128, 
                        help="Size of mini-batches for SGD")

    args = parser.parse_args()

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Data Pipeline
    print('Dataset loading...')
    # MNIST Dataset
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5))])

    train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='data/MNIST/', train=False, transform=transform, download=False)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=args.batch_size, shuffle=False)
    print('Dataset Loaded.')


    print('Model Loading...')
    mnist_dim = 784
    G = torch.nn.DataParallel(Generator(g_output_dim = mnist_dim)).cuda()
    D = torch.nn.DataParallel(Discriminator(mnist_dim)).cuda()


    # model = DataParallel(model).cuda()
    print('Model loaded.')


    # Optimizers
    G_optimizer = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    print('Start Training:')
    
    n_epoch = args.epochs
    for epoch in trange(1, n_epoch+1, leave=True):
        print('-------------epoch: ', epoch, '-------------')    
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, mnist_dim)
            D_loss = D_train(x, G, D, D_optimizer)
            G_loss = G_train(x, G, D, G_optimizer)

        D_loss = D_loss / len(train_loader)
        G_loss = G_loss / len(train_loader)
        print('D_loss: ', D_loss, 'G_loss: ', G_loss)

        if epoch % 10 == 0:
            save_models(G, D, 'checkpoints')

        # generate sample to see the improvement of the model
        z = torch.randn(args.batch_size, 100).cuda()
        x = G(z)
        x = x.reshape(args.batch_size, 28, 28)
        # visualize generated samples
        torchvision.utils.save_image(x[0], os.path.join('samples', f'{epoch}.png'))   

    print('Training done')
'''
################################################################################



################################# WGAN-GP ######################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Wasserstein GAN-GP.')
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.0001,  # Lower learning rate
                      help="The learning rate to use for training.")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Size of mini-batches for SGD")
    parser.add_argument("--gp_weight", type=int, default=10, 
                        help="Weight for gradient penalty")

    args = parser.parse_args()


    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Data Pipeline
    print('Dataset loading...')
    # MNIST Dataset
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5))])

    train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='data/MNIST/', train=False, transform=transform, download=False)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=args.batch_size, shuffle=False)
    print('Dataset Loaded.')


    print('Model Loading...')
    mnist_dim = 784
    G = torch.nn.DataParallel(Generator(g_output_dim = mnist_dim)).cuda()
    D = torch.nn.DataParallel(Discriminator(mnist_dim)).cuda()

    # model = DataParallel(model).cuda()
    print('Model loaded.')


    G_optimizer = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    print('Start Training :')

    n_epoch = args.epochs
    for epoch in trange(1, n_epoch + 1, leave=True):
        print('-------------epoch: ', epoch, '-------------')  
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, mnist_dim).cuda()
            D_loss = D_train(x, G, D, D_optimizer, args.gp_weight)
            G_loss = G_train(x, G, D, G_optimizer)

        D_loss = D_loss / len(train_loader)
        G_loss = G_loss / len(train_loader)
        print('D_loss: ', D_loss, 'G_loss: ', G_loss)

        if epoch % 10 == 0:
            save_models(G, D, 'checkpoints')

        # generate sample to see the improvement of the model
        z = torch.randn(args.batch_size, 100).cuda()
        x = G(z)
        x = x.reshape(args.batch_size, 28, 28)
        # visualize generated samples
        torchvision.utils.save_image(x[0], os.path.join('samples', f'{epoch}.png')) 

    print('Training done')

################################################################################