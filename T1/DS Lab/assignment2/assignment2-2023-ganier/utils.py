import torch
import os
import torch.autograd as autograd


############################### Vanilla GAN ####################################
'''
def D_train(x, G, D, D_optimizer, criterion):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x, torch.ones(x.shape[0], 1)
    x_real, y_real = x_real.cuda(), y_real.cuda()

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on fake
    z = torch.randn(x.shape[0], 100).cuda()
    x_fake, y_fake = G(z), torch.zeros(x.shape[0], 1).cuda()

    D_output =  D(x_fake)
    
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
        
    return  D_loss.data.item()

    
def G_train(x, G, D, G_optimizer, criterion):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 100).cuda()
    y = torch.ones(x.shape[0], 1).cuda()
                 
    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()
'''
################################################################################



############################# Wasserstein GAN ##################################
'''
def D_train(x, G, D, D_optimizer):
    # Train the discriminator
    D.zero_grad()

    # Train discriminator on real
    x_real = x.cuda()
    D_real_loss = -torch.mean(D(x_real))

    # Train discriminator on fake
    z = torch.randn(x.shape[0], 100).cuda()
    x_fake = G(z).detach()
    D_fake_loss = torch.mean(D(x_fake))

    # Update discriminator
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
        
    for p in D.parameters():
        p.data.clamp_(-0.01, 0.01)  # Weight clipping for stability

    return D_loss.data.item()


def G_train(x, G, D, G_optimizer):
    G.zero_grad()

    z = torch.randn(x.shape[0], 100).cuda()    
    # For WGAN, the generator loss is -D(G(z))
    G_loss = -torch.mean(D(G(z)))

    G_loss.backward()
    G_optimizer.step()

    return G_loss.data.item()
'''
################################################################################



################################# WGAN-GP ######################################

def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1).expand_as(real_samples).cuda()
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    interpolated.requires_grad_(True)
    d_interpolated = D(interpolated)
    fake = torch.ones(real_samples.size(0), 1, device='cuda', dtype=torch.float32).requires_grad_(False)
    gradients = autograd.grad(outputs=d_interpolated, inputs=interpolated,
                              grad_outputs=fake, create_graph=True,
                              retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def D_train(x, G, D, D_optimizer, gp_weight):
    # Train the discriminator
    D.zero_grad()

    # Train discriminator on real
    x_real = x.cuda()
    D_real_loss = -torch.mean(D(x_real))

    # Train discriminator on fake
    z = torch.randn(x.shape[0], 100).cuda()
    x_fake = G(z).detach()
    D_fake_loss = torch.mean(D(x_fake))

    # Update discriminator
    gradient_penalty = compute_gradient_penalty(D, x_real.data, x_fake.data)
    D_loss = D_real_loss + D_fake_loss + gp_weight * gradient_penalty
    D_loss.backward()
    D_optimizer.step()

    return D_loss.item()

def G_train(x, G, D, G_optimizer):
    G.zero_grad()

    z = torch.randn(x.size(0), 100).cuda()
    G_loss = -torch.mean(D(G(z)))

    G_loss.backward()
    G_optimizer.step()

    return G_loss.item()

################################################################################



def save_models(G, D, folder):
    torch.save(G.state_dict(), os.path.join(folder,'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder,'D.pth'))


def load_model(G, folder):
    ckpt = torch.load(os.path.join(folder,'G.pth'))
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G
