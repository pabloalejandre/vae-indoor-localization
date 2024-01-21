import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot

class Encoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64, 32)
        self.fc_mean = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_shape):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, output_shape)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        reconstruction = self.fc4(h) #torch.sigmoid(self.fc3(h))
        return reconstruction


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(input_shape, latent_dim)
        self.decoder = Decoder(latent_dim, input_shape)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mean + eps*std

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        return self.decoder(z), mean, logvar

#Reconstruction loss (MSE, BCE...) + KL-divergence between latent distribution and prior distribution
def vae_loss(recon_x, x, mean, logvar):
    print("VAE input:" , x)
    print("VAE output:" , recon_x)
    MSE = F.mse_loss(recon_x, x, reduction='sum') #BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return MSE + KLD  #BCE + KLD


# Visualize structure of the model
#input_shape = 198 # size of each delay profile
#latent_dim = 20 # size of the latent vector
#vae = VariationalAutoencoder(input_shape, latent_dim)

#sample_input = torch.randn(1, input_shape)
#output, _, _ = vae(sample_input)
#make_dot(output, params=dict(list(vae.named_parameters()))).render("vae_graph", format="png")
