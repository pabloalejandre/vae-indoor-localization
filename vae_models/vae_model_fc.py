#Model to test a fully connected architecture for scenarios with only one transmitter

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, feature_length, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(in_features=feature_length, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=16)
        
        self.fc_mean = nn.Linear(in_features= 16, out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features= 16, out_features=latent_dim)        

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, feature_length):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(in_features=latent_dim, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=feature_length)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        reconstruction = self.fc3(z)
        return reconstruction


class VariationalAutoencoder(nn.Module):
    def __init__(self, feature_length, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(feature_length, latent_dim)
        self.decoder = Decoder(latent_dim, feature_length)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        return self.decoder(z), mean, logvar
    

#Reconstruction loss (MSE, BCE...) + KL-divergence between latent distribution and prior distribution
def vae_loss(recon_x, x, mean, logvar, beta=0.7):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return (1-beta)*MSE + beta*KLD, MSE, KLD

