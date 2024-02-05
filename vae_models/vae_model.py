import torch
import torch.nn as nn
import torch.nn.functional as F

#Pooling size
N = 1

class Encoder(nn.Module):
    def __init__(self, number_of_transmitters, latent_dim):
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(in_channels=number_of_transmitters, out_channels=16, kernel_size=3, stride=1, padding=1)
        
        self.fc_mean = nn.Linear(in_features= 64*N, out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features= 64*N, out_features=latent_dim)        

    def forward(self, x):
        #Conv Layers
        x = F.relu(self.conv(x))

        
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, number_of_transmitters):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(in_features=latent_dim, out_features= 64*N) 
        
        self.conv_transpose0 = nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_transpose1 = nn.ConvTranspose1d(32, number_of_transmitters, kernel_size=3, stride=2, padding=1, output_padding=1)
        
    def forward(self, z, size):
        #Linear layer to upscale from latent space
        z = self.fc(z)
        #Reshape back to channeled representation and undo pooling 
        z = z.view(-1, 64, N) 
        z = F.interpolate(z, size, mode='nearest')
        #Conv Layers
        z = F.relu(self.conv_transpose0(z))
        reconstruction = torch.sigmoid(self.conv_transpose1(z))
        return reconstruction


class VariationalAutoencoder(nn.Module):
    def __init__(self, number_of_transmitters, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(number_of_transmitters, latent_dim)
        self.decoder = Decoder(latent_dim, number_of_transmitters)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar, size = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        return self.decoder(z, size), mean, logvar
    

#Reconstruction loss (MSE, BCE...) + KL-divergence between latent distribution and prior distribution
def vae_loss(recon_x, x, mean, logvar):
    x_totalsize = x.size()[1]*x.size()[2]
    BCE = F.binary_cross_entropy(recon_x.view(-1, x_totalsize), x.view(-1, x_totalsize), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_channels=4, conv_out_channels=16, conv_kernel_size=3, conv_stride=1, conv_padding=1, feature_size=128, latent_dim=20):
        super(VAE, self).__init__()
        # Encoder
        self.conv1d = nn.Conv1d(in_channels=input_channels, out_channels=conv_out_channels, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding)
        self.fc1 = nn.Linear(conv_out_channels, feature_size)
        self.fc_mu = nn.Linear(feature_size, latent_dim)
        self.fc_logvar = nn.Linear(feature_size, latent_dim)

    def encode(self, x):
        x = F.relu(self.conv1d(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc_mu(x), self.fc_logvar(x)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    # Placeholder for the decoder, which is not defined yet
    def decode(self, z):
        return z # This should be replaced with the actual decoder implementation
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Example instantiation of the VAE model
vae = VAE()
print(vae)
