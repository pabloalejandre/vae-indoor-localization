import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvEncoder(nn.Module):
    def __init__(self, num_feature_vectors, feature_vector_length, latent_dim):
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_feature_vectors, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        # Calculate the output size after each convolution
        conv_output_size = calculate_conv_output_size(feature_vector_length, 3, 1, 2)
        conv_output_size = calculate_conv_output_size(conv_output_size, 3, 1, 2)
        conv_output_size = calculate_conv_output_size(conv_output_size, 3, 1, 2)
        self.fc_mu = nn.Linear(in_features=64 * conv_output_size, out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=64 * conv_output_size, out_features=latent_dim)

    def forward(self, x):
        print(x.shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the output
        mean = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mean, logvar
    
    
def calculate_conv_output_size(input_size, kernel_size, padding, stride):
    return (input_size - kernel_size + 2 * padding) // stride + 1


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim, num_feature_vectors, output_vector_length):
        super(ConvDecoder, self).__init__()
        self.fc = nn.Linear(in_features=latent_dim, out_features=64 * output_vector_length // 8)
        self.conv_transpose1 = nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_transpose2 = nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_transpose3 = nn.ConvTranspose1d(16, num_feature_vectors, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, z):
        z = self.fc(z)
        z = z.view(z.size(0), 64, -1)  # Reshape into a 'feature map'
        z = F.relu(self.conv_transpose1(z))
        z = F.relu(self.conv_transpose2(z))
        z = self.conv_transpose3(z) #torch.sigmoid(self.conv_transpose3(z))  # Sigmoid for outputs between 0 and 1
        return z


class VariationalAutoencoder(nn.Module):
    def __init__(self, num_feature_vectors, feature_vector_length, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = ConvEncoder(num_feature_vectors, feature_vector_length, latent_dim)
        self.decoder = ConvDecoder(latent_dim, num_feature_vectors, feature_vector_length)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

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

# Initialize the VAE
# num_feature_vectors = number of rows in your input matrix
# feature_vector_length = number of columns in your input matrix
# latent_dim = size of the latent space
#num_feature_vectors = 3  # For example
#feature_vector_length = 66  # For example
#latent_dim = 20
#conv_vae = VariationalAutoencoder(num_feature_vectors, feature_vector_length, latent_dim)
