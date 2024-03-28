import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.data_utils import MyDataset, get_MDP_from_data, preprocess_mdps, augment_data

#Function to calculate output size of 1D convolution
def calculate_conv1d_output_size(length, kernel, padding=0, stride=1):
    return (length - kernel + 2*padding) // stride + 1

#Loss function of the VAE
def vae_loss(recon_x, x, mean, logvar, beta=0.5):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return (1-beta)*MSE + beta*KLD, MSE, KLD

#Hyperparameters
scenario_name, feature_length = 'LivingRoom', 16
#scenario_name, feature_length = 'BoxLectureRoom', 14
#scenario_name, feature_length = 'LShapedRoom', 15
data_path = f'scenarios/{scenario_name}/data/training.json'
latent_dim = 2
num_epochs = 300
batch_size = 64
learning_rate = 1e-3

class Encoder(nn.Module):
    def __init__(self, feature_length, latent_dim):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=4, stride=1)
        conv_output_length = calculate_conv1d_output_size(feature_length, 4)
        
        self.fc1 = nn.Linear(in_features=4*conv_output_length, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=8)
        self.fc_mean = nn.Linear(in_features=8, out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=8, out_features=latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, feature_length):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(in_features=latent_dim, out_features=8)
        self.fc2 = nn.Linear(in_features=8, out_features=16)
        conv_output_length = calculate_conv1d_output_size(feature_length, 4)
        self.fc3 = nn.Linear(in_features=16, out_features=4*conv_output_length)
        self.convt1 = nn.ConvTranspose1d(in_channels=4, out_channels=1, kernel_size=4, stride=1)

    def forward(self, z, feature_length):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = F.relu(self.fc3(z))

        conv_output_length = calculate_conv1d_output_size(feature_length, 4)
        z = z.view(z.size(0), 4, conv_output_length)
        reconstruction = self.convt1(z)
        return reconstruction

class VariationalAutoencoder(pl.LightningModule):
    def __init__(self, feature_length, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        self.save_hyperparameters()

        self.encoder = Encoder(feature_length, latent_dim)
        self.decoder = Decoder(latent_dim, feature_length)
        self.feature_length = feature_length

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        return self.decoder(z, self.feature_length), mean, logvar

    def training_step(self, batch, batch_idx):
        x = batch

        if self.current_epoch < 150:
            group_size = 3
        elif self.current_epoch in range(150, 250):
            group_size = 4
        else:
            group_size = 5

        delete_prob = 0.5 if self.current_epoch < 150 else 0.65

        x_augmented = augment_data(x, delete_probability=delete_prob, max_group_size=group_size)
        recon_x, mean, logvar = self(x_augmented)
        loss, mse, kld = vae_loss(recon_x, x, mean, logvar)
        self.log('train_loss', loss, prog_bar=True)
        self.log('MSE_loss', mse)
        self.log('KLD_loss', kld)
        return loss
    
    def configure_optimizers(self):
        lr = learning_rate
        return torch.optim.Adam(self.parameters(), lr=lr)

    def train_dataloader(self):
        train_dataset = get_MDP_from_data(path=data_path)
        train_dataset = preprocess_mdps(train_dataset, feature_length)
        train_dataset = MyDataset(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        return train_loader
    

if __name__ == '__main__':

    checkpoint_callback = ModelCheckpoint(
        dirpath = 'trained_models/',
        filename= scenario_name + '_{epoch}-{train_loss:.2f}',
        monitor='train_loss',
        verbose=True,
        save_last=True,
        save_top_k=1,
        mode='min',
        save_weights_only=False,
        every_n_epochs=1
    )

    trainer = Trainer(
        callbacks= [checkpoint_callback],
        max_epochs=num_epochs, 
        fast_dev_run=False)
    
    model = VariationalAutoencoder(feature_length=feature_length, latent_dim=latent_dim)
    trainer.fit(model)
    