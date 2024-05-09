"""VAE模型"""

import torch
import torch.nn as nn
from typing import List

class VAE(nn.Module):
    def __init__(self, 
                 in_channels: int=3,
                 latent_dim: int=128,
                 hidden_dim: List=[32, 64, 128, 256],
                 **kwargs):
        super(VAE, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.encoder = Encoder(self.in_channels, self.latent_dim, self.hidden_dim)
        self.decoder = Decoder(self.hidden_dim[::-1], self.latent_dim, self.in_channels)
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decoder(z)
        return (recon_x, x, mu, log_var)


    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

    def loss_function(self,*args,**kwargs):
        results = args[0]
        recon_x = results[0]
        x = results[1]
        mu = results[2]
        log_var = results[3]

        recons_loss = torch.nn.functional.mse_loss(recon_x, x)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - torch.exp(log_var), dim=1), dim=0)

        loss = recons_loss + kwargs["kld_weight"]*kld_loss
        return {'loss': loss, "Reconstruction_loss": recons_loss.detach(), "KLD": -kld_loss.detach()}
        


    def sample(self, num_samples, device):
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(device)
        samples = self.decoder(z)
        return samples

    def generate(self, x):
        return self.forward(x)[0]

class Encoder(nn.Module):
    def __init__(self,
                 in_channels: int=3,
                 latent_dim: int=128,
                 hidden_dim: List = [32, 64, 128, 256]):
        super(Encoder, self).__init__()
        moudles = []
        for i in range(len(hidden_dim)): # Nx28x28x3 -> Nx14x14x32 -> Nx7x7x64 -> Nx4x4x128 -> Nx2x2x256 
            moudles.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim[i], kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dim[i]),
                    nn.LeakyReLU()
                )
            )
            in_channels = hidden_dim[i]
        
        self.encoder = nn.Sequential(*moudles)
        self.fc_mu = nn.Linear(hidden_dim[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dim[-1]*4, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.flatten(start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return [mu, log_var]



class Decoder(nn.Module):
    def __init__(self,
                 hidden_dim: List = [256, 128, 64, 32],
                 latent_dim: int=128,
                 out_channel: int=1):
        super(Decoder, self).__init__()
        self.decoder_input = nn.Linear(latent_dim, hidden_dim[0]*4)
        
        moudles = []
        for i in range(len(hidden_dim)-1): # Nx2x2x256 -> Nx4x4x128 -> Nx7x7x64 -> Nx14x14x32
            moudles.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dim[i], hidden_dim[i+1], kernel_size=3, stride=2, padding=1, output_padding=0 if hidden_dim[i+1]==64 else 1),
                    nn.BatchNorm2d(hidden_dim[i+1]),
                    nn.LeakyReLU(),
                )
            )
        self.decoder = nn.Sequential(*moudles)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim[-1], hidden_dim[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dim[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim[-1], out_channel, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.decoder_input(x)
        x = x.view(-1, 256, 2, 2)
        x = self.decoder(x)
        x = self.final_layer(x)
        return x



