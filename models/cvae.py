import torch
import torch.nn as nn
import torch.nn.functional as F
ACTIVATION_FUNCTIONS = {
    "relu": F.relu,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "leaky_relu": F.leaky_relu,
}
class CVAE(nn.Module):
    def __init__(self,input_dim, encoder_layers, decoder_layers, latent_dim, cond_dim,activation):
        super(CVAE, self).__init__()
        self.activation = ACTIVATION_FUNCTIONS[activation]


        encoder_layers.insert(0, input_dim + cond_dim)
        self.encoder = nn.ModuleList()
        for i in range(len(encoder_layers)-1):
            self.encoder.append(nn.Linear(encoder_layers[i],encoder_layers[i+1]))
        self.fc_mu=nn.Linear(encoder_layers[-1],latent_dim)
        self.fc_logvar=nn.Linear(encoder_layers[-1],latent_dim)

        decoder_layers.insert(0, latent_dim + cond_dim) 
        self.decoder = nn.ModuleList()
        for i in range(len(decoder_layers) - 1):
            self.decoder.append(nn.Linear(decoder_layers[i], decoder_layers[i+1]))
        self.fc_output = nn.Linear(decoder_layers[-1], input_dim) 

    def encode(self, x, c):
        x_cond = torch.cat([x, c], dim=1)
        for layer in self.encoder:
            x_cond = self.activation(layer(x_cond))
        mu = self.fc_mu(x_cond)
        logvar = self.fc_logvar(x_cond)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        z_cond = torch.cat([z, c], dim=1)
        for layer in self.decoder:
            z_cond = self.activation(layer(z_cond))
        return torch.sigmoid(self.fc_output(z_cond))

    def forward(self, x,c):
        mu, logvar = self.encode(x,c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z,c), mu, logvar
