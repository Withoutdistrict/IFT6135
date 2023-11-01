import random
import time
import json
import os
import pickle
import numpy
import matplotlib
# matplotlib.use("tkagg")
import matplotlib.pyplot

import numpy as np
from tqdm.auto import tqdm

import torch

from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision.utils import make_grid, save_image
from torchvision import transforms

import matplotlib.pyplot as plt
from pathlib import Path


def fix_experiment_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


fix_experiment_seed()

results_folder = Path("./results")
results_folder.mkdir(exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"


# Helper Functions
def show_image(image, nrow=8):
    # Input: image
    # Displays the image using matplotlib
    grid_img = make_grid(image.detach().cpu(), nrow=nrow, padding=0)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis('off')


# Visualize the Dataset
def visualize():
    train_dataloader, _ = get_dataloaders(data_root=data_root, batch_size=train_batch_size)
    imgs, labels = next(iter(train_dataloader))

    save_image((imgs + 1) * 0.5, './results/orig.png')
    show_image((imgs + 1) * 0.5)


def get_dataloaders(data_root, batch_size):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    transform = transforms.Compose((
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize))

    train = datasets.SVHN(data_root, split='train', download=True, transform=transform)
    test = datasets.SVHN(data_root, split='test', download=True, transform=transform)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size)

    return train_dataloader, test_dataloader


def save_logs(dictionary, log_dir):
    log_dir = os.path.join(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    # Log arguments
    with open(os.path.join(log_dir, "log.json"), "w") as f:
        json.dump(dictionary, f, indent=2)


def interpolate(model, z_1, z_2, n_samples):
    # Interpolate between z_1 and z_2 with n_samples number of points, with the first point being z_1 and last being z_2.
    # Inputs:
    #   z_1: The first point in the latent space
    #   z_2: The second point in the latent space
    #   n_samples: Number of points interpolated
    # Returns:
    #   sample: The mode of the distribution obtained by decoding each point in the latent space
    #           Should be of size (n_samples, 3, 32, 32)
    lengths = torch.linspace(0, 1, n_samples).unsqueeze(1).to(device)
    z = lengths * z_1 + (1 - lengths) * z_2  # WRITE CODE HERE (interpolate z_1 to z_2 with n_samples points)
    return model.decode(z).mode()


class Encoder(nn.Module):
    def __init__(self, nc, nef, nz, isize, device):
        super(Encoder, self).__init__()

        # Device
        self.device = device

        # Encoder: (nc, isize, isize) -> (nef*8, isize//16, isize//16)
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, nef, 4, 2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(nef),

            nn.Conv2d(nef, nef * 2, 4, 2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(nef * 2),

            nn.Conv2d(nef * 2, nef * 4, 4, 2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(nef * 4),

            nn.Conv2d(nef * 4, nef * 8, 4, 2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(nef * 8),
        )

    def forward(self, inputs):
        batch_size = inputs.size(0)
        hidden = self.encoder(inputs)
        hidden = hidden.view(batch_size, -1)
        return hidden


class Decoder(nn.Module):
    def __init__(self, nc, ndf, nz, isize):
        super(Decoder, self).__init__()

        # Map the latent vector to the feature map space
        self.ndf = ndf
        self.out_size = isize // 16
        self.decoder_dense = nn.Sequential(
            nn.Linear(nz, ndf * 8 * self.out_size * self.out_size),
            nn.ReLU(True)
        )

        self.decoder_conv = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf * 8, ndf * 4, 3, 1, padding=1),
            nn.LeakyReLU(0.2, True),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf * 4, ndf * 2, 3, 1, padding=1),
            nn.LeakyReLU(0.2, True),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf * 2, ndf, 3, 1, padding=1),
            nn.LeakyReLU(0.2, True),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf, nc, 3, 1, padding=1)
        )

    def forward(self, input):
        batch_size = input.size(0)
        hidden = self.decoder_dense(input).view(batch_size, self.ndf * 8, self.out_size, self.out_size)
        output = self.decoder_conv(hidden)
        return output


class DiagonalGaussianDistribution(object):
    # Gaussian Distribution with diagonal covariance matrix
    def __init__(self, mean, logvar=None):
        super(DiagonalGaussianDistribution, self).__init__()
        # Parameters:
        # mean: A tensor representing the mean of the distribution
        # logvar: Optional tensor representing the log of the standard variance
        #         for each of the dimensions of the distribution

        self.mean = mean
        if logvar is None:
            logvar = torch.zeros_like(self.mean)
        self.logvar = torch.clamp(logvar, -30, 20)

        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self):
        # Provide a reparameterized sample from the distribution
        # Return: Tensor of the same size as the mean
        return self.mean + self.std * torch.randn(size=self.mean.shape).to(self.mean.device)

    def kl(self):
        # Compute the KL-Divergence between the distribution with the standard normal N(0, I)
        # Return: Tensor of size (batch size,) containing the KL-Divergence for each element in the batch
        return - 1 / 2 * (1 + self.logvar - self.mean ** 2 - self.var).sum(dim=-1)

    def nll(self, sample, dims=[1, 2, 3]):
        # Computes the negative log likelihood of the sample under the given distribution
        # Return: Tensor of size (batch size,) containing the log-likelihood for each element in the batch
        log_2pi = 1.8378770664093453  # np.log(2.0 * np.pi)
        return 1 / 2 * (log_2pi + self.logvar + (sample - self.mean) ** 2 / self.var).sum(dim=dims)

    def mode(self):
        # Returns the mode of the distribution
        return self.mean


class VAE(nn.Module):
    def __init__(self, in_channels=3, decoder_features=32, encoder_features=32, z_dim=100, input_size=32,
                 device=torch.device("cuda:0")):
        super(VAE, self).__init__()

        self.z_dim = z_dim
        self.in_channels = in_channels
        self.device = device

        # Encode the Input
        self.encoder = Encoder(nc=in_channels, nef=encoder_features, nz=z_dim, isize=input_size, device=device)

        # Map the encoded feature map to the latent vector of mean, (log)variance
        out_size = input_size // 16
        self.mean = nn.Linear(encoder_features * 8 * out_size * out_size, z_dim)
        self.logvar = nn.Linear(encoder_features * 8 * out_size * out_size, z_dim)

        # Decode the Latent Representation
        self.decoder = Decoder(nc=in_channels, ndf=decoder_features, nz=z_dim, isize=input_size)

    def encode(self, x):
        # Input:
        #   x: Tensor of shape (batch_size, 3, 32, 32)
        # Returns:
        #   posterior: The posterior distribution q_\phi(z | x)
        enc = self.encoder(x)
        # enc = enc.view(enc.size(0), -1)
        mean = self.mean(enc)
        logvar = self.logvar(enc)
        return DiagonalGaussianDistribution(mean, logvar)

    def decode(self, z):
        # Input:
        #   z: Tensor of shape (batch_size, z_dim)
        # Returns
        #   conditional distribution: The likelihood distribution p_\theta(x | z)
        mean = self.decoder(z)
        return DiagonalGaussianDistribution(mean)

    def sample(self, batch_size):
        # Input:
        #   batch_size: The number of samples to generate
        # Returns:
        #   samples: Generated samples using the decoder
        #            Size: (batch_size, 3, 32, 32)
        z = torch.randn(batch_size, self.z_dim).to(self.device)
        conditional_distribution = self.decode(z)
        return conditional_distribution.mode()

    def log_likelihood(self, x, K=100):
        # Approximate the log-likelihood of the data using Importance Sampling
        # Inputs:
        #   x: Data sample tensor of shape (batch_size, 3, 32, 32)
        #   K: Number of samples to use to approximate p_\theta(x)
        # Returns:
        #   ll: Log likelihood of the sample x in the VAE model using K samples
        #       Size: (batch_size,)
        posterior = self.encode(x)
        prior = DiagonalGaussianDistribution(torch.zeros_like(posterior.mean))

        log_likelihood = torch.zeros(x.shape[0], K).to(self.device)
        for i in range(K):
            z = posterior.sample()  # (sample from q_phi)
            recon = self.decode(z)  # (decode to conditional distribution)
            log_likelihood[:, i] = posterior.nll(z, dims=-1) - recon.nll(x) - prior.nll(z, dims=-1)
            del z, recon

        return torch.logsumexp(log_likelihood, dim=-1) - np.log(K)  # (compute the final log-likelihood using the log-sum-exp trick)

    def forward(self, x):
        # Input:
        #   x: Tensor of shape (batch_size, 3, 32, 32)
        # Returns:
        #   reconstruction: The mode of the distribution p_\theta(x | z) as a candidate reconstruction
        #                   Size: (batch_size, 3, 32, 32)
        #   Conditional Negative Log-Likelihood: The negative log-likelihood of the input x under the distribution p_\theta(x | z)
        #                                         Size: (batch_size,)
        #   KL: The KL Divergence between the variational approximate posterior with N(0, I)
        #       Size: (batch_size,)
        posterior = self.encode(x)  #
        latent_z = posterior.sample()  # (sample a z)
        recon = self.decode(latent_z)  # (decode)

        return recon.mode(), recon.nll(x), posterior.kl()


# Training Hyperparameters
train_batch_size = 64  # Batch Size
z_dim = 32  # Latent Dimensionality
lr = 1e-4  # Learning Rate
# Define Dataset Statistics
image_size = 32
input_channels = 3
data_root = './data'

epochs = 85
logging_frequency = 5

if __name__ == '__main__':
    model = VAE(in_channels=input_channels,
                input_size=image_size,
                z_dim=z_dim,
                decoder_features=32,
                encoder_features=32,
                device=device
                )
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    logger = dict()
    logger['train_time'] = [0]
    logger['train_losses'] = []
    start_time = time.time()
    epoch_loss = 0
    logging_loss = 0

    data_dict = dict()
    data_dict["origs"], data_dict["origs_valid"], data_dict["recons"], data_dict["recons_valid"], data_dict["samples"] = [], [], [], [], []

    train_dataloader, testing_dataloader = get_dataloaders(data_root, batch_size=train_batch_size)
    test_iter = iter(testing_dataloader)
    for epoch in range(epochs):

        with tqdm(train_dataloader, unit="batch", leave=False) as tepoch:
            model.train()
            for batch in tepoch:
                tepoch.set_description(f"Epoch: {epoch}")
                optimizer.zero_grad()
                imgs, _ = batch
                batch_size = imgs.shape[0]
                x = imgs.to(device)
                recon, nll, kl = model(x)
                loss = (nll + kl).mean()
                logging_loss += loss.item()
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())

            new_time = time.time() - start_time
            logger['train_time'].append(new_time)
            logger['train_losses'].append(logging_loss / logging_frequency / train_batch_size)
            logging_loss = 0

            samples = model.sample(batch_size=64)
            data_dict["samples"].append(samples)
            save_image((samples + 1) * 0.5, f'./results/samples/samples_{epoch}.png')

            train_recon, nll, kl = model(x)
            data_dict["origs"].append(x), data_dict["recons"].append(train_recon)
            save_image((x + 1) * 0.5, f'./results/origs/orig_{epoch}.png')
            save_image((train_recon + 1) * 0.5, f'./results/recons/recon_{epoch}.png')

            test_batch = (next(test_iter)[0]).to(device)
            test_recon, nll, kl = model(test_batch)
            data_dict["origs_valid"].append(test_batch), data_dict["recons_valid"].append(test_recon)
            save_image((test_batch + 1) * 0.5, f'./results/origs_valid/orig_{epoch}.png')
            save_image((test_recon + 1) * 0.5, f'./results/recons_valid/recon_{epoch}.png')

            if epoch % 5 == 0:
                with open(f"results/models/model_{epoch}" + ".pickle", 'wb') as f:
                    pickle.dump(model, f)

    with open(f"results/models/model_{epoch}" + ".pickle", 'wb') as f:
        pickle.dump(model, f)
    with open(f"results/data_dict" + ".pickle", 'wb') as f:
        pickle.dump(data_dict, f)

    save_logs(logger, f"results/log")
