import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# !pip install pennylane
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from classical_discriminator import Discriminator
from dataloader import DigitsDataset
from quantum_generator import QuantumGenerator

constants = {
    "BATCH_SIZE": 1,
    "IMAGE_SIZE": 8
}

quantum_params = {
    "n_qubits": 5,
    "n_a_qubits": 1,
    "q_depth": 6,
    "n_generators": 4
}

train_params = {
    'learnrate_gen': 0.3,
    'learnrate_discrim': 0.01,
    'num_iter': 500
}

# Set the random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# create dataloader instance
transform_to_pytorch_tensor = transforms.Compose([transforms.ToTensor()])
dataset = DigitsDataset(
    filepath = "MNIST_images.tra",
    transform = transform_to_pytorch_tensor
)

# Create a DataLoader for the dataset
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size = constants["BATCH_SIZE"],
    shuffle = True,
    drop_last = True
)

# define the quantum device
dev = qml.device("lightning.qubit", wires=quantum_params["n_qubits"])
# Enable CUDA device if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define the generator and discriminator
discriminator = Discriminator().to(device)
generator = QuantumGenerator(quantum_params["n_generators"]).to(device)

# Binary cross entropy
criterion = nn.BCELoss()

# Optimisers - stochastic grad descent
optD = optim.SGD(discriminator.parameters(), lr=train_params['learnrate_discrim'])
optG = optim.SGD(generator.parameters(), lr=train_params['learnrate_gen'])

real_labels = torch.full((constants["BATCH_SIZE"],), 1.0, dtype=torch.float, device=device)
fake_labels = torch.full((constants["BATCH_SIZE"],), 0.0, dtype=torch.float, device=device)

# Fixed noise allows us to visually track the generated images throughout training
fixed_noise = torch.rand(8, quantum_params["n_qubits"], device=device) * math.pi / 2

# Iteration counter
counter = 0

# Collect images for plotting later
results = []

while True:
    for i, (data, _) in enumerate(dataloader):

        # Data for training the discriminator
        data = data.reshape(-1, constants["IMAGE_SIZE"] * constants["IMAGE_SIZE"])
        real_data = data.to(device)

        # Noise follwing a uniform distribution in range [0,pi/2)
        noise = torch.rand(constants["BATCH_SIZE"], quantum_params["n_qubits"], device=device) * math.pi / 2
        fake_data = generator(noise)

        # Training the discriminator
        discriminator.zero_grad()
        outD_real = discriminator(real_data).view(-1)
        outD_fake = discriminator(fake_data.detach()).view(-1)

        errD_real = criterion(outD_real, real_labels)
        errD_fake = criterion(outD_fake, fake_labels)
        # Propagate gradients
        errD_real.backward()
        errD_fake.backward()

        errD = errD_real + errD_fake
        optD.step()

        # Training the generator
        generator.zero_grad()
        outD_fake = discriminator(fake_data).view(-1)
        errG = criterion(outD_fake, real_labels)
        errG.backward()
        optG.step()

        counter += 1

        # Show loss values
        if counter % 10 == 0:
            print(f'Iteration: {counter}, Discriminator Loss: {errD:0.3f}, Generator Loss: {errG:0.3f}')
            test_images = generator(fixed_noise).view(8,1,constants["IMAGE_SIZE"],constants["IMAGE_SIZE"]).cpu().detach()

            # Save images every 50 iterations
            if counter % 50 == 0:
                results.append(test_images)

        if counter == train_params['num_iter']:
            break
    if counter == train_params['num_iter']:
        break

fig = plt.figure(figsize=(10, 5))
outer = gridspec.GridSpec(5, 2, wspace=0.1)

for i, images in enumerate(results):
    inner = gridspec.GridSpecFromSubplotSpec(1, images.size(0),
                    subplot_spec=outer[i])

    images = torch.squeeze(images, dim=1)
    for j, im in enumerate(images):

        ax = plt.Subplot(fig, inner[j])
        ax.imshow(im.numpy(), cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        if j==0:
            ax.set_title(f'Iteration {50+i*50}', loc='left')
        fig.add_subplot(ax)

plt.show()