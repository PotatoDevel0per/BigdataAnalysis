import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

batch_size = 512

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc, num_classes=10):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz + num_classes, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 1, 1, 2, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        labels = self.label_emb(labels)
        labels = labels.unsqueeze(-1).unsqueeze(-1)
        gen_input = torch.cat((noise, labels), dim=1)
        return self.main(gen_input)

class Discriminator(nn.Module):
    def __init__(self, nz, ndf, nc, num_classes=10):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.main = nn.Sequential(
            nn.Conv2d(nc + num_classes, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        labels = self.label_emb(labels)
        labels = labels.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.size(2), x.size(3))
        d_input = torch.cat((x, labels), dim=1)
        return self.main(d_input).view(-1, 1)

# =======================================
# 여기서부터 device 자동 인식 구조 시작
# =======================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nz = 100
ngf = 64
ndf = 64
nc = 1

generator = Generator(nz, ngf, nc).to(device)
discriminator = Discriminator(nz, ndf, nc).to(device)

criterion = nn.BCELoss().to(device)
optimizer_G = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

num_epochs = 40

for epoch in range(num_epochs):
    g_loss_sum, d_loss_sum = 0, 0
    for i, (real_images, real_labels) in enumerate(train_dataloader):
        batch_size = real_images.size(0)

        real_label_values = torch.ones(batch_size, 1).to(device)
        fake_label_values = torch.zeros(batch_size, 1).to(device)

        real_labels = real_labels.to(device)
        real_images = real_images.to(device)

        noise = torch.randn(batch_size, nz, 1, 1).to(device)
        fake_labels = torch.randint(0, 10, (batch_size,), dtype=torch.long).to(device)

        # Train Discriminator
        discriminator.zero_grad()
        real_outputs = discriminator(real_images, real_labels)
        real_loss = criterion(real_outputs, real_label_values)
        real_loss.backward()

        fake_images = generator(noise, fake_labels)
        fake_outputs = discriminator(fake_images.detach(), fake_labels)
        fake_loss = criterion(fake_outputs, fake_label_values)
        fake_loss.backward()

        d_loss = real_loss + fake_loss
        optimizer_D.step()
        d_loss_sum += d_loss.item()

        # Train Generator
        generator.zero_grad()
        fake_outputs = discriminator(fake_images, fake_labels)
        g_loss = criterion(fake_outputs, real_label_values)
        g_loss.backward()
        optimizer_G.step()
        g_loss_sum += g_loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}] D_loss: {d_loss_sum / len(train_dataloader):.8f}, G_loss: {g_loss_sum / len(train_dataloader):.8f}")

# 이미지 생성 및 시각화
num_samples = 25
num_idx = 6
sample_noise = torch.randn(num_samples, nz, 1, 1).to(device)
gen_labels = torch.randint(num_idx, num_idx+1, (num_samples,), dtype=torch.long).to(device)
generated_images = generator(sample_noise, gen_labels).detach().cpu()

fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(generated_images[i].squeeze(), cmap='gray')
    ax.axis('off')
plt.show()
