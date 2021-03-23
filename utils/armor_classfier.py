import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

pl.seed_everything(42)


class ArmorClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2, 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4, 6, 4, 2, 1, bias=False),
            nn.BatchNorm2d(6),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(6, 10, 3, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.cnn(x)
        return y.view(-1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self.cnn(x)
        loss = F.mse_loss(F.one_hot(y, num_classes=10).float(), y_hat.view(-1, 10))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        with torch.no_grad():
            y_hat = self.cnn(x)
            loss = F.mse_loss(F.one_hot(y, num_classes=10).float(), y_hat.view(-1, 10))
        self.log("val_loss", loss)


# data
# train_trans = transforms.Compose(
#     [
#         transforms.RandomPerspective(),
#         transforms.RandomRotation(),
#         transforms.RandomAffine(),
#         transforms.GaussianBlur(),
#         transforms.RandomErasing(),
#         transforms.ToTensor(),
#     ]
# )

# dataset = datasets.ImageFolder("~rm")
dataset = datasets.MNIST(
    "~/dataset", train=True, download=True, transform=transforms.ToTensor()
)

mnist_train, mnist_val = random_split(dataset, [55000, 5000])

train_loader = DataLoader(mnist_train, batch_size=32, num_workers=8, pin_memory=True)
val_loader = DataLoader(mnist_val, batch_size=32, num_workers=8, pin_memory=True)

# model
model = ArmorClassifier()

# training
trainer = pl.Trainer(
    gpus=1,
    num_nodes=1,
    precision=16,
    gradient_clip_val=0.5,
    stochastic_weight_avg=True,
    # fast_dev_run=True,
    # overfit_batches=0.01,
    # limit_train_batches=0.1,
    # limit_val_batches=0.2,
    # limit_test_batches=0.3,
    weights_summary="full",
    callbacks=[
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            verbose=True,
            mode="min",
        )
    ],
)

trainer.fit(model, train_loader, val_loader)

# ----------------------------------
# onnx
# ----------------------------------
filepath = "armor_classifier.onnx"
input_sample = torch.randn((1, 1, 28, 28))
model.to_onnx(filepath, input_sample)
