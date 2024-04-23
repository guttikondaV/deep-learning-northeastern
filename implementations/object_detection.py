import lightning as L
import torch
import torch.nn as nn
import torch.utils.data as data


class SimpleModel(L.LightningModule):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(5, 10), nn.Linear(10, 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc1(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.mse_loss(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


class DummyDataset(data.Dataset):
    def __init__(self) -> None:
        super(DummyDataset, self).__init__()

    def __len__(self):
        return 100

    def __getitem__(self, index):
        return torch.rand((1, 5)), torch.rand((1, 1))


dataloader = data.DataLoader(DummyDataset(), batch_size=32)

trainer = L.Trainer(max_epochs=2, logger=False)

model = SimpleModel()

trainer.fit(model, dataloader)
