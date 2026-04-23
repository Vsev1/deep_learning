import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize, Compose

class FashionMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.features(x)


class Trainer:
    def __init__(self, model, device, loss_fn, optimizer):
        self.model = model.to(device)
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train_epoch(self, loader):
        self.model.train()
        total = len(loader.dataset)

        for i, (x, y) in enumerate(loader):
            x, y = x.to(self.device), y.to(self.device)

            preds = self.model(x)
            loss = self.loss_fn(preds, y)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if i % 100 == 0:
                done = i * len(x)
                print(f"[train] loss={loss.item():.4f}  {done}/{total}")

    def evaluate(self, loader):
        self.model.eval()

        correct = 0
        total_loss = 0
        n_batches = len(loader)

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)

                preds = self.model(x)
                total_loss += self.loss_fn(preds, y).item()

                correct += (preds.argmax(1) == y).sum().item()

        acc = 100 * correct / len(loader.dataset)
        avg_loss = total_loss / n_batches

        print(f"[test] accuracy={acc:.2f}%  loss={avg_loss:.4f}")


def main():
    train_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=Compose([
            ToTensor(),
            Normalize((0.5,), (0.5,))
        ])
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=Compose([
            ToTensor(),
            Normalize((0.5,), (0.5,))
        ])
    )

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FashionMLP()
    loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    trainer = Trainer(model, device, loss_fn, optimizer)

    epochs = 10
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}")
        trainer.train_epoch(train_loader)
        trainer.evaluate(test_loader)

    torch.save(model.state_dict(), "model_weights.pth")
    print(f"\nModel weights saved to model_weights.pth")


if __name__ == "__main__":
    main()