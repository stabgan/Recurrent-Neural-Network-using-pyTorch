import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets


class RNNModel(nn.Module):
    """RNN classifier for MNIST digits.

    Treats each 28×28 image as a sequence of 28 time-steps,
    each with 28 features, and classifies into 10 digit classes.
    """

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.rnn = nn.RNN(
            input_dim, hidden_dim, layer_dim,
            batch_first=True, nonlinearity="tanh",
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(
            self.layer_dim, x.size(0), self.hidden_dim,
            device=x.device,
        )

        out, _ = self.rnn(x, h0)
        # Use the hidden state from the last time-step
        out = self.fc(out[:, -1, :])
        return out


def main():
    # ── 1. Load dataset ──────────────────────────────────────────────
    train_dataset = dsets.MNIST(
        root="./data", train=True,
        transform=transforms.ToTensor(), download=True,
    )
    test_dataset = dsets.MNIST(
        root="./data", train=False,
        transform=transforms.ToTensor(),
    )

    # ── 2. Data loaders ──────────────────────────────────────────────
    batch_size = 100
    n_iters = 3000
    num_epochs = n_iters // (len(train_dataset) // batch_size)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False,
    )

    # ── 3. Model, loss, optimizer ────────────────────────────────────
    input_dim = 28
    hidden_dim = 100
    layer_dim = 2       # two stacked RNN layers
    output_dim = 10
    seq_dim = 28         # number of time-steps to unroll

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim).to(device)

    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # ── 4. Training loop ─────────────────────────────────────────────
    iteration = 0
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, seq_dim, input_dim).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            iteration += 1

            if iteration % 500 == 0:
                # ── Evaluate on test set ─────────────────────────────
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for test_images, test_labels in test_loader:
                        test_images = test_images.view(-1, seq_dim, input_dim).to(device)
                        test_labels = test_labels.to(device)

                        test_outputs = model(test_images)
                        _, predicted = torch.max(test_outputs, 1)

                        total += test_labels.size(0)
                        correct += (predicted == test_labels).sum().item()

                accuracy = 100.0 * correct / total
                print(
                    f"Iteration: {iteration}. "
                    f"Loss: {loss.item():.4f}. "
                    f"Accuracy: {accuracy:.2f}%"
                )
                model.train()  # switch back to training mode


if __name__ == "__main__":
    main()
