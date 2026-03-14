# Recurrent Neural Network — MNIST Digit Classification

A vanilla RNN trained on the MNIST handwritten-digit dataset using PyTorch.  
Each 28×28 image is treated as a sequence of 28 time-steps (rows), each with 28 features (pixels).

## How It Works

| Component | Detail |
|-----------|--------|
| Architecture | 2-layer stacked RNN → fully-connected output |
| Input | 28 time-steps × 28 features (one row per step) |
| Hidden size | 100 |
| Optimizer | SGD (lr = 0.1) |
| Loss | Cross-Entropy |
| Dataset | MNIST (auto-downloaded via `torchvision`) |

The model reads each image row-by-row, builds a hidden representation across 28 steps, and classifies the final hidden state into one of 10 digit classes.

## 🛠 Tech Stack

| | Tool | Purpose |
|---|------|---------|
| 🐍 | Python 3.8+ | Language |
| 🔥 | PyTorch | Deep learning framework |
| 🖼 | torchvision | MNIST dataset & transforms |

## Getting Started

```bash
# Install dependencies
pip install torch torchvision

# Train the model
python rnn.py
```

MNIST is downloaded automatically on first run into a `./data` directory.

## Results

The model reaches ~96 % test accuracy after 3 000 iterations with a 2-layer RNN.

![results](https://image.ibb.co/h2Xga7/Screen_Shot_2018_03_09_at_2_08_23_PM.png)

## ⚠️ Known Issues

- A vanilla RNN struggles with longer sequences; consider LSTM/GRU for harder tasks.
- No learning-rate scheduler or early stopping is used.

## License

MIT
