# 🔁 Recurrent Neural Network (RNN) — MNIST Digit Classification

A vanilla RNN built with PyTorch that classifies handwritten digits from the MNIST dataset. The network treats each 28×28 image as a sequence of 28 time steps (rows), each with 28 features (pixels), making it a natural fit for recurrent processing.

## 🏗️ Architecture

| Component | Details |
|-----------|---------|
| Model | Multi-layer vanilla RNN (`nn.RNN`) |
| Input | 28 × 28 MNIST images (28 time steps, 28 features) |
| Hidden layers | 2 stacked RNN layers, 100 hidden units each |
| Activation | Tanh (default RNN nonlinearity) |
| Readout | Fully connected layer → 10 classes |
| Loss | Cross-Entropy Loss |
| Optimizer | SGD (lr = 0.1) |

```
Input (28×28) → RNN (2 layers, 100 hidden) → FC (100→10) → Softmax → Prediction
```

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| 🐍 Python 3 | Language |
| 🔥 PyTorch | Deep learning framework |
| 🖼️ torchvision | MNIST dataset & transforms |

## 📦 Dependencies

```bash
pip install torch torchvision
```

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/stabgan/Recurrent-Neural-Network-using-pyTorch.git
cd Recurrent-Neural-Network-using-pyTorch

# Install dependencies
pip install torch torchvision

# Train & evaluate
python rnn.py
```

MNIST data is downloaded automatically on first run into `./data/`.

The script trains for ~5 epochs (3000 iterations) and prints test accuracy every 500 iterations. GPU is used automatically when available.

## 📊 Results

![results](https://image.ibb.co/h2Xga7/Screen_Shot_2018_03_09_at_2_08_23_PM.png)

## ⚠️ Known Issues

- The vanilla RNN suffers from vanishing gradients on longer sequences — consider switching to LSTM or GRU for better performance.
- The model is defined and trained at module level (no `if __name__ == "__main__"` guard), so importing `rnn.py` will trigger training.
- No checkpointing — training restarts from scratch every run.
- The results screenshot above is from an earlier version and may not match current output exactly.

## 📄 License

[MIT](LICENSE)
