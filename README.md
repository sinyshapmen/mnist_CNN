# 🧠 Digit Recognition MNIST CNN

Modular CNN pipeline for classifying handwritten digits from the MNIST dataset. The project includes training, evaluation, logging, testing, and Docker support

---

## 📦 Features
- Train & evaluate MNIST CNN
- Save best model weights
- TensorBoard support
- Configurable via `config.yaml`
- Inference on a random test digit
- Docker container 

---

## 🚀 Setup

```bash
git clone https://github.com/sinyshapmen/mnist_CNN
cd digit-recognition-cnn

## Local setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Docker image
docker build -t digit-cnn .

# Train the model
docker run digit-cnn python main.py --mode train
tensorboard --logdir=runs
# Evaluate the model
docker run digit-cnn python main.py --mode eval
# Test inference on a random digit
docker run -it digit-cnn python test_model.py
```

## 📁 Project Structure

```bash
digit-recognition-cnn/
│
├── config.yaml              # Training and path configuration
├── main.py                  # Entry point for training/eval
├── test_model.py            # Test single image inference
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker image setup
│
├── data/                    # Data loading logic
│   └── load_data.py
│
├── src/                     # Core source files
│   ├── model.py             # CNN architecture
│   ├── train.py             # Trainer class
│   ├── inference.py         # Inference wrapper
│   ├── metrics.py           # Accuracy, precision, recall, F1
│   └── logger.py            # Logging setup
│
├── weights/                 # Stores best_model.pth after training
├── logs/                    # Logs for training and main
└── runs/                    # TensorBoard log directory
```




