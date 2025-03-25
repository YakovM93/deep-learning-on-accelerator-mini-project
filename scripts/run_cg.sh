PATH_TO_DATA="./data"
MNIST_RESULTS_CG="./results/MNIST/CG"
CIFAR_RESULTS_CG="./results/CIFAR/CG"

mkdir -p results/CIFAR/CG
mkdir -p results/MNIST/CG

echo "Start Training MNIST"


EPOCHS_CG_CIFAR=60      # Joint Autoencoder and Classifier  CIFAR
EPOCHS_CG_MNIST=30      # Joint Autoencoder and Classifier  MNIST
