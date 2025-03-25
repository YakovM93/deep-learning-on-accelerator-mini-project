## Script to run training of all models on both dataset
## and produce results and plots

# Set Variables

SEED=0
## run the script with --test to run a quick test
if [[ "$@" == *"--test"* ]]; then
    echo "Testing Mode!"
    EPOCHS_AE_MNIST=1
    EPOCHS_CL_MNIST=1
    EPOCHS_CG_MNIST=1

    EPOCHS_AE_CIFAR=1
    EPOCHS_CL_CIFAR=1
    EPOCHS_CG_CIFAR=1
else
    EPOCHS_AE_MNIST=30      # Autoencoder Self Supervised       MNIST
    EPOCHS_CL_MNIST=30     # Classifier with fromzen encoder   MNIST
    EPOCHS_CG_MNIST=30      # Joint Autoencoder and Classifier  MNIST

    EPOCHS_AE_CIFAR=50      # Autoencoder Self Supervised       CIFAR
    EPOCHS_CL_CIFAR=180     # Classifier with fromzen encoder   CIFAR
    EPOCHS_CG_CIFAR=60      # Joint Autoencoder and Classifier  CIFAR
fi

PATH_TO_DATA="./data"

MNIST_RESULTS_AE_CL="./results/MNIST/AE_CL"
MNIST_RESULTS_CG="./results/MNIST/CG"
MNIST_RESULTS_CONTRASTIVE="./results/MNIST/CONTRASTIVE"

CIFAR_RESULTS_AE_CL="./results/CIFAR/AE_CL"
CIFAR_RESULTS_CG="./results/CIFAR/CG"
CIFAR_RESULTS_CONTRASTIVE="./results/CIFAR/CONTRASTIVE"

# prepare
mkdir -p results/CIFAR/AE_CL
mkdir -p results/CIFAR/CG
mkdir -p results/CIFAR/CONTRASTIVE

mkdir -p results/MNIST/AE_CL
mkdir -p results/MNIST/CG
mkdir -p results/MNIST/CONTRASTIVE

echo "Training Epochs:"
echo "---------------"
echo "MNIST"
echo "   Autoencoder:       $EPOCHS_AE_MNIST"
echo "   Classifier:        $EPOCHS_CL_MNIST"
echo "   Joint Training:    $EPOCHS_CG_MNIST"
echo
echo "CIFAR"
echo "   Autoencoder:       $EPOCHS_AE_CIFAR"
echo "   Classifier:        $EPOCHS_CL_CIFAR"
echo "   Joint Training:    $EPOCHS_CG_CIFAR"

echo
echo "########################"
echo "Starting training..."
echo "########################"
echo

sleep 3

# # Self Supervised (1.2.1)
echo "========================"
echo "Self Supervised:"
echo "========================"

sleep 1

echo
echo "== Self Supervised: MNIST =="
echo
# 1
python src/mini_project/main.py --save-path $MNIST_RESULTS_AE_CL --mnist --mode self_supervised --epochs-ae $EPOCHS_AE_MNIST --epochs-clf $EPOCHS_CL_MNIST --seed $SEED

echo
echo "== Self Supervised: CIFAR =="
echo
# 2
python src/mini_project/main.py --save-path $CIFAR_RESULTS_AE_CL --mode self_supervised --epochs-ae $EPOCHS_AE_CIFAR --epochs-clf $EPOCHS_CL_CIFAR  --seed $SEED

# Joint Classifier training (1.2.2)

echo
echo "========================"
echo "Joint Classifier training:"
echo "========================"
echo
sleep 1

echo
echo "=== Joint Classifier training: MNIST ==="
echo

# 3
python src/mini_project/main.py --save-path $MNIST_RESULTS_CG --mnist --mode classification_guided --epochs-cg $EPOCHS_CG_MNIST --seed $SEED

echo
echo "=== Joint Classifier training: CIFAR ==="
echo

# 4
python src/mini_project/main.py --save-path $CIFAR_RESULTS_CG --mode classification_guided --epochs-cg $EPOCHS_CG_CIFAR --seed $SEED

# Contrastive Learning (1.2.3)
echo
echo "========================"
echo "Contrastive Learning:"
echo "========================"
echo
sleep 1

echo
echo "=== Contrastive Learning: MNIST ==="
echo
# 5
python src/mini_project/main_contrastive.py --save-path $MNIST_RESULTS_CONTRASTIVE --mnist  --epochs-contrastive $EPOCHS_AE_MNIST --epochs-clf $EPOCHS_CL_MNIST --seed $SEED

echo
echo "=== Contrastive Learning: CIFAR ==="
echo
# 6
python src/mini_project/main_contrastive.py --save-path $CIFAR_RESULTS_CONTRASTIVE --mode contrastive --epochs-contrastive $EPOCHS_AE_CIFAR --epochs-clf $EPOCHS_CL_CIFAR  --seed $SEED

echo
echo "==================="
echo "Finished training!"
echo "==================="
echo
