# Robustness Certificates for Sparse Adversarial Attacks by Randomized Ablation
Code for the paper "Robustness Certificates for Sparse Adversarial Attacks by Randomized Ablation" by Alexander Levine and Soheil Feizi.
Files are provided for training and evaluation of certifiably robust classifiers, robust to L_0 attacks, on MNIST, CIFAR-10, and ImageNet datasets.

On MNIST and CIFAR-10, there are two architectures provided, which differ in how they encode ablated (NULL) pixels: a standard (multichannel) architecture, and an architecture which encodes NULL as the mean value on the dataset. Files for the mean encoding have their names suffixed with `_mean`.

Code should run with Python 3.7 and PyTorch 1.2.0.

Explanation of files: (substitute `mnist` for `cifar` or `imagenet` appropriately)

- `train_mnist.py` will train the base classifier on ablated images, and save the model to the `checkpoints` directory

- `mnist_certify.py` will load a model from `checkpoints`, certify the robustness of images from the test set, and save the list of robustness certificates to the `radii` directory as a PyTorch save file (`.pth`)

- `mnist_predict.py` Will load a model from `checkpoints`, evaluate the prediction accuracy on images from the test set, and save summary statistics to the `accuracies` directory as a PyTorch save file (`.pth`)

Example Usage: (training MNIST with 45 retained pixels)

```
python3 train_mnist.py --keep 45 --lr 0.01
python3 train_mnist.py --keep 45 --lr 0.001 --resume mnist_lr_0.01_keep_45_epoch_199.pth
python3 mnist_certify.py --keep 45 --model mnist_lr_0.001_keep_45_epoch_399_resume_mnist_lr_0.01_keep_45_epoch_199.pth.pth
python3 mnist_predict.py --keep 45 --model mnist_lr_0.001_keep_45_epoch_399_resume_mnist_lr_0.01_keep_45_epoch_199.pth.pth
```

Caveats:

- `imagenet` files expect the ILSVRC2012 training and validation sets to be in the directories `imagenet-train/train` and `imagenet-val/val`, respectively. This can be changed using the `--trainpath` and `--valpath` options.

- While `train_cifar.py` has the option of training a ResNet50 model, `cifar_certify.py` and `cifar_predict.py` are hardcoded to use ResNet18.

Adversarial Attack Tests: for MNIST only, there is code to attack the robust model using the Pointwise attack from FoolBox:

- `mnist_test_pointwise.py` will load a model from 'checkpoints', perform the Pointwise attack on images from the mnist test set, and save data on each attacked image as a separate file  to to the `empirical` directory as PyTorch save files (`.pth`)

Attributions:
- Code in the `pytorch_cifar` directory is from https://github.com/kuangliu/pytorch-cifar, with slight modification to allow for 6-channel input.
- The file `resnet_imgnt.py` is modified from the PyTorch torchvision implementation of ResNet (https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py), again with slight modification to allow for 6-channel input.

