# StyleGAN-Inspired CIFAR-10 Image Generation

A compact, educational implementation of a **StyleGAN-inspired image
generator** built with TensorFlow and Keras. The project trains on a
subset of CIFAR-10 and demonstrates style-based image synthesis, style
mixing, and latent-space interpolation.

> **Note:** This is a small teaching implementation inspired by
> StyleGAN, not a reproduction of the original high-resolution StyleGAN
> architecture. Generated images are 32×32 pixels and may appear blurry.

## Project Overview

The notebook demonstrates how a generator can learn to produce synthetic
images through adversarial training. It includes:

-   CIFAR-10 dataset loading and preprocessing
-   A mapping network that transforms latent vectors from **Z-space** to
    **W-space**
-   A style-based synthesis network with a learned constant
-   Adaptive Instance Normalization (AdaIN)
-   Noise injection for stochastic visual details
-   A convolutional discriminator
-   Generator and discriminator loss tracking
-   Style mixing between two latent codes
-   Latent interpolation between generated images

## Technologies Used

-   Python
-   TensorFlow
-   Keras
-   NumPy
-   Matplotlib
-   CIFAR-10 dataset

## Dataset

The project uses the CIFAR-10 dataset, which contains 32×32 RGB images
across 10 classes, including airplanes, automobiles, birds, cats, deer,
dogs, frogs, horses, ships, and trucks.

For a manageable training run, the notebook uses the first **20,000
training images**. Pixel values are normalized to the range `[-1, 1]`.

## Model Architecture

### Generator

1.  **Mapping network:** Eight dense layers transform a random latent
    vector into an intermediate style representation.
2.  **Synthesis network:** Starts from a learned 4×4 constant and
    progressively generates feature maps at 4×4, 8×8, 16×16, and 32×32
    resolutions.
3.  **AdaIN:** Modulates feature-map statistics using style-derived
    scale and bias values.
4.  **Noise injection:** Adds noise at synthesis layers to introduce
    stochastic details.
5.  **Output layer:** Produces a 32×32 RGB image.

### Discriminator

A convolutional discriminator evaluates real CIFAR-10 images and
generated images. Its feedback trains the generator to produce more
realistic samples.

## Training Configuration

  Setting             Value
  ------------------- ------------------------------
  Dataset             CIFAR-10
  Training subset     20,000 images
  Image resolution    32×32 RGB
  Batch size          64
  Epochs              10
  Z-space dimension   128
  W-space dimension   128
  Optimizer           Adam
  Learning rate       0.0002
  GAN objective       Non-saturating logistic loss
  Random seed         42

Actual training time depends on the runtime and available hardware.

## How to Run

### Option 1: Google Colab

1.  Open the notebook in Google Colab.
2.  Select **Runtime → Run all**.
3.  Allow the notebook to download CIFAR-10 when prompted.
4.  Wait for training to finish and inspect the output plots.

### Option 2: Local Environment

Install the required packages:

``` bash
pip install tensorflow numpy matplotlib
```

Then open and run `StyleGAN_ipynb_txt.ipynb` in Jupyter Notebook or
JupyterLab.

## Outputs

The notebook produces:

-   Examples of real CIFAR-10 training images
-   Generator and discriminator loss curves
-   A grid of generated images after training
-   A style-mixing comparison using different layer cutoffs
-   A latent-interpolation sequence showing transitions between two
    latent vectors

## Results and Limitations

In the included experiment, the model is trained for 10 epochs on 20,000
CIFAR-10 images. The generated samples learn some color and texture
patterns, but they remain blurry and do not consistently form
recognizable objects.

This is an educational experiment. It demonstrates key StyleGAN-inspired
concepts, but it does not claim the image quality or performance of the
original StyleGAN. Better results would generally require more training,
tuning, and computational resources.

## Future Improvements

-   Train for more epochs and evaluate different hyperparameters.
-   Experiment with model capacity and regularization.
-   Save generated samples and model checkpoints during training.
-   Compare results across different random seeds.
-   Explore higher-resolution datasets and more complete StyleGAN
    techniques.

## References

-   Karras, T., Laine, S., and Aila, T. **A Style-Based Generator
    Architecture for Generative Adversarial Networks.** ICLR, 2019.
-   [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
-   [TensorFlow](https://www.tensorflow.org/)
