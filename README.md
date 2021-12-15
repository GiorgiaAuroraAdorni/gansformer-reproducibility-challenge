# gansformer-reproducibility-challenge

> Project for Advance Topic in Machine Learning course @ USI 21/22.  
> See https://github.com/GiorgiaAuroraAdorni/gansformer-reproducibility-challenge, https://drive.google.com/drive/folders/1sqHD-X4mLOOkoT-xJvWGdPlwxb5et0kA?usp=sharing for datasets and https://drive.google.com/drive/folders/1ZFfO4HVINH-aDQbgLscJxNTqGLEIOMZv?usp=sharing for models.

### Contributors

**Giorgia Adorni** — giorgia.adorni@usi.ch [GiorgiaAuroraAdorni](https://github.com/GiorgiaAuroraAdorni)

**Felix Boelter** — felix.boelter@usi.ch [felixboelter](https://github.com/felixboelter)

**Stefano Carlo Lambertenghi** — stefano.carlo.lambertenghi@usi.ch [steflamb](https://github.com/steflamb)

### Prerequisites

- Python 3
- Tensorflow 1.X

### Installation

Clone our repository and install the requirements

```sh
$ git clone https://github.com/GiorgiaAuroraAdorni/gansformer-reproducibility-challenge
$ cd gansformer-reproducibility-challenge/src
$ pip install -r requirements.txt
```

#### Usage

For the usage, go to the `colab notebooks` directory: 
- Run `Reproducibility_model_trainer.ipynb` for training the models: Stylegan2, GANformers with Simplex and Duplex Attention and GANformers with Simplex and Duplex Attention (with vanilla StyleGAN2 discriminator).  
- Run `Reproducibility_result_visualizer.ipynb` for the visualisation phase: here you can select the model that you want to use and generate random images, perform a symple interpolation of the latent space or even perform style mixing starting from a chosen target image.
