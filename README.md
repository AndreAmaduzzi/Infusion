# Text-driven Shape generation with Denoising Diffusion Models
## Installation
Python 3.8 is required.

Create and activate a virtual environment
```console
python -m venv env
source env/bin/activate
```

Install all the dependencies
```console
pip install -r requirements.txt
```

# Architecture
The architecture of this project is inspired by [PVD](https://arxiv.org/pdf/2104.03670.pdf), a method for unconditional point cloud generation. In order to generate shapes directly from text, two conditional schemes have been implemented and evaluated:
* Concatenation of text features with point cloud features
* Cross-attention between text feature and point cloud features.

In addition, we have explored the possibility to reduce the resolution of the original architecture, in order to speed up training and test. 
## Higher-resolution architecture
The higher-resolution architecture we propose shares the same backbone of the original PVD method. The table below summarizes all the layers composing this model.
## Lower-resolution architecture
With the aim of reducing the training and inference time, we have reduced the resolution of the original model. In doing so, we have been able to reduce the computation time by a factor of <> during training and inference, without empairing significantly the quality of the generated output.
The table below shows the layers and parameters of this smaller architecture.
## Training
## Inference
