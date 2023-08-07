# Project for Natural Language Processing course
## Text-driven Shape generation with Denoising Diffusion Models
### Installation
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

### Architecture
The architecture of this project is inspired by [PVD](https://arxiv.org/pdf/2104.03670.pdf), a method for unconditional point cloud generation. In order to generate shapes directly from text, two conditional schemes have been implemented and evaluated:
* Concatenation of text features with point cloud features
* Cross-attention between text feature and point cloud features.

### Concatenation

### Cross-Attention

### Extra: lower resolution architecture
In addition, we have explored the possibility to reduce the resolution of the original architecture, in order to speed up training and test. 
The **original resolution** architecture we propose shares the same backbone of the original PVD method. The table below summarizes all the layers composing this model.
With the aim of reducing the training and inference time, we have reduced the resolution of the original model. In doing so, we have been able to reduce the computation time by a factor of <> during training and inference, without empairing significantly the quality of the generated output.
The table below shows the layers and parameters of this smaller architecture.

### Data
This model has been trained on [Text2Shape](http://text2shape.stanford.edu/), the only existing dataset with paired 3D shapes and textual descriptions. Such dataset is limited to the chair and table categories of ShapeNet. Text2Shape provides a total of 75k shape-text pairs, referred to 15032 distint 3D shapes.


### Training
```shell
python train.py
```

### Inference

### Experimental results

### Qualitative examples
