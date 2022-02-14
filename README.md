# Image-to-Point cloud Auto-Encoder
2021_DX_team4 project

Multiview images to point cloud auto-encoder implementation in pytorch.

The model is composed of a pre-trained ResNet18 as an image encoder and MLPs as a point cloud decoder.

## Setup
Clone the repo.
```
$ git clone https://github.com/63days/21_DX_team4
$ cd 21_DX_team4
```

Download `data/1SET_STL` and `data/2SET_STL` and place them in the data directory.

Download libraries `pip install -r requirements.txt`

## Project Structure
```
├── src                    <- project source codes
│   ├── main.py              <- Run pipeline
│   ├── model.py             <- Model implementation code
│   ├── dataset.py           <- Dataset implementation code
│   ├── preprocess.py        <- Some codes related with preprocessing
│   └── utils.py
│
├── data
│   ├── 1SET_STL            <- validation data
│   │   ├── models.txt         <- all model names list
│   │   ├── images/            <- multiview images
│   │   ├── objs/              <- obj files for preprocessing (rendering and point sampling).
│   │   ├── points/            <- sampled point clouds
│   │   ├── stls/              <- stl files
│   │
│   └── 2SET_STL            <- train data
│       ├── ...                <- same as 1SET_STL dir   
│        
└── results              <- save training results


```
## Train
```
$ cd src
$ python main.py [--batch_size 32] [--epochs 100] [--lr 1e-3] [--gpu_num 0] [--num_points 10000]
```

