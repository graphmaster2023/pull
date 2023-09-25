# ACCURATE LINK PREDICTION VIA PU-LEARNING

This is the code repository for Accurate link prediction via PU-learning.
This includes the implementation of PULL (**PU**-**L**earning-based **L**ink prediction), our novel
approach for PU learning to solve the link prediction problem achieves the state-of-the-art performance on public graph datasets.

## Requirements

We recommend using the following versions of packages:
- `python==3.7.13`
- `cuda==11.6`
- `cudnn==8.5.0`
- `pytorch==1.13.1`
- `torch-geometric==2.3.1`

## Code Description
- `src/models.py` implements for the GCN Link Predictor.
- `src/train.py` contains functions for training the link predictor via PU-learning.
- `main.py` is the main script for training our link predictor for a graph dataset.

## Data Overview
| **Dataset**    |           **Path or Package**            | 
|:--------------:|:----------------------------------------:| 
|   **PubMed**     | `torch_geometric.datasets.CitationFull`     | 
| **Cora-full**   | `torch_geometric.datasets.CitationFull`     | 
| **Chameleon**     | `torch_goemetric.datasets.WikipediaNetwork` | 
| **Crocodile**     | `torch_goemetric.datasets.WikipediaNetwork` | 
| **Facebook**     | `torch_goemetric.datasets.FacebookPagePage` | 

We load public datasets from the Torch Geometric package. 

## How to Run
You can run the demo script in the directory by `bash run.sh`. It produces the following results: 

| **Dataset**  | **AUC (Valid)** |**AUC (Test)**| 
|:------------:|:----------:|:----------:|
| **PubMed**     | 96.7        | 96.6       |
| **Cora-full**  | 96.3        | 95.9       |
| **Chameleon**  | 98.0        | 98.1       |
| **Crocodile**  | 98.5        | 98.6       |
| **Facebook**   | 97.5        | 97.3       |

You can reproduce the experimental results in the paper with the following commands:
```shell
python main.py --data PubMed --epoch 10 --val-ratio 0.1 --test-ratio 0.1
python main.py --data Cora_full --epoch 10 --val-ratio 0.1 --test-ratio 0.1
python main.py --data chameleon --epoch 10 --val-ratio 0.1 --test-ratio 0.1
python main.py --data crocodile --epoch 10 --val-ratio 0.1 --test-ratio 0.1
python main.py --data FacebookPagePage --epoch 10 --val-ratio 0.1 --test-ratio 0.1
```

Hyperparameters for the main script are summarized as follows:
- `gpu`: index of a GPU to use.
- `seed`: a random seed (any integer).
- `data`: name of a dataset.
- `epochs`: number of epochs to train.
- `val-ratio`: ratio of edges to use in validation.
- `test-ratio`: ratio of edges to use in test.
- `verbose`: print details while running the experiment if set to 'y'.
- `early-stop`: patience number for early stop.
- `n_hops`: number of hops for the GCN link predictor.
- `layer`: number of layers in GCN link predictor.
- `units`: number of units in GCN link predictor.
