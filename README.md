# Data and Code for "Word Relation Networks for Fake News Detection"


## Data
Our work is based on the `PolitiFact` and `GossipCop` datasets from the [FakeNewsNet benchmark](https://github.com/KaiDMML/FakeNewsNet),
We provide the data files utilized for training and evaluating WRGNN under `data/`. 

##Original Unaltered Training / Test Articles

The `.pkl` files under `data/news_articles/` contain the unaltered news article texts. 

##Adversarial Test Sets

The .pkl files under data/adversarial_test/ contain the four adversarial test sets under LLM-empowered style attacks, denoted as A through D. 

## Run DCGCN
 

Start training with the following command:

python src/Bert_WRGNN.py --dataset_name politifact

[--dataset_name]: politifact / gossipcop 

## Acknowledgments

We have adopted the project structure and Training code and data from: https://github.com/jiayingwu19/SheepDog/tree/main
