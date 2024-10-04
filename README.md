# FDT4SR
The detailed modules code will be published after the paper is received.
# Implementation
## Environment
```
python 3.8.16
torch 1.13.1+cu117
```
## Training
```
python main.py \
--data_name [DATA_NAME] \
--hidden_size 200 \
--attention_probs_dropout_prob 0.5 \
--hidden_dropout_prob 0.5 \
--ss_probability 0.05\
--lr 0.0001
```

## Fine-Tuning
If you use your own environment or dataset, we provide some suggestions and ranges for fine-tuning of hyper-parameters.
* num_hidden_layers ∈ {1,2,3}
* num_attention_heads ∈ {2,4}
* dropout rate ∈ {0.3,0.5,0.7}
* learning rate ∈ {0.001,0.0001}
* ss_probability ∈ [0,0.1], step by 0.01
## Reproduce
We give the model files trained on the Home, Sports, Beauty, Tools, Office, and LastFM datasets in folder 'reproduction'.
```
python main.py --data_name=[DATA_NAME] --checkpoint_path="./reproduction/[DATA_NAME].pt" --do eval
```
