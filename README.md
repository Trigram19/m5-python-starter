# M5 Starter (Python Version)

Python framework for a good neural network for the Makidrakis 5 (M5) competition hosted on Kaggle.

This was originally supposed to be hosted on Google Colab but the memory and GPU are very restrictive over there. However, on GitHub it is easier for me to share my code. I hope that I will be of some use.

This is a truncated version of my model, so please bear if it scores significantly lesser.

# Directory Structure

It is structured like this:
```
m5-python-starter
|__________model.py
|__________features.py
|__________train.py
|__________data_______________
                            ||____sales_train_validation.csv
                            ||____sell_prices.csv
                            ||____sample_submission.csv
                            ||____calendar.csv
```
 
The model was trained on Google Colab and with the following parameters was able to attain a 0.53 (without overfitting):

```
Transformer(
    n_token=1000,
    n_layer=36,
    n_head=64,
    d_model=512,
    d_head=64,
    d_inner=1024,
    dropout=0.2,
    dropatt=0.2,
    dtype=torch.float32,
    attention_dropout_prob=0.15,
    output_dropout_prob=0.175,
    init_method=torch.optim.SGD,
    bi_data=10
)
```
