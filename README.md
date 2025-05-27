# DL-CS6910: Assignment3

![Image](https://media.licdn.com/dms/image/D5612AQGAhR4CtFg3Vw/article-cover_image-shrink_720_1280/0/1703904607394?e=2147483647&v=beta&t=S_16IBUesTVZYDd8h1tJCWS_CZwC5jLrLafoUNuJMUw)

The CS6910_Assignment3 repository contains code and documentation related to the assignment on sequence-to-sequence learning using Recurrent Neural Networks (RNNs), attention mechanisms, and the importance of Transformers in machine transliteration and natural language processing (NLP).

## Problem Statement

The assignment aims to address the following objectives:

1. **Sequence-to-Sequence Learning**: Implement sequence-to-sequence learning using RNNs.
2. **Comparison of RNN Cells**: Compare different RNN cells such as vanilla RNN, LSTM, and GRU.
3. **Attention Mechanisms**: Understand how attention networks overcome the limitations of vanilla seq2seq models.


## Process

The project follows a structured process:

1. **Data Preparation**: The dataset consists of pairs of words in native and Latin scripts. A class `Lang` is used to convert characters into numbers and vice versa. Additionally, a function `Tensorpair` converts words into numerical representations and creates torch tensors.

2. **Model Architecture**: The project includes classes for Encoder and two variants of the Decoder: one with attention and one without attention. The Encoder encodes input sequences, while the Decoder generates output sequences. 

3. **Training Function**: The `train_model` function trains the model using the specified hyperparameters and training configurations. It utilizes PyTorch's DataLoader for efficient batching and logs metrics using Weights & Biases.

4. **Hyperparameter Tuning**: Hyperparameters such as the number of epochs, hidden size, number of layers, dropout probabilities, embedding size, bidirectional flag, learning rate, optimizer, and teacher forcing ratio are tuned using a parameter dictionary.

## Code Specifications

The Python script accepts command-line arguments with default values and descriptions:

| Argument                  | Default Value | Description                                                  |
|---------------------------|---------------|--------------------------------------------------------------|
| `-wp`, `--wandb_project`  | myprojectname | Project name for Weights & Biases tracking                   |
| `-we`, `--wandb_entity`   | myname        | Wandb Entity for Weights & Biases tracking                   |
| `-e`, `--epochs`          | 20            | Number of epochs to train the neural network                 |
| `-hs`, `--hidden_size`    | 512           | Hidden size of the model                                     |
| `-nl`, `--num_layers`     | 3             | Number of layers in the encoder and decoder                  |
| `-e_dp`, `--e_drop_out`   | 0.5           | Dropout probability in the encoder                            |
| `-d_dp`, `--d_drop_out`   | 0.5           | Dropout probability in the decoder                            |
| `-es`, `--embedding_size` | 64            | Size of the embedding layer                                  |
| `-bi`, `--bidirectional`  | True          | Whether to use bidirectional RNNs or not                     |
| `-lg`, `--logs`           | False         | Whether to log or not                                        |
| `-lr`, `--lr`             | 1e-3          | Learning rate of the model                                   |
| `-m`, `--model`           | LSTM          | Choice of RNN model (LSTM, GRU, RNN)                         |
| `-o`, `--optimizer`       | sgd           | Choice of optimizer (sgd, adam)                              |
| `-tr`, `--teacher_forcing`| 0.5           | Teacher forcing ratio during training                        |
| `-w_d`, `--weight_decay`  | 0.0           | Weight decay used by optimizers                              |

## Conclusion

The CS6910_Assignment3 repository provides a comprehensive implementation of sequence-to-sequence learning, allowing for experimentation with various RNN cells, attention mechanisms, and hyperparameters. By leveraging the provided code and documentation, users can gain insights into the effectiveness of different techniques in machine transliteration and NLP tasks.
