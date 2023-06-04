# Dual LSTM Tagger

This is a repository for a Dual LSTM Tagger, a deep learning model used for part-of-speech (POS) tagging. The model is implemented using the PyTorch framework.

## Prerequisites

Make sure you have the following dependencies installed:

- pandas
- numpy
- torch
- tqdm

## Installation

To use this project, follow these steps:

1. Clone the repository:
git clone https://github.com/rhuthik/Lstm-tagger.git

2. Install the required dependencies:
pip install pandas numpy torch tqdm

3. Download the training and testing data files (`en_atis-ud-train.conllu` and `en_atis-ud-test.conllu`) and place them in the `Data` directory.

## Usage

To run the Dual LSTM Tagger, execute the following command:
python dual_lstm_tagger.py

The model will be trained on the provided training data and the test results will be displayed.

## Model Architecture

The Dual LSTM Tagger consists of two LSTM layers: one for processing word embeddings and the other for processing character embeddings. The model takes as input a sentence and its corresponding words. It embeds the words and characters, concatenates the embeddings, and passes the combined embeddings through the LSTM layers. Finally, the output is passed through a linear layer to obtain the tag probabilities.

The model hyperparameters are as follows:

- Word Embedding Dimension: 1024
- Character Embedding Dimension: 128
- Word Hidden Dimension: 1024
- Character Hidden Dimension: 1024
- Number of Epochs: 10

## Training

During the training process, the model is trained using the provided training data (`en_atis-ud-train.conllu`). The loss is calculated using the negative log-likelihood loss function. The Adam optimizer is used to optimize the model parameters with a learning rate of 0.01.

The training progress is displayed using a progress bar, and the loss is recorded after each epoch. At the end of training, the average loss per epoch is printed.

## Testing

After training, the model is ready to perform POS tagging on new input sentences. You can enter a sentence when prompted, and the model will return the predicted tags for each word in the sentence.
On testing on the training set, the weighted f1 score obtained was 0.9661693654670092

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

