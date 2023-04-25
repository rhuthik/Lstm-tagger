import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
import conllu
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
warnings.filterwarnings("ignore")

# Read data from the conllu files
with open("Data/en_atis-ud-train.conllu", "r", encoding = "utf-8") as f:
    trainingData = f.read()
with open("Data/en_atis-ud-test.conllu", "r", encoding = "utf-8") as f:
    testingData = f.read()

trainSentences = conllu.parse(trainingData)
testSentences = conllu.parse(testingData)

# Extract words, tags, and characters
word2index = {}
tag2index = {}
char2index = {}
taggedTrainSentences = []

for sentence in trainSentences:
    taggedSentence = []
    for token in sentence:
        form, pos_tag = token["form"], token["upos"]
        taggedSentence.append((form, pos_tag))
        if form not in word2index:
            word2index[form] = len(word2index)
        for char in form:
            if char not in char2index:
                char2index[char] = len(char2index)
        if pos_tag not in tag2index:
            tag2index[pos_tag] = len(tag2index)
    taggedTrainSentences.append(taggedSentence)

word_vocab_size = len(word2index)
tag_vocab_size = len(tag2index)
char_vocab_size = len(char2index)

# Define the model
WORD_EMBEDDING_DIM = 1024
CHAR_EMBEDDING_DIM = 128
WORD_HIDDEN_DIM = 1024
CHAR_HIDDEN_DIM = 1024
EPOCHS = 10

class DualLSTMTagger(nn.Module):
    def __init__(self, word_embedding_dim, word_hidden_dim, char_embedding_dim, char_hidden_dim, word_vocab_size, char_vocab_size, tag_vocab_size):
        super().__init__()
        self.word_embeddings = nn.Embedding(word_vocab_size, word_embedding_dim)
        self.char_embeddings = nn.Embedding(char_vocab_size, char_embedding_dim)
        self.char_lstm = nn.LSTM(char_embedding_dim, char_hidden_dim)
        self.lstm = nn.LSTM(word_embedding_dim + char_hidden_dim, word_hidden_dim)
        self.hidden2tag = nn.Linear(word_hidden_dim, tag_vocab_size)

    def forward(self, sentence, words):
        # Embed words and characters
        word_embeds = self.word_embeddings(sentence)
        char_hidden_total = []
        for word in words:
            char_embeds = self.char_embeddings(word)
            _, (char_hidden, char_cell_state) = self.char_lstm(char_embeds.view(len(word), 1, -1))
            word_char_hidden_state = char_hidden.view(-1)
            char_hidden_total.append(word_char_hidden_state)
        char_hidden_total = torch.stack(char_hidden_total)

        # Concatenate embeddings
        combined = torch.cat((word_embeds, char_hidden_total), 1)

        # Run LSTM and compute scores
        lstm_out, _ = self.lstm(combined.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)

        return tag_scores

model = DualLSTMTagger(WORD_EMBEDDING_DIM, WORD_HIDDEN_DIM, CHAR_EMBEDDING_DIM, CHAR_HIDDEN_DIM, word_vocab_size, char_vocab_size, tag_vocab_size).to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

lossList = []
interval = round(len(taggedTrainSentences) / 100.)
epochInterval = round(EPOCHS / 2.)

for epoch in tqdm(range(EPOCHS)):
    epochLoss = 0
    for sentence, tags in taggedTrainSentences:
        model.zero_grad()

        sentence_in = torch.tensor(sequence2index(sentence, word2index), dtype=torch.long).to(device)
        tags_in = torch.tensor(sequence2index(tags, tag2index), dtype=torch.long).to(device)

        words = []
        for word in sentence:
            char_indices = sequence2index(word, char2index)
            char_tensor = torch.tensor(char_indices, dtype=torch.long).to(device)
            words.append(char_tensor)

        tag_scores = model(sentence_in, words)
        loss = criterion(tag_scores, tags_in)
        loss.backward()
        optimizer.step()

        epochLoss += loss.item()

    epochLoss /= len(taggedTrainSentences)
    lossList.append(epochLoss)

    if (epoch + 1) % epochInterval == 0:
        print(f"Epoch {epoch+1} Completed,\tLoss {np.mean(lossList[-epochInterval:])}")

testSentence = input("Please enter sentence:")
testSequence = testSentence.split()

with torch.no_grad():
    words = [torch.tensor(sequence2index(s, char2index), dtype=torch.long).to(device) for s in testSequence]
    sentence = torch.tensor(sequence2index(testSequence, word2index), dtype=torch.long).to(device)

    tagScores = model(sentence, words)
    _, indices = torch.max(tagScores, 1)
    ans = [(testSequence[i], index2tag[index.item()]) for i, index in enumerate(indices)]
    print(ans)

