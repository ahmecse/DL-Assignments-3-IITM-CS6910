#!/usr/bin/env python
# coding: utf-8

# In[4]:


####importing necessary packages#######
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import wandb
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import argparse


# In[2]:


def unique_char(lan):
    unique_cha = ''  # Initialize an empty string to store unique characters
    for word in lan:  # Iterate through each word in the list
        for char in word:  # Iterate through each character in the word
            if char not in unique_cha:  # Check if the character is not already in the unique_cha string
                unique_cha += char  # Append the unique character to the string
    return unique_cha  # Return the string containing all unique characters
class LANG:
    def __init__(self, lang):
        self.lang = lang  # Initialize the language data
        self.word2index = {}  # Dictionary to map characters to indices
        self.index2word = {0: 'SOS', 1: 'EOS'}  # Dictionary to map indices to characters
        self.max_length = 0  # Variable to store the length of the longest word
        self.count = 2  # Counter for indexing, starting after SOS and EOS
        self.max_word = ''  # Variable to store the longest word

    def addchar(self):
        for word in self.lang:  # Iterate through each word in the language data
            length = len(word)  # Get the length of the current word
            if length > self.max_length:  # Check if the current word is the longest so far
                self.max_length = length  # Update the maximum word length
                self.max_word = word  # Update the longest word
            for char in word:  # Iterate through each character in the word
                if char not in self.word2index:  # Check if the character is not already indexed
                    self.word2index[char] = self.count  # Assign the current count as the index
                    self.index2word[self.count] = char  # Map the current count to the character
                    self.count += 1  # Increment the counter
        return self.word2index, self.index2word, self.max_length, self.max_word  # Return the mappings and max values
def Tensorpair(eng, mal, access_eng, access_mal):
    # Unpack access_eng and access_mal
    eng_word2index, eng_index2word, eng_maxlength, eng_word = access_eng
    mal_word2index, mal_index2word, mal_maxlength, mal_word = access_mal
    
    n = len(eng)
    # Initialize tensors for input and target sequences
    input_ids = torch.zeros((n, eng_maxlength + 1), dtype=torch.int32)
    target_ids = torch.zeros((n, eng_maxlength + 1), dtype=torch.int32)
    
    for i, (eng_word, mal_word) in enumerate(zip(eng, mal)):
        try:
            # Convert characters to indices using word2index mappings
            input_indx = [eng_word2index[char] for char in eng_word]
            input_indx.append(1)  # Append EOS token index
            input_indx = torch.tensor(input_indx, dtype=torch.long)
            
            target_indx = [mal_word2index[char] for char in mal_word]
            target_indx.append(1)  # Append EOS token index
            target_indx = torch.tensor(target_indx, dtype=torch.long)
            
            # Update input and target tensors
            input_ids[i, :len(input_indx)] = input_indx
            target_ids[i, :len(target_indx)] = target_indx
        except Exception as e:
            print(e)  # Print any exception that occurs
    
    # Create a TensorDataset from input and target tensors
    tensor_data = TensorDataset(input_ids, target_ids)
    
    return tensor_data


# In[3]:


class Encoder(nn.Module):
    def __init__(self, hidden_size=512, num_layers=2, drop_out=0.2, embedding_size=256, bidirection=False, model='LSTM'):
        super().__init__()
        # Initialize the embedding layer
        self.embedding = nn.Embedding(len(eng_index2word), embedding_size)
        # Define the parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.drop_out = drop_out
        # Choose the RNN model (LSTM/GRU/RNN)
        if model == 'LSTM':
            self.model = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=bidirection, batch_first=True)
        elif model == 'GRU':
            self.model = nn.GRU(embedding_size, hidden_size, num_layers, bidirectional=bidirection, batch_first=True)
        elif model == 'RNN':
            self.model = nn.RNN(embedding_size, hidden_size, num_layers, bidirectional=bidirection, batch_first=True)
        else:
            raise ValueError('Given model is not found')
        # Apply dropout regularization
        self.drop_out = nn.Dropout(p=drop_out)
    
    def forward(self, input):
        # Embed the input sequence
        embedded = self.drop_out(self.embedding(input))
        # Pass the embedded sequence through the RNN model
        output, hidden = self.model(embedded)
        return output, hidden

class Decoder(nn.Module):
    def __init__(self, hidden_size=512, num_layers=2, drop_out=0.2, embedding_size=256, bidirection=False, model='LSTM'):
        super().__init__()
        # Initialize the embedding layer
        self.embedding = nn.Embedding(len(mal_index2word), embedding_size)
        # Define the parameters
        self.hidden_size = hidden_size
        self.num_layer = num_layers
        self.drop_out = drop_out
        # Choose the RNN model (LSTM/GRU/RNN)
        if model == 'LSTM':
            self.model = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=bidirection, batch_first=True)
        elif model == 'GRU':
            self.model = nn.GRU(embedding_size, hidden_size, num_layers, bidirectional=bidirection, batch_first=True)
        elif model == 'RNN':
            self.model = nn.RNN(embedding_size, hidden_size, num_layers, bidirectional=bidirection, batch_first=True)
        else:
            raise ValueError('Given model is not found')
        # Apply dropout regularization
        self.drop_out = nn.Dropout(p=drop_out)
        # Define the output layer
        if bidirection:
            self.out = nn.Linear(2 * hidden_size, len(mal_index2word))
        else:
            self.out = nn.Linear(hidden_size, len(mal_index2word))
            
    def forward(self, input, hidden):
        # Embed the input sequence
        embedded = self.drop_out(self.embedding(input))
        # Pass the embedded sequence through the RNN model
        output, hidden = self.model(embedded, hidden)
        # Predict the output
        pred = self.out(output)
        return pred, hidden

class seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        # Initialize the encoder and decoder
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, inputs, targets, teacher_force_ratio):
        # Pass the input sequence through the encoder
        encoder_output, encoder_hidden = self.encoder(inputs)
        # Initialize the decoder input with zeros
        decoder_input = torch.empty(targets.shape[0], 1, dtype=torch.long, device='cuda').fill_(0)
        decoder_hidden = encoder_hidden
        # Initialize the output tensor
        output = torch.zeros((targets.shape[0], targets.shape[1], len(mal_index2word)), device='cuda')
        
        for i in range(output.shape[1]):
            # Pass the decoder input and hidden state through the decoder
            pred, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            pred = torch.squeeze(pred)
            # Store the predicted output
            output[:, i, :] = pred
            # Update the decoder input for the next time step
            best_guess = torch.argmax(pred, axis=-1).view(-1, 1)
            decoder_input = best_guess if np.random.rand() > teacher_force_ratio else targets[:, i].view(-1, 1)
            # Keep the decoder hidden state unchanged
            decoder_hidden = decoder_hidden
        return output

def word_finder_eng(eng_):
    # Initialize an empty list to store English words
    full = []
    for eng in eng_:
        # Find the index of the end-of-sequence token (EOS)
        eng_eos = np.where(eng == 1)[0][0]
        # Convert the numerical representation to an English word
        eng_word = num_word(eng[0:eng_eos], eng_index2word)
        full.append(eng_word)
    return np.array(full)

def word_finder_mal(mal_):
    # Initialize an empty list to store Malayalam words
    full = []
    for mal in mal_:
        # Find the index of the end-of-sequence token (EOS) if present, else use the length of the sequence
        mal_eos = np.where(mal == 1)[0][0] if 1 in mal else len(mal)
        # Convert the numerical representation to a Malayalam word
        mal_word = num_word(mal[0:mal_eos], mal_index2word)
        full.append(mal_word)
    return np.array(full)

def num_word(number, converter):
    # Convert a sequence of numerical representations to a word using the provided converter
    number = np.array(number)
    word = ''.join(converter[num] for num in number)
    return word
def train_model(epochs=30, hidden_size=512, num_layers=3, encoder_drop_out=0.2, decoder_drop_out=0.2, embedding_size=256,
                bidirection=False, model='LSTM', lr=1e-3, optimizer_='adam', teacher_force_ratio=0.5,log=False):
    # Set hyperparameters
    epochs = epochs
    hidden_size = hidden_size
    num_layers = num_layers
    encoder_drop_out = encoder_drop_out
    decoder_drop_out = decoder_drop_out
    embedding_size = embedding_size
    bidirection = bidirection
    model = model
    lr = lr
    optimizer_ = optimizer_
    teacher_force_ratio = teacher_force_ratio

    # Initialize encoder and decoder
    encoder = Encoder(hidden_size=hidden_size, num_layers=num_layers, drop_out=encoder_drop_out, embedding_size=embedding_size,
                      bidirection=bidirection, model=model)
    decoder = Decoder(hidden_size=hidden_size, num_layers=num_layers, drop_out=decoder_drop_out, embedding_size=embedding_size,
                      bidirection=bidirection, model=model)
    # Create seq2seq model
    model = seq2seq(encoder, decoder)
    model = model.to('cuda')

    # Define loss function
    lossfun = nn.CrossEntropyLoss()

    # Choose optimizer
    if optimizer_ == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_ == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError('Optimizer not found')

    # Initialize lists to store training and validation metrics
    train_loss, train_acc, val_loss, val_acc = [], [], [], []

    # Main training loop
    for i in range(epochs):
        # Initialize lists to store current epoch metrics
        train_loss_curr, train_acc_curr, val_loss_curr, val_acc_curr = [], [], [], []

        # Set model to training mode
        model.train()

        # Iterate over training data
        for eng, mal in tqdm(train_loader):
            eng = eng.to('cuda')
            mal = mal.to('cuda')

            # Forward pass
            pred = model(eng, mal, teacher_force_ratio=teacher_force_ratio)
            pred_loss = pred.reshape(-1, pred.shape[2])
            mal_loss = mal.reshape(-1,).long()
            loss = lossfun(pred_loss, mal_loss)

            # Compute and store training loss
            train_loss_curr.append(loss.cpu().detach().numpy())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute training accuracy
            mal_ground = word_finder_mal(mal.cpu().detach().numpy())
            pred_acc = np.argmax(pred.cpu().detach().numpy(), axis=-1).astype(np.int64)
            mal_predicted = word_finder_mal(pred_acc)
            acc = np.sum(mal_ground == mal_predicted) / len(mal_ground)
            train_acc_curr.append(acc)

        # Set model to evaluation mode
        model.eval()

        # Iterate over validation data
        for eng, mal in val_loader:
            eng = eng.to('cuda')
            mal = mal.to('cuda')

            # Forward pass (no teacher forcing)
            with torch.no_grad():
                pred = model(eng, mal, teacher_force_ratio=0)

            pred_loss = pred.reshape(-1, pred.shape[2])
            mal_loss = mal.reshape(-1,).long()

            # Compute and store validation loss
            loss = lossfun(pred_loss, mal_loss)
            val_loss_curr.append(loss.cpu().detach().numpy())

            # Compute validation accuracy
            mal_ground = word_finder_mal(mal.cpu().detach().numpy())
            pred_acc = np.argmax(pred.cpu().detach().numpy(), axis=-1)
            mal_predicted = word_finder_mal(pred_acc)
            acc = np.sum(mal_ground == mal_predicted) / len(mal_ground)
            val_acc_curr.append(acc)

        # Compute average metrics for the epoch
        train_loss.append(np.average(train_loss_curr))
        val_loss.append(np.average(val_loss_curr))
        train_acc.append(np.average(train_acc_curr))
        val_acc.append(np.average(val_acc_curr))



        # Log metrics using wandb
        if log==True:
            wandb.log({"Train_Accuracy": np.round(train_acc[i] * 100, 2), "Train_Loss": train_loss[i],
                       "Val_Accuracy": np.round(val_acc[i] * 100, 2), "Val_Loss": val_loss[i], "Epoch": i + 1})
        print(f'Epochs {i} completed, train loss and accuracy ={train_loss[i],train_acc[i]}' 
          f',and val loss and accuracy ={val_loss[i],val_acc[i]} ')


# In[ ]:


parser = argparse.ArgumentParser()
 
parser.add_argument("-wp", "--wandb_project", default = "myprojectname", help = "Project name used to track experiments ")
parser.add_argument("-we", "--wandb_entity", default = "ee22s060", help = "Wandb Entity ")
parser.add_argument("-e", "--epochs", default = 15, choices=[10,15], help = "Number of epochs to train neural network." , type=int)
parser.add_argument("-hs", "--hidden_size", default = 512,choices=[32, 64, 256, 512], help = "Hidden size of the model ", type=int)
parser.add_argument("-nl","--num_layers",default=2,choices=[1,2,3],help='Number of layers in the encoder and decoder',type=int)
parser.add_argument('-e_dp','--e_drop_out',default=0.5,choices=[0.2, 0.3, 0.5],help='Dropout probability in the encoder',type=float)
parser.add_argument('-d_dp','--d_drop_out',default=0.2,choices=[0.2, 0.3, 0.5],help='Dropout probability in the decoder',type=float)
parser.add_argument('-es','--embedding_size',default=32,choices=[32, 64, 256, 512],help='Size of the embedding layer',type=int)
parser.add_argument('-bi','--bidirectional',default=True,choices=["True","False"],help='Whether to use bidirectional RNNs or not')
parser.add_argument("-lg", "--logs", default = "False", choices = ["True","False"],help = "whether to log or not" )
parser.add_argument('-lr',"--lr",default=1e-3,choices=[1e-3,1e-4,1e-5],help="Learning rate of the model",type=float)
parser.add_argument("-m","--model",default='LSTM',choices=['LSTM', 'GRU', 'RNN'])
parser.add_argument("-opt","--optimizer",default='adam',choices=['sgd','adam'],help="optimizer function")
parser.add_argument("-tr","--teacher_forcing",default=0.5,choices=[0.2, 0.3, 0.4, 0.5],help='Teacher forcing ratio during training')
args = parser.parse_args()

train_data=pd.read_csv(r'data\mal_train.csv') # Load training data from CSV file
val_data=pd.read_csv(r'data/mal_valid.csv')# Load validation data from CSV file
test_data=pd.read_csv(r'data/mal_test.csv') # Load test data from CSV file

eng_train = train_data['English']  # Extract English training data
mal_train = train_data['Malayalam']  # Extract Malayalam training data
eng_val = val_data['English']  # Extract English validation data
mal_val = val_data['Malayalam']  # Extract Malayalam validation data
eng_test = test_data['English']  # Extract English test data
mal_test = test_data['Malayalam']  # Extract Malayalam test data
english_all = pd.concat([eng_train, eng_val, eng_test], ignore_index=True)  # Concatenate all English data
malayalam_all = pd.concat([mal_train, mal_val, mal_test], ignore_index=True)  # Concatenate all Malayalam data

# Initialize language objects for English and Malayalam
eng = LANG(english_all)
# Get mappings and max values for English
eng_word2index, eng_index2word, eng_maxlength, eng_word = eng.addchar()
access_eng = eng_word2index, eng_index2word, eng_maxlength, eng_word

mal = LANG(malayalam_all)
# Get mappings and max values for Malayalam
mal_word2index, mal_index2word, mal_maxlength, mal_word = mal.addchar()
access_mal = mal_word2index, mal_index2word, mal_maxlength, mal_word

# Generating tensor pairs for training, validation, and testing data
train_data=Tensorpair(eng_train,mal_train,access_eng,access_mal)
val_data=Tensorpair(eng_val,mal_val,access_eng,access_mal)
test_data=Tensorpair(eng_test,mal_test,access_eng,access_mal)

# Setting batch size and creating data loaders for training, validation, and testing
batch_size=64
train_loader=DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True,drop_last=True)
val_loader=DataLoader(dataset=val_data,batch_size=batch_size,drop_last=True)
test_loader=DataLoader(dataset=test_data,batch_size=batch_size,drop_last=True)

'''
project_name='With attention'
wandb.login(key="5bfaaa474f16b4400560a3efa1e961104ed54810")
wandb.init(project=args.wandb_project,entity=args.wandb_entity)
'''

train_model(epochs=args.epochs,hidden_size=args.hidden_size,num_layers=args.num_layers,encoder_drop_out=args.e_drop_out,
            decoder_drop_out=args.d_drop_out,embedding_size=args.embedding_size,bidirection=args.bidirectional,
            model=args.model,lr=args.lr,optimizer_=args.optimizer,teacher_force_ratio=args.teacher_forcing,log=args.logs)

