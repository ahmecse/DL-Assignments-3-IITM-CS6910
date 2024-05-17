# CS6910_Assignment3
The goal of this assignment is threefold: (i) learn how to model sequence-to-sequence learning problems using Recurrent Neural Networks (ii) compare different cells such as vanilla RNN, LSTM and GRU (iii) understand how attention networks overcome the limitations of vanilla seq2seq models (iv) understand the importance of Transformers in the context of machine transliteration
and NLP in general
## Problem Statement
In this assignment, you will experiment with a sample of the Aksharantar dataset released by AI4Bharat. This dataset contains pairs of the following form:\
x﻿,y﻿﻿\
ajanabee,अजनबी
i.e., a word in the native script and its corresponding transliteration in the Latin script (how we type while chatting with our friends on WhatsApp etc). Given many such
$(x_i,y_i)_{i=1}^n$ pairs your goal is to train a model $y=f^{'}(x)$ which takes as input a romanized string (ghar) and produces the corresponding word in Devanagari (घर). 
This is the problem of mapping a sequence of characters in one language to a sequence of characters in another 
## Process
* The train, val and test are read from the csv file
* A class 'Lang' is created for converting the character into numbers and numbers back to character
```
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
```
* A function Tensorpair is created for converting the words into number for both english and malayalam. Finally it is converted into torch tensors
```
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
  ```
* Train, val and test loaders are created with a batch size of 64
* A encoder class is created for encoding the input english characters
```
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
```
* There are two decoder class, one- when there is attention and second- when is there is no attention
## when there is attention
```
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
            self.out = nn.Linear(4 * hidden_size, len(mal_index2word))
        else:
            self.out = nn.Linear(2 * hidden_size, len(mal_index2word))
    
    def Attention(self, decoder_output, encoder_output):
        # Calculate attention scores
        score = (decoder_output @ encoder_output.transpose(1, 2))
        # Apply softmax to get attention weights
        weight = F.softmax(score, dim=-1)
        # Calculate the context vector
        content = weight @ encoder_output
        return content, weight
    
    def forward(self, input, hidden, encoder_output):
        # Embed the input sequence
        embedded = self.drop_out(self.embedding(input))
        # Pass the embedded sequence through the RNN model
        output, hidden = self.model(embedded, hidden)
        # Compute attention and get context vector
        content, weight = self.Attention(output, encoder_output)
        # Concatenate output with context vector
        final = torch.cat([output, content], dim=-1)
        # Predict the output
        pred = self.out(final)
        return pred, hidden, weight
```
## when there is no attention
```
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
```
* A sequence to sequence class is created for combining the encoder and decoder
```
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
        # Initialize the list to store attention weights
        attention = []
        
        for i in range(output.shape[1]):
            # Pass the decoder input, hidden state, and encoder output through the decoder
            pred, decoder_hidden, weight = self.decoder(decoder_input, decoder_hidden, encoder_output)
            # Append the attention weights to the list
            attention.append(weight)
            pred = torch.squeeze(pred)
            # Store the predicted output
            output[:, i, :] = pred
            # Update the decoder input for the next time step
            best_guess = torch.argmax(pred, axis=-1).view(-1, 1)
            decoder_input = best_guess if np.random.rand() > teacher_force_ratio else targets[:, i].view(-1, 1)
            # Keep the decoder hidden state unchanged
            decoder_hidden = decoder_hidden
        
        return output, attention
```
* For calculating accuracy three additional functions were defined
```
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
```
* Finally a training function is created for training and validating the model
```
def train_model(epochs=30, hidden_size=512, num_layers=3, encoder_drop_out=0.2, decoder_drop_out=0.2, embedding_size=256,
                bidirection=False, model='LSTM', lr=1e-3, optimizer_='adam', teacher_force_ratio=0.5):
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
            pred, attention = model(eng, mal, teacher_force_ratio=teacher_force_ratio)
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
                pred, attention = model(eng, mal, teacher_force_ratio=0)

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
        wandb.log({"Train_Accuracy": np.round(train_acc[i] * 100, 2), "Train_Loss": train_loss[i],
                   "Val_Accuracy": np.round(val_acc[i] * 100, 2), "Val_Loss": val_loss[i], "Epoch": i + 1})
```
*Following where the different hyperparameter values
```
parameters_dict = {
    'epochs': {
        'values': [10, 15]  # Number of epochs
    },
    'hidden_size': {
        'values': [32, 64, 256, 512]  # Hidden size of the model
    },
    'num_layers': {
        'values': [1, 2, 3]  # Number of layers in the encoder and decoder
    },
    'encoder_drop_out': {
        'values': [0.2, 0.3, 0.5]  # Dropout probability in the encoder
    },
    'decoder_drop_out': {
        'values': [0.2, 0.3, 0.5]  # Dropout probability in the decoder
    },
    'embedding_size': {
        'values': [32, 64, 256, 512]  # Size of the embedding layer
    },
    'bidirectional': {
        'values': [True, False]  # Whether to use bidirectional RNNs or not
    },
    'model': {
        'values': ['LSTM', 'GRU', 'RNN']  # Type of RNN model
    },
    'lr': {
        'values': [1e-3, 1e-4, 1e-5]  # Learning rate
    },
    'teacher_force_ratio': {
        'values': [0.2, 0.3, 0.4, 0.5]  # Teacher forcing ratio during training
    },
    'optimizer': {
        'values': ['adam', 'sgd']  # Optimizer to use
    }
}
```
## Code specifications
A python script with attention and with out attention was created that accepts the following command line arguments with the specified values

```
| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-wp`, `--wandb_project` | myprojectname | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | myname  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-e`, `--epochs` | 20 |  Number of epochs to train neural network.|
| `-hs`, `--hidden_size` | 512 | Hidden size of the model. | 
| "-nl","--num_layers" | 3 | Number of layers in the encoder and decoder |
|'-e_dp','--e_drop_out'|0.5|Dropout probability in the encoder
| '-d_dp','--d_drop_out' | 0.5 | Dropout probability in the decoder | 
| '-es','--embedding_size' | 64 | Size of the embedding layer |
| '-bi','--bidirectional' | True | Whether to use bidirectional RNNs or not | 
| "-lg", "--logs" | False | whether to log or not | 
| '-lr',"--lr" | 1e-3 | Learning rate of the model |
| "-m","--model" | LSTM | which model to choose |
| `-w_d`, `--weight_decay` | .0 | Weight decay used by optimizers. |
| "-tr","--teacher_forcing" | 0.5 | "Teacher forcing ratio during training" |
| `-o`, `--optimizer` | sgd | choices:  ["sgd", "adam"] | 
```
