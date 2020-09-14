import re
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import sol.pytorch as sol
sol.cache.clear()
import time

#https://www.analyticsvidhya.com/blog/2020/08/build-a-natural-language-generation-nlg-system-using-pytorch/

#sol.device.set(sol.device.ve, 0)
device = torch.device("hip:0")

# read pickle file
pickle_in = open("plots_text.pickle","rb")
movie_plots = pickle.load(pickle_in)

# count of movie plot summaries
len(movie_plots)

# sample random summaries
#print(random.sample(movie_plots, 5))

# clean text
movie_plots = [re.sub("[^a-z' ]", "", i) for i in movie_plots]

# create sequences of length 5 tokens
def create_seq(text, seq_len = 5):
    sequences = []
    # if the number of tokens in 'text' is greater than 5
    if len(text.split()) > seq_len:
        for i in range(seq_len, len(text.split())):
            # select sequence of tokens
            seq = text.split()[i-seq_len:i+1]
            # add to the list
            sequences.append(" ".join(seq)) 
        return sequences
        # if the number of tokens in 'text' is less than or equal to 5
    else:
        return [text]

seqs = [create_seq(i) for i in movie_plots]

# merge list-of-lists into a single list
seqs = sum(seqs, [])

# count of sequences
#print(len(seqs))

# create inputs and targets (x and y)
x = []
y = []

for s in seqs:
    x.append(" ".join(s.split()[:-1]))
    y.append(" ".join(s.split()[1:]))

# create integer-to-token mapping
int2token = {}
cnt = 0

for w in set(" ".join(movie_plots).split()):
    int2token[cnt] = w
    cnt+= 1

# create token-to-integer mapping
token2int = {t: i for i, t in int2token.items()}

#print(token2int["the"], int2token[14271])

# set vocabulary size
vocab_size = len(int2token)
#print(vocab_size)

def get_integer_seq(seq):
    return [token2int[w] for w in seq.split()]

# convert text sequences to integer sequences
x_int = [get_integer_seq(i) for i in x]
y_int = [get_integer_seq(i) for i in y]

# convert lists to numpy arrays
x_int = np.array(x_int)
y_int = np.array(y_int)

def get_batches(arr_x, arr_y, batch_size):
    # iterate through the arrays
    prv = 0
    for n in range(batch_size, arr_x.shape[0], batch_size):
        x = arr_x[prv:n,:]
        y = arr_y[prv:n,:]
        prv = n
        yield x, y

class WordClassifier(nn.Module):
    def __init__(self, n_hidden, vocab_size, drop_prob):
        super().__init__()
        self.n_hidden = n_hidden
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden, vocab_size)
        
    def forward(self, lstm_output):
        out = self.dropout(lstm_output * 1.00001) # Quick fix
        #out = out.reshape(-1, self.n_hidden)
        out = self.fc(out)
        return out

class WordLSTM(nn.Module):
    def __init__(self, n_hidden=256, n_layers=4, drop_prob=.3, lr=.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        ## define the Embedding
        emb = nn.Embedding(vocab_size, 200)
        self.emb_layer = sol.optimize(emb, sol.input([0, 5], dtype=torch.long), batch_size=32)
        # CHANGE: Don't forget to load the state dict! In future version this will be done automatically.
        self.emb_layer.load_state_dict(emb.state_dict())
        
        ## define the LSTM
        self.lstm = nn.LSTM(200, n_hidden, n_layers, dropout=drop_prob, batch_first=True)
        
        ## define the Classifier
        classifier        = WordClassifier(n_hidden, vocab_size, drop_prob)
        #sol.config["compiler::name"] = "Classifier"
        #sol.config["compiler::debug"] = True
        #sol.config["compiler::debug_params"] = True
        self.classifier    = sol.optimize(classifier, sol.input([0, 5, n_hidden], dtype=torch.float32, requires_grad=True), batch_size=32)
        # CHANGE: Don't forget to load the state dict! In future version this will be done automatically.
        self.classifier.load_state_dict(classifier.state_dict())
    
    def forward(self, x, hidden):
        '''
        Forward pass through the network. 
        These inputs are x, and the hidden/cell state `hidden`.
        '''
        ## pass input through embedding layer
        embedded = self.emb_layer(x)

        # CHANGE: Pytorch VE support is only minimal. SOL only adds the absolute necessary
        # functionality, which does not include LSTMs. So we need to copy the data to the host
        #  and then run the LSTM on CPU and then copy back to Auror.a
        embedded = embedded.cpu()
        lstm_output, hidden = self.lstm(embedded, hidden)
        lstm_output = lstm_output.contiguous().to(x.device)
        
        out = self.classifier(lstm_output)
        out = out.reshape(-1, vocab_size)
        return out, hidden

    def init_hidden(self, batch_size):
        ''' initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        hidden = (torch.zeros(self.n_layers, batch_size, self.n_hidden), torch.zeros(self.n_layers, batch_size, self.n_hidden))

        # CHANGE: DON'T COPY HIDDEN ONTO DEVICE!
        #hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
        #          weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        return hidden

net = WordLSTM()

def train(net, epochs=10, batch_size=32, lr=0.001, clip=1, print_every=32):
    # optimizer
    opt = torch.optim.SGD(net.parameters(), lr=lr)            
    # loss
    criterion = nn.CrossEntropyLoss()                        
    # push model to Aurora
    # CHANGE: only copy the parts that get executed on VE to the device!
    net.emb_layer.to(device)
    net.classifier.to(device)                                   
    counter = 0
    net.train()
    #for k, v in net.named_parameters():
    #        print(k, v.data.device)
    for e in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size)                                                                        
        for x, y in get_batches(x_int, y_int, batch_size):
            counter+= 1
            # convert numpy arrays to PyTorch arrays
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
            # push tensors to GPU
            #inputs, targets = inputs.cuda(), targets.cuda()
            # push input to Aurora
            inputs = inputs.to(device)
            # detach hidden states
            h = tuple([each.data for each in h])
            # zero accumulated gradients
            net.zero_grad()
            # get the output from the model
            #print("input", inputs.device)
            #for i in h:
            #        print("hidden", i.device)
            output, h = net(inputs, h)
            # move output back to cpu
            output=output.cpu()
            # calculate the loss and perform backprop
            print(output.size(), targets.view(-1).size())
            loss = criterion(output, targets.view(-1))
            # back-propagate error 
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            # update weigths
            opt.step() 
            if counter % print_every == 0:
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(counter))


# train the model
train(net, batch_size = 32, epochs=20, print_every=25)


#optimizer = optim.SGD(opt.parameters(), lr=.01, momentum=.9)

#start_time = time.time()
#print("Training starts")

#opt.train()
#for epoch in range(2):  # loop over the dataset multiple times
#    running_loss = 0.0
#    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
#        inputs, labels = data #data[0].to(device), data[1].to(device)
        
        #labels = to_categorical(labels, 10)

        # zero the parameter gradients
#        optimizer.zero_grad()

        # forward + backward + optimize
#        inputs = inputs.to(device)
        #labels = labels.to(device)
#        outputs = opt(inputs)
#        outputs = outputs.cpu()
#        loss = criterion(outputs, labels)
#        loss.backward()
        #torch.hip.synchronize()
#        optimizer.step()

        # print statistics
#        running_loss += loss.item()
        #print("%d\t%9.4f"%(i, running_loss))
#        if i % 200 == 199:    # print every 2000 mini-batches
            #print('[%d, %5d] loss: %.3f' %
            #    (epoch + 1, i + 1, running_loss / 200))
#            running_loss = 0.0

#elapsed_time = time.time() - start_time

#print('Finished Training')
#print("Elapsed time[s]: %.2f"%elapsed_time)
