from torchnlp.datasets import imdb_dataset
from transformers import BertModel, BertTokenizer
from keras.preprocessing.sequence import pad_sequences
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
#import sol.pytorch as sol
import random as rn
import time

"""
https://github.com/shudima/notebooks/blob/master/BERT_to_the_rescue.ipynb
"""

#sol.cache.clear()

rn.seed(321)
np.random.seed(321)
torch.manual_seed(321)
torch.cuda.manual_seed(321)
#sol.set_seed(321)

device=torch.device("cuda:0")

train_data, test_data = imdb_dataset(train=True, test=True)

n = 1000

train_data = train_data[:n]
test_data = test_data[:n]

print(len(train_data))

#exit()

train_texts, train_labels = list(zip(*map(lambda d: (d['text'], d['sentiment']), train_data)))
test_texts, test_labels = list(zip(*map(lambda d: (d['text'], d['sentiment']), test_data)))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

train_tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:511], train_texts))
test_tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:511], test_texts))
train_tokens_ids = list(map(tokenizer.convert_tokens_to_ids, train_tokens))
test_tokens_ids = list(map(tokenizer.convert_tokens_to_ids, train_tokens_ids))
train_tokens_ids = pad_sequences(train_tokens_ids, maxlen=512, truncating="post", padding="post", dtype="int")
test_tokens_ids = pad_sequences(test_tokens_ids, maxlen=512, truncating="post", padding="post", dtype="int")
train_y = np.array(train_labels) == 'pos'
test_y = np.array(test_labels) == 'pos'

bert = BertModel.from_pretrained('bert-base-uncased')
x = torch.tensor(train_tokens_ids[:3])
y, pooled = bert(x, output_attentions=False, output_hidden_states=False)

print('x shape:', x.shape)
print('y shape:', y.shape)
print('pooled shape:', pooled.shape)

class BertBinaryClassifier(torch.nn.Module):
    def __init__(self, dropout=0.1):
        super(BertBinaryClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(768, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, tokens):
        _, pooled_output = self.bert(tokens, output_attentions=False, output_hidden_states=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        proba = self.sigmoid(linear_output)
        return proba

BATCH_SIZE = 1
EPOCHS = 5

train_tokens_tensor = torch.tensor(train_tokens_ids)
train_y_tensor = torch.tensor(train_y.reshape(-1, 1)).float()
test_tokens_tensor = torch.tensor(test_tokens_ids)
test_y_tensor = torch.tensor(test_y.reshape(-1, 1)).float()

train_dataset = TensorDataset(train_tokens_tensor, train_y_tensor)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE)

test_dataset = TensorDataset(test_tokens_tensor, test_y_tensor)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE)

bert_clf = BertBinaryClassifier()
print(test_dataset.tensors[0].size())
#opt = sol.optimize(bert_clf, sol.input([0, 512], dtype=torch.long), batch_size=BATCH_SIZE)
#opt.load_state_dict(bert_clf.state_dict(), strict=False)
#opt.to(device)
optimizer = torch.optim.Adam(bert_clf.parameters(), lr=3e-6)
bert_clf.to(device)

#opt.train()
#start_time = time.time()
#for epoch_num in range(EPOCHS):
#    for step_num, batch_data in enumerate(train_dataloader):
#        token_ids, labels = tuple(t for t in batch_data)
#        token_ids = token_ids.to(device)
#        probas = opt(token_ids)
#        probas = probas.cpu()
#        loss_func = torch.nn.BCELoss()
#        batch_loss = loss_func(probas, labels)
#        opt.zero_grad()
#        #print(batch_loss)
#        batch_loss.backward()
#        optimizer.step()
#    #print('end of epoch', time.time())

bert_clf.eval()
bert_predicted = []
all_logits = []
start_time = time.time()
with torch.no_grad():
    for step_num, batch_data in enumerate(test_dataloader):
        token_ids, labels = tuple(t for t in batch_data)
        token_ids = token_ids.to(device)
        logits = bert_clf(token_ids)
        logits = logits.cpu()
        loss_func = torch.nn.BCELoss()
        loss = loss_func(logits, labels)
        numpy_logits = logits.detach().numpy()
        bert_predicted += list(numpy_logits[:, 0] > 0.5)
        all_logits += list(numpy_logits[:, 0])

tot_time = time.time() - start_time
print("%.2f"%tot_time)
