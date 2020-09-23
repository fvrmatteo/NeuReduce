#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

import torchtext
from torchtext.data import Field, BucketIterator


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# Define tokenizer
def _tokenize(text):
    return [char for char in text]

# Define field
SRC = Field(
    tokenize=_tokenize, 
    init_token='<sos>', 
    eos_token='<eos>',
    pad_token='<pad>',
    lower=False
)
TRG = Field(
    tokenize=_tokenize,
    init_token='<sos>',
    eos_token='<eos>',
    pad_token='<pad>',
    lower=False
)

exprs = torchtext.data.TabularDataset(
    path='./dataset.csv',
    format='csv',
    fields=[
        ('src', SRC),
        ('trg', TRG)
    ]
)

train_data, valid_data = exprs.split(split_ratio=0.8)

print(f'Total {len(exprs)} samples.')
print(f'Total {len(train_data)} train samples.')
print(f'Total {len(valid_data)} valid samples.')


# Build vocab only from the training set, which can prevent information leakage
SRC.build_vocab(train_data)
TRG.build_vocab(train_data)
print(f'Total {len(SRC.vocab)} unique tokens in source vocabulary')
print(f'Total {len(TRG.vocab)} unique tokens in target vocabulary')


batch_size = 128
device = torch.device('cuda')

train_iter, valid_iter = BucketIterator.splits(
    (train_data, valid_data),
    batch_size=batch_size,
    sort=False,
    device=device
)

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.embedding = nn.Embedding(input_dim, emb_dim)    
        self.rnn = nn.GRU(emb_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        #src = [src_len, batch_size], 每一列是一个样本，每个样本长度固定为src_len(102)
        embedded = self.dropout(self.embedding(src))
        #embedded = [src_len, batch_size, emb_dim]
        outputs, hidden = self.rnn(embedded)
        #outputs = [src_len, batch_size, hid_dim * n_directions]
        #hidden = [n_layers * n_directions, batch_size, hid_dim]
        #cell = [n_layers * n_directions, batch_size, hid_dim]
        #outputs are always from the top hidden layer

        return hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.output_dim = output_dim        
        self.embedding = nn.Embedding(output_dim, emb_dim)    
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim)        
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, context):        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #context = [n layers * n directions, batch size, hid dim]        
        #n layers and n directions in the decoder will both always be 1, therefore:
        #hidden = [1, batch size, hid dim]
        #context = [1, batch size, hid dim]        
        input = input.unsqueeze(0)        
        #input = [1, batch size]        
        embedded = self.dropout(self.embedding(input))        
        #embedded = [1, batch size, emb dim]                
        emb_con = torch.cat((embedded, context), dim=2)       
        #emb_con = [1, batch size, emb dim + hid dim]         
        output, hidden = self.rnn(emb_con, hidden)        
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]        
        #seq len, n layers and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [1, batch size, hid dim]        
        output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)), dim = 1)        
        #output = [batch size, emb dim + hid dim * 2]        
        prediction = self.fc_out(output)        
        #prediction = [batch size, output dim]

        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.hid_dim == decoder.hid_dim,             "Hidden dimensions of encoder and decoder must be equal!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5): 
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time        
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)        
        #last hidden state of the encoder is the context
        context = self.encoder(src)        
        #context also used as the initial hidden state of the decoder
        hidden = context        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]  
              
        for t in range(1, trg_len):            
            #insert input token embedding, previous hidden state and the context state
            #receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, context)            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1)            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs


INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.01)
        
model.apply(init_weights)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')


# In[12]:


optimizer = optim.Adam(model.parameters())

TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)


def train(model, iterator, optimizer, criterion, clip):    
    model.train()    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):        
        src = batch.src
        trg = batch.trg        
        optimizer.zero_grad()        
        output = model(src, trg)        
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]        
        output_dim = output.shape[-1]        
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)        
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]        
        loss = criterion(output, trg)        
        loss.backward()        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)        
        optimizer.step()        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):    
    model.eval()    
    epoch_loss = 0
    
    with torch.no_grad():    
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg, 0) #turn off teacher forcing
            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]
            output_dim = output.shape[-1]            
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]
            loss = criterion(output, trg)            
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


import datetime
from tqdm import tqdm

today = datetime.date.today()
N_EPOCHS = 100
CLIP = 1

best_valid_loss = float('inf')
# pbar = tqdm(range(N_EPOCHS))
# for epoch in pbar: 
for epoch in range(N_EPOCHS):
    train_loss = train(model, train_iter, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iter, criterion)
#     pbar.set_description(f'Epoch: {epoch+1}, train loss: {train_loss:.3f}, val loss: {valid_loss:.3f}')

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), f'gru-{today}.pt')
    
    print(f'Epochs: {epoch + 1}, Train Loss: {train_loss:.3f}, Val. Loss: {valid_loss:.3f}')


def translate(sentence, src_field, trg_field, model, device, max_len = 50):
    model.eval()
        
    tokens = [token.lower() for token in sentence]
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
        
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]    
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    src_len = torch.LongTensor([len(src_indexes)]).to(device)
    
    with torch.no_grad():
        context = model.encoder(src_tensor)
    
    hidden = context
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
                
        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden, context)
            
        pred_token = output.argmax(1).item()        
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    
    return trg_tokens[1:]


idx = 0

src = vars(valid_data.examples[idx])['src']
trg = vars(valid_data.examples[idx])['trg']

translation = translate(src, SRC, TRG, model, device)
translation = ''.join(translation[:-1])

src = ''.join(src)
trg = ''.join(trg)

print(f'src \t\t= {src}')
print(f'trg \t\t= {trg}')
print(f'predicted trg \t= {translation}')


def count_acc(dataset, SRC, TRG, model, device):
    count = 0

    for idx in range(len(dataset)):
        src = vars(dataset.examples[idx])['src']
        trg = vars(dataset.examples[idx])['trg']

        translation = translate(src, SRC, TRG, model, device)
        
        if translation[:-1] == trg:
            count += 1
    return count


train_acc_count = count_acc(train_data, SRC, TRG, model, device)
print(f'Accuracy on train set: {train_acc_count/len(train_data):.3f}, count: {train_acc_count:>5d}/{len(train_data):>5d}')

valid_acc_count = count_acc(valid_data, SRC, TRG, model, device)
print(f'Accuracy on valid set: {valid_acc_count/len(valid_data):.3f}, count: {valid_acc_count:>5d}/{len(valid_data):>5d}')

