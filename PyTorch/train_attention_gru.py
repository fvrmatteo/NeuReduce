import torch
import torch.nn as nn
import torch.optim as optim

import datetime
import random
import numpy as np
import torch.nn.functional as F

import torchtext
from torchtext.data import Field, BucketIterator
from torch.utils.tensorboard import SummaryWriter
import warnings

warnings.filterwarnings("ignore")

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def _tokenize(text):
    return [char for char in text]

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
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()        
        self.embedding = nn.Embedding(input_dim, emb_dim)     
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)        
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)     
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):        
        #src = [src len, batch size]        
        embedded = self.dropout(self.embedding(src))        
        #embedded = [src len, batch size, emb dim]        
        outputs, hidden = self.rnn(embedded)                
        #outputs = [src len, batch size, hid dim * num directions]
        #hidden = [n layers * num directions, batch size, hid dim]        
        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #outputs are always from the last layer        
        #hidden [-2, :, : ] is the last of the forwards RNN 
        #hidden [-1, :, : ] is the last of the backwards RNN  
        #initial decoder hidden is final hidden state of the forwards and backwards 
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))        
        #outputs = [src len, batch size, enc hid dim * 2]
        #hidden = [batch size, dec hid dim]        
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]        
        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)    
        encoder_outputs = encoder_outputs.permute(1, 0, 2)    
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2)))        
        #energy = [batch size, src len, dec hid dim]
        attention = self.v(energy).squeeze(2)        
        #attention= [batch size, src len]        
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention        
        self.embedding = nn.Embedding(output_dim, emb_dim)    
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)        
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):        
        #input = [batch size]
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]        
        input = input.unsqueeze(0)        
        #input = [1, batch size]        
        embedded = self.dropout(self.embedding(input))        
        #embedded = [1, batch size, emb dim]        
        a = self.attention(hidden, encoder_outputs)           
        #a = [batch size, src len]        
        a = a.unsqueeze(1)        
        #a = [batch size, 1, src len]        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)    
        #encoder_outputs = [batch size, src len, enc hid dim * 2]        
        weighted = torch.bmm(a, encoder_outputs)        
        #weighted = [batch size, 1, enc hid dim * 2]        
        weighted = weighted.permute(1, 0, 2)        
        #weighted = [1, batch size, enc hid dim * 2]        
        rnn_input = torch.cat((embedded, weighted), dim = 2)  
        #rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]            
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))        
        #output = [seq len, batch size, dec hid dim * n directions]
        #hidden = [n layers * n directions, batch size, dec hid dim]        
        #seq len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [1, batch size, dec hid dim]
        #hidden = [1, batch size, dec hid dim]
        #this also means that output == hidden
        assert (output == hidden).all()        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))        
        #prediction = [batch size, output dim]        
        return prediction, hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio = 0.5): 
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time        
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)        
        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)           
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]        
        for t in range(1, trg_len):            
            #insert input token embedding, previous hidden state and all encoder hidden states
            #receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, encoder_outputs)            
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
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, device).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)
            
model.apply(init_weights)


optimizer = optim.Adam(model.parameters())
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

def train(model, iterator, optimizer, criterion, clip): 
    model.train()
    epoch_loss = 0

    pbar = tqdm(iterator, unit='batchs', ncols=100)  
    for i, batch in enumerate(pbar):      
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


today = datetime.date.today()
from tqdm import tqdm

N_EPOCHS = 100
CLIP = 1

writer = SummaryWriter()

best_valid_loss = float('inf') 
for epoch in range(N_EPOCHS):   
    train_loss = train(model, train_iter, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iter, criterion)
    writer.add_scalar('train_loss', train_loss, epoch)
    writer.add_scalar('valid_loss', valid_loss, epoch)
    scheduler.step(valid_loss)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), f'attention-gru-{today}.pt')
    
    print(f'\nEpoch: {epoch + 1}, Train Loss: {train_loss:.3f}, Val. Loss: {valid_loss:.3f}')
writer.close()


def translate(sentence, src_field, trg_field, model, device, max_len = 50):
    model.eval()  
    tokens = [token.lower() for token in sentence]
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    src_len = torch.LongTensor([len(src_indexes)]).to(device)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
    
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden, encoder_outputs)
            
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    return trg_tokens[1:]


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
print(f'Accuracy rate on train set: {train_acc_count/len(train_data):.3f}')

valid_acc_count = count_acc(valid_data, SRC, TRG, model, device)
print(f'Accuracy rate on valid set: {valid_acc_count/len(valid_data):.3f}')