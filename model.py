import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, RobertaTokenizer, XLNetTokenizer
from transformers import BertForSequenceClassification, RobertaForSequenceClassification, XLNetModel
import math
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys

if torch.cuda.is_available():
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    ByteTensor = torch.cuda.ByteTensor
else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    ByteTensor = torch.ByteTensor
def mask_emotion_logits(logits, change_count, curr_emotion, confusion_map):
    # logits: [T, B, C], change_count: [T, B], curr_emotion: [T, B]
    T, B, C = logits.size()
    mask = torch.zeros_like(logits)

    for t in range(T):
        for b in range(B):
            cc = change_count[t, b].item()
            label = curr_emotion[t, b].item()
            if cc < 2:
                allowed = confusion_map[label]  # e.g. [label, 4] means the label itself and a commonly confused class
            else:
                allowed = list(range(C))
            mask[t, b, allowed] = 1.0
    logits = logits.masked_fill(mask == 0, -1e9)
    return logits

class MaskedNLLLoss(nn.Module):
    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')      ## nn.NLLLoss
        '''
        nn.NLLLoss
        Official docs: nn.NLLLoss takes log-probabilities and a target label.
        Relationship with nn.CrossEntropyLoss: softmax(x) + log + nn.NLLLoss == nn.CrossEntropyLoss
        '''
    def forward(self, pred, target, mask):
        '''
        param pred: (batch_size, num_utterances, n_classes)
        param target: (batch_size, num_utterances)
        param mask: (batch_size, num_utterances)
        '''
        mask_ = mask.view(-1,1) 
        if type(self.weight)==type(None):
            loss = self.loss(pred*mask_, target)/torch.sum(mask)
        else:
            loss = self.loss(pred*mask_, target)/torch.sum(self.weight[target]*mask_.squeeze())
        return loss

# This part references the github project: https://github.com/declare-lab/conv-emotion
class EncoderModel(nn.Module):
    def __init__(self, D_h, cls_model, transformer_model_family, mode, attention=False, residual=False):
        '''
        param transformer_model_family: bert or roberta or xlnet
        param mode: 0(base) or 1(large)
        '''
        super().__init__()
        
        if transformer_model_family == 'bert':
            if mode == '0':
                model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                hidden_dim = 768
            elif mode == '1':
                model = BertForSequenceClassification.from_pretrained('bert-large-uncased')
                tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
                hidden_dim = 1024       
        elif transformer_model_family == 'roberta':
            if mode == '0':
                model = RobertaForSequenceClassification.from_pretrained('roberta-base')
                tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
                hidden_dim = 768
            elif mode == '1':
                model = RobertaForSequenceClassification.from_pretrained('roberta-large')
                tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
                hidden_dim = 1024      
        elif transformer_model_family == 'xlnet':
            if mode == '0':
                model = XLNetModel.from_pretrained('xlnet-base-cased')
                tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
                hidden_dim = 768

        self.transformer_model_family = transformer_model_family
        self.model = model.cuda()
        self.hidden_dim = hidden_dim
        self.cls_model = cls_model
        self.D_h = D_h
        self.residual = residual
        if transformer_model_family == 'xlnet':
            if mode == '0':
                self.model.mem_len = 900
                self.model.attn_type = 'bi'
        
        if self.transformer_model_family in ['bert', 'roberta', 'xlnet']:
            self.tokenizer = tokenizer
        
    def pad(self, tensor, length):
        if length > tensor.size(0):
            return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:]).cuda()])
        else:
            return tensor
    def forward(self, conversations, lengths, umask, qmask):
        '''
        param conversations: list of conversations, each a list of utterances
        param lengths: list of conversation lengths
        returns: semantic vectors encoded by BERT and a mask
        '''
        # collect all utterances from multiple conversations
        lengths = torch.Tensor(lengths).long()
            # tokenize
        start = torch.cumsum(torch.cat((lengths.data.new(1).zero_(), lengths[:-1])), 0)
        utterances = [sent for conv in conversations for sent in conv]
        if self.transformer_model_family in ['bert', 'roberta']:
            # feed into BERT and use the [CLS] token as the sentence embedding
            batch = self.tokenizer(utterances, padding=True, return_tensors="pt") 
            input_ids = batch['input_ids'].cuda()
            features = outputs.hidden_states[-1][:, 0, :]  # take only the [CLS] vector
                # if features is 2D, do not index further
            features = outputs.hidden_states[-1][:, 0, :]  # only take [CLS] vector
            if self.transformer_model_family == 'roberta' and features.dim() == 2:
                pass
            elif self.transformer_model_family == 'roberta':
                features = features[:, 0, :]
        elif self.transformer_model_family == 'xlnet':
            batch = self.tokenizer(utterances, padding=True, return_tensors="pt")
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            features, new_mems = self.model(input_ids, None)[:2]
            features = features[:, -1, :]
        # rearrange output features into batch form: (total_utterances, hidden_dim) -> (utterances_per_conv, batch_size, hidden_dim)
        features = torch.stack([self.pad(features.narrow(0, s, l), max(lengths))
                                for s, l in zip(start.data.tolist(), lengths.data.tolist())], 0).transpose(0, 1)
        # compute mask tensor
        umask = umask.cuda()
        mask = umask.unsqueeze(-1).type(FloatTensor) # (batch, num_utt) -> (batch, num_utt, 1)
        mask = mask.transpose(0, 1) # (batch, num_utt, 1) -> (num_utt, batch, 1)
        mask = mask.repeat(1, 1, 2*self.D_h) #  (num_utt, batch, 1) -> (num_utt, batch, output_size)
        return features, umask, mask


class LstmLayer(nn.Module):
    def __init__(self, hidden_dim, D_h):
        super().__init__()
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=D_h,num_layers=2, bidirectional=True).cuda() #biLSTM
    def forward(self, features, mask):
        hidden, _ = self.lstm(features)
        hidden = hidden * mask 
        return hidden # return biLSTM sequentially encoded features
        
# This part references the blog: https://blog.csdn.net/qq_44766883/article/details/112008655
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position_encoding = np.array([[pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)] for pos in range(max_len)])
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
        pe = torch.from_numpy(position_encoding).float()
        pe.cuda()
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        '''
        param x: (utterances_num, batch_size, d_model)
        '''
        x = torch.transpose(x,0,1)
        x = x + self.pe[:, :x.size(1)]
        x = torch.transpose(x,0,1)
        return self.dropout(x)

class LSTMReducer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.smax_fc = nn.Linear(input_dim,6)  # used to reduce offset feature dimensionality
        self.smax_fc = self.smax_fc.cuda()
    def forward(self, x):
        logits = self.smax_fc(x)  # adjust offset part dimensionality
        print("logits shape",logits.shape)
        return  logits

class LinearClassifer(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.smax_fc = nn.Linear(input_dim, num_classes)  # used to compute softmax logits for main task
        self.smax_fc = self.smax_fc.cuda()
        self.vad_fc = nn.Linear(input_dim, 3)  # VAD regression
        self.vad_fc = self.vad_fc.cuda()
    def forward(self, x):
        logits = self.smax_fc(x)    # mian task
        log_prob = F.log_softmax(logits, dim=-1)
        vad_pred = self.vad_fc(x)  # shape: (batch_size, con_len, 3)
        # no logits
        return log_prob, vad_pred

class VADClassifer(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.vad_fc = nn.Linear(input_dim, 3)  # VAD regression
        self.vad_fc = self.vad_fc.cuda()
    def forward(self, x):
        vad_pred = self.vad_fc(x) 
        return vad_pred
  
class JointModel(nn.Module):
    def __init__(self, D_h, cls_model, transformer_model_family, mode, num_classes, num_subclasses, context_encoder_layer, attention=False, residual=False, no_lstm=False):
        '''
        param D_h: LSTM hidden size
        param transformer_model_family: bert or roberta or xlnet
        param mode: 0(base) or 1(large)
        param num_classes: number of emotion classes
        param num_subclasses: number of emotion bias classes
        param context_encoder_layer: number of context transformer layers
        '''
        super().__init__()
        # sentence encoder
        self.encoderModel = EncoderModel(D_h, cls_model, transformer_model_family, mode, attention, residual)
        # emotion shift perception task
        if mode == '0':
            hidden_dim = 768
        elif mode == '1':
            hidden_dim = 1024
        self.lstmLayer = LstmLayer(hidden_dim, D_h)
        self.LSTMReducer = LSTMReducer(D_h*2, 6)
        self.VADClassifer = VADClassifer(D_h*2, 3) # VAD predictor for subtask
        if mode =='0':
            self.tfLayer = nn.TransformerEncoderLayer(d_model=D_h*2 + hidden_dim, nhead=1).cuda()  # (768+3)/3
        elif mode == '1':
            self.tfLayer = nn.TransformerEncoderLayer(d_model= 0 + hidden_dim, nhead=1,dropout=0.3).cuda()  # (d_model+hidden_dim)/nhead  (d_model+hidden_dim)/nhead
        self.tfNorm = nn.LayerNorm(0 + hidden_dim)
        self.tfEncoder = nn.TransformerEncoder(self.tfLayer, context_encoder_layer, norm=self.tfNorm).cuda()
        # main emotion recognition task
        self.mainClassifer = LinearClassifer(0+ hidden_dim, num_classes)
        self.subClassifer = LinearClassifer(0+ hidden_dim, 3)  # VAD predictor for subtask
        self.delta_guidance = nn.Sequential(nn.Linear(1, 128),nn.ReLU(),nn.Linear(128,1),nn.Sigmoid()).cuda()        
        self.delta_vad_proj = nn.Sequential(nn.Linear(3, 128),nn.ReLU(),nn.Linear(128,128)).cuda()  # projection network: 3 -> 128
        self.vad_to_feature = nn.Sequential(nn.Linear(128, 1024),   # projection network: 128 -> 1024
            nn.Tanh()  # optional activation
        ).cuda()
        # ΔVAD 
        self.delta_v_fc = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        ).cuda()

        self.delta_a_fc = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        ).cuda()

        self.delta_d_fc = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        ).cuda()
        
    def find_last(self, utterances, index):
        '''
        param utterances: semantic features tensor, (utterances_per_conv, batch_size, hidden_dim)
        param index: previous-utterance index of the same speaker, tensor, (utterances_per_conv, batch_size)
        returns: per-utterance difference vectors
        This function extracts the previous-utterance feature for each utterance according to index
        and computes the difference. Used to capture context for emotion prediction.
        '''
        utterances = utterances.cuda()
        index = index.cuda()
        # check that utterances and index have matching batch sizes
        if utterances.size(1) != index.size(1):
            raise ValueError(f"Batch size of utterances ({utterances.size(1)}) does not match batch size of index ({index.size(1)})")
        # prepend a zero vector as the "previous utterance" for the first utterance
        # note: torch.zeros shape must match utterances
        zero_tensor = torch.zeros(1, utterances.size(1), utterances.size(2)).cuda()
        utterances_ = torch.cat((zero_tensor, utterances), dim=0)
        # add 1 to index so that the first utterance's previous index points to the zero vector
        index_ = index + 1  
        # gather the previous-utterance features from utterances_
        # the gather indices are provided by index_
        last_utterances = utterances_.gather(0, index_.unsqueeze(2).expand(-1, -1, utterances.size(2)))
        # compute the difference between current and previous utterance
        sub = torch.sub(utterances, last_utterances) 
        # Example alternative: compute attention weights between current and history
        # attention_weights = F.softmax(torch.matmul(utterances, utterances.transpose(1, 2)), dim=-1)  # (batch, seq_len, seq_len)
        # weighted_history = torch.matmul(attention_weights, utterances)  # (batch, seq_len, hidden_dim)
        # sub = torch.sub(utterances, weighted_history)
        
        return sub
    def forward(self, conversations, subindex, lengths, umask, qmask):

        '''
        param conversations: list of conversations, each a list of utterances
        param lengths: list of conversation lengths
        param subindex: previous-utterance indices for same speaker, tensor (utterances_per_conv, batch)
        returns: outputs for main and auxiliary tasks
        '''
        # sentence encoder
        features, umask, mask = self.encoderModel(conversations, lengths, umask, qmask) # encode utterances with BERT
        # emotion shift detection task
        subdata = self.find_last(features, subindex) # use subindex to get previous-utterance feature differences
        lstm_features = self.lstmLayer(subdata, mask)
        delta_vad_pred = self.VADClassifer(lstm_features)
      
        delta_v = delta_vad_pred[:, :, 0:1]
        delta_a = delta_vad_pred[:, :, 1:2]
        delta_d = delta_vad_pred[:, :, 2:3]
        delta_v_enc = self.delta_v_fc(delta_v)
        delta_a_enc = self.delta_a_fc(delta_a)
        delta_d_enc = self.delta_d_fc(delta_d)
        delta_vad_fused = delta_v_enc + delta_a_enc + delta_d_enc  # [T, B, 128]

        delta_vad_residual = self.vad_to_feature(delta_vad_fused)
        delta_magnitude = torch.norm(delta_vad_residual, dim=-1, keepdim=True)  # [B, T, 1]
        guide_weight = self.delta_guidance(delta_magnitude)
        fushion_features = features * guide_weight # apply guide_weight as a soft mask to features (can guide attention)
        tf_features = self.tfEncoder(fushion_features) # transformer context encoder
        main_output, _  = self.mainClassifer(tf_features) # classifier: main_output are log-probabilities for emotion classes
        _,vad_pred = self.subClassifer(tf_features)

        delta_v_pred = delta_vad_pred[..., 0]  # valence
        delta_a_pred = delta_vad_pred[..., 1]  # arousal
        delta_d_pred = delta_vad_pred[..., 2]  # dominance
        # split VAD predictions
        v_pred = vad_pred[..., 0]  # valence
        a_pred = vad_pred[..., 1]  # arousal
        d_pred = vad_pred[..., 2]  # dominance

        return main_output,  delta_v_pred, delta_a_pred, delta_d_pred,v_pred, a_pred, d_pred,tf_features
