import torch
from torch import nn
from ffnn import FFNN

class BiLSTM_FFNN_PRE(nn.Module):
    def __init__(self, embeddings, bilstm_hidden_size, ffnn_hidden_size, ffnn_output_size, freeze = True):
        super(BiLSTM_FFNN_PRE, self).__init__()
        _, self.input_size = embeddings.size()
        self.bilstm_hidden_size = bilstm_hidden_size
        self.ffnn_hidden_size = ffnn_hidden_size
        self.ffnn_output_size = ffnn_output_size
        self.freeze = freeze

        self.emb = nn.Embedding.from_pretrained(embeddings, self.freeze)
        
        self.bilstm = nn.LSTM(self.input_size,self.bilstm_hidden_size, bidirectional=True)
        self.ffnn = FFNN(self.bilstm_hidden_size*2,self.ffnn_hidden_size, self.ffnn_output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, data, lengths):
        sequence_embed = self.emb(data)
        pack = torch.nn.utils.rnn.pack_padded_sequence(
            sequence_embed,
            lengths, enforce_sorted=False
        )
        packed_output,_ = self.bilstm(pack)
        output,_ = torch.nn.utils.rnn.pad_packed_sequence(packed_output)

        for i in range(output.size()[1]):
            if i==0:
                vec = output.index_select(1,torch.tensor(i)).squeeze(1).index_select(0,torch.tensor(lengths[i]-1)).mean(dim=0).unsqueeze(0)
            else:
                vec = torch.cat((vec,output.index_select(1,torch.tensor(i)).squeeze(1).index_select(0,torch.tensor(lengths[i]-1)).mean(dim=0).unsqueeze(0)),0)

        ffnn = self.ffnn(vec)
        return self.log_softmax(ffnn)