import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=6, dropout=0.1, device='cpu'):
        super(LSTMModel, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.device = device

        self.lstm = nn.LSTM(input_size=input_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers, 
                            dropout=dropout,
                            batch_first=True)
        
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h_0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim).to(self.device)
        c_0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim).to(self.device)
        
        output, (h_n, _) = self.lstm(x, (h_0, c_0))
        h_n = h_n[-1, :, :]
        out = self.fc_out(h_n)
        
        return out
    
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=6, dropout=0.1, device='cpu'):
        super(GRUModel, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.device = device
        
        self.gru = nn.GRU(input_size=input_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers, 
                            dropout=dropout,
                            batch_first=True)
        
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h_0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim).to(self.device)
        
        ula, h_out = self.gru(x, h_0)
        h_out = h_out[-1, :, :]
        out = self.fc_out(h_out)
        
        return out
    
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, num_layers=6, dim_ff=2048, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        # Define the Transformer model
        self.transformer = nn.Transformer(
            d_model=input_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, src, tgt):
        output = self.transformer(src, tgt)
        #output = output[:, -1, :] # For one output instead of sequence
        output = self.fc_out(output)
        return output