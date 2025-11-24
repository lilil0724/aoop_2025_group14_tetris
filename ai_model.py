import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TetrisTransformer(nn.Module):
    def __init__(self, board_height=20, board_width=10, d_model=64, nhead=4, num_layers=3, num_actions_rot=4, num_actions_x=10):
        super(TetrisTransformer, self).__init__()
        
        self.board_height = board_height
        self.board_width = board_width
        self.d_model = d_model
        
        # Embeddings
        # 0: Empty, 1: Filled, 2: Current Piece (if represented on board)
        self.cell_embedding = nn.Embedding(3, d_model) 
        
        # Piece type embedding (7 types: S, Z, I, O, J, L, T) + 1 for None/Padding
        self.piece_embedding = nn.Embedding(8, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model, max_len=board_height * board_width + 2)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Actor Heads
        self.actor_rot = nn.Linear(d_model, num_actions_rot)
        self.actor_x = nn.Linear(d_model, num_actions_x)
        
        # Critic Head
        self.critic = nn.Linear(d_model, 1)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, board, current_piece, next_piece):
        # board: (Batch, 20, 10) -> Flatten to (Batch, 200)
        # current_piece: (Batch,)
        # next_piece: (Batch,)
        
        batch_size = board.size(0)
        
        # Flatten board
        board_flat = board.view(batch_size, -1).long() # (Batch, 200)
        
        # Embeddings
        board_emb = self.cell_embedding(board_flat) # (Batch, 200, d_model)
        curr_piece_emb = self.piece_embedding(current_piece).unsqueeze(1) # (Batch, 1, d_model)
        next_piece_emb = self.piece_embedding(next_piece).unsqueeze(1) # (Batch, 1, d_model)
        
        # Concatenate: [CLS, Board..., Curr, Next]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # Sequence: CLS + 200 board cells + Current Piece + Next Piece
        x = torch.cat((cls_tokens, board_emb, curr_piece_emb, next_piece_emb), dim=1) # (Batch, 203, d_model)
        
        # Transpose for Transformer: (Seq_Len, Batch, d_model)
        x = x.permute(1, 0, 2)
        
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x)
        
        # Use CLS token output for prediction
        cls_output = output[0, :, :] # (Batch, d_model)
        
        # Heads
        rot_logits = self.actor_rot(cls_output)
        x_logits = self.actor_x(cls_output)
        value = self.critic(cls_output)
        
        return rot_logits, x_logits, value
