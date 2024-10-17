import torch
import torch.nn as nn

class PlayerSelfAttention(nn.Module):
    
    def __init__(self, embed_size, heads):
    
        super(PlayerSelfAttention, self).__init__()
        
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size//heads

        assert(self.head_dim*heads == embed_size), "Embed size needs to be divisible by heads"

        # compute the values, keys and queries for all heads
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)

    def forward(self, values, keys, queries, mask=None):
    
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2).expand(N, 1, query_len, key_len)
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy/ (self.head_dim ** 0.5), dim = 3) # normalize accross the key_len

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads*self.head_dim)

        out = self.fc_out(out)

        return out, attention

class PlayerTransformerBlock(nn.Module):
    
    def __init__(self, embed_size, heads, dropout, forward_expansion) :
        
        super(PlayerTransformerBlock, self).__init__()
        
        self.attention = PlayerSelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size*forward_expansion),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask=None):
        
        attention, attention_matrix = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        
        return out, attention_matrix


class PlayerEncoder(nn.Module):
    
    def __init__(self, embed_size, num_layers, heads, forward_expansion, dropout, form_stats_size,
                  players_bank_size, teams_bank_size, n_positions, use_teams_embeddings = False):

        super(PlayerEncoder, self).__init__()

        self.embed_size = embed_size
        self.use_teams_embeddings = use_teams_embeddings

        self.form_embeddings = nn.Linear(form_stats_size, embed_size)
        self.players_embeddings = nn.Embedding(players_bank_size+1, embed_size, padding_idx = players_bank_size)
        if self.use_teams_embeddings:
            self.teams_embeddings = nn.Embedding(teams_bank_size+1, embed_size, padding_idx=teams_bank_size)
        self.positions_embeddings = nn.Embedding(n_positions+1, embed_size, padding_idx = n_positions)

        self.layers = nn.ModuleList([PlayerTransformerBlock(embed_size, heads, dropout, forward_expansion)
                                     for _ in range(num_layers)])

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, player_id, position_id, team_id, form_stats, attention_mask):

        if self.use_teams_embeddings:
            out = self.dropout(self.relu(self.players_embeddings(player_id))+\
                            self.form_embeddings(form_stats)+\
                            self.teams_embeddings(team_id)+\
                            self.positions_embeddings(position_id))
        
        else:
            out = self.dropout(self.relu(self.players_embeddings(player_id))+\
                            self.form_embeddings(form_stats)+\
                            self.positions_embeddings(position_id))

        attention_matrices = []
        for layer in self.layers:
            out, attention_matrix = layer(out, out, out, attention_mask)
            attention_matrices.append(attention_matrix)

        return out, attention_matrices