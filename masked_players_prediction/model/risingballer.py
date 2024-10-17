import json

import torch.nn as nn

from transformers.modeling_outputs import MaskedLMOutput

from model.risingballer_utils import PlayerEncoder

config = json.load(open("config/statsbomb_dataset/config.json", "r"))

FORM_STATS_SIZE, PLAYERS_BANK_SIZE, TEAMS_BANK_SIZE, POSITION_BANK_SIZE = config["FORM_STATS_SIZE"], config["PLAYERS_BANK_SIZE"], config["TEAMS_BANK_SIZE"], config["POSITION_BANK_SIZE"]

class TransformerForMaskedPM(nn.Module):

    """
    A Transformer model designed for masked player prediction task in football analytics.

    This model encodes player data and produces embeddings for players
    using a multi-layer transformer architecture. It is specifically tailored to handle 
    masked language modeling scenarios where the goal is to predict the index of the masked players 
    in the inputs.

    Original Args:
        form_stats_size (int, optional): The number of raw statistics available per player.
        players_bank_size (int, optional): The number of unique players in the dataset, including the 
                                            mask and pad tokens, counting from 0.
        teams_bank_size (int, optional): The number of unique teams in the dataset, including the
                                            pad token, counting from 0.
        n_positions (int, optional): The number of unique position label in the dataset, including the
                                            pad token, counting from 0. Default to 25 from statsbomb 
                                            annotation format. 
    
    """

    def __init__(self, embed_size, num_layers, heads, forward_expansion, dropout, form_stats_size= FORM_STATS_SIZE,
                  players_bank_size = PLAYERS_BANK_SIZE, teams_bank_size = TEAMS_BANK_SIZE,
                  n_positions = POSITION_BANK_SIZE):

        super(TransformerForMaskedPM, self).__init__()

        self.players_bank_size = players_bank_size

        self.player_encoder = PlayerEncoder(embed_size, num_layers, heads, forward_expansion, dropout, form_stats_size,
                                            players_bank_size, teams_bank_size, n_positions)

        self.decoder = nn.Linear(embed_size, players_bank_size)

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, labels, position_id, team_id, form_stats, attention_mask):

        input_ids = input_ids.squeeze(0)
        labels = labels.squeeze(0)
        position_id = position_id.squeeze(0)
        team_id = team_id.squeeze(0)
        form_stats = form_stats.squeeze(0)
        attention_mask = attention_mask.squeeze(0)

        players_embeddings, attention_matrices = self.player_encoder(input_ids, position_id, team_id, form_stats, attention_mask)

        output = self.decoder(players_embeddings)
        
        loss = self.criterion(output.view(-1, self.players_bank_size), labels.view(-1))


        return MaskedLMOutput(loss = loss,
                              logits = output,
                              hidden_states = players_embeddings,
                              attentions=attention_matrices)
