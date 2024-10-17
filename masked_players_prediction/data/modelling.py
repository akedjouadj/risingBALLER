import pickle
import json
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

config = json.load(open("config/statsbomb_dataset/config.json", "r"))

FORM_STATS_SIZE, PLAYERS_BANK_SIZE, TEAMS_BANK_SIZE, POSITION_BANK_SIZE = config["FORM_STATS_SIZE"], config["PLAYERS_BANK_SIZE"], config["TEAMS_BANK_SIZE"], config["POSITION_BANK_SIZE"]

LABEL2PLAYER_NAME = pickle.load(open(config["LABEL2PLAYER_NAME"], "rb"))
PLAYER_NAME2LABEL = pickle.load(open(config["PLAYER_NAME2LABEL"], "rb"))
LABEL2TEAM_NAME = pickle.load(open(config["LABEL2TEAM_NAME"], "rb"))
TEAM_NAME2LABEL = pickle.load(open(config["TEAM_NAME2LABEL"], "rb"))

class DataCollatorMaskedPM(Dataset):

    """
    A data collator for preparing masked player modeling data in football analytics.

    This class processes input data to create batches for training masked player models. 
    It masks a specified percentage of player IDs in the input data and prepares the 
    necessary features such as player IDs, team IDs, position IDs, form statistics, 
    and attention masks.

    Attributes:
        df_input (pd.DataFrame): The input DataFrame containing match data, player names, and statistics.
        player_pad_token_id (int): The token ID used for padding player IDs.
        player_mask_token_id (int): The token ID used to mask player IDs.
        team_pad_token_id (int): The token ID used for padding team IDs.
        position_pad_token_id (int): The token ID used for padding position IDs.
        player_name2label (dict): A mapping from player names to their corresponding label IDs.
        team_name2label (dict): A mapping from team names to their corresponding label IDs.
        max_length (int): The maximum length of input sequences for padding, allowing batch processing.
        mask_percentage (float): The percentage of players to be masked in the input.

    Args:
        df_input (pd.DataFrame): DataFrame containing the input players data. 

    """
    
    def __init__(self,
                 df_input,
                 player_pad_token_id=config["PLAYER_PAD_TOKEN_ID"],
                 player_mask_token_id=config["PLAYER_MASK_TOKEN_ID"],
                 team_pad_token_id=config["TEAM_PAD_TOKEN_ID"],
                 position_pad_toekn_id = config["POSITION_PAD_TOKEN_ID"],
                 player_name2label=PLAYER_NAME2LABEL,
                 team_name2label=TEAM_NAME2LABEL,
                 mask_percentage = 0.25):

        self.df_input = df_input
        self.player_pad_token_id = player_pad_token_id
        self.player_mask_token_id = player_mask_token_id
        self.team_pad_token_id = team_pad_token_id
        self.position_pad_token_id = position_pad_toekn_id
        self.player_name2label = player_name2label
        self.team_name2label = team_name2label
        self.max_length = 2*config["TEAM_MAX_LENGTH"]
        self.mask_percentage = mask_percentage

    def __len__(self):
        
        return self.df_input.match_id.nunique()

    def mask_players(self, match_input_player_id, match_output_player_id, match_input_form_stats, match_attention_mask, player_mask_token_id, mask_percentage):

        maskable_idx = np.where(match_attention_mask == 1)[0]

        number_masked_players = int(len(maskable_idx)*mask_percentage)

        masked_idx = np.random.choice(maskable_idx, number_masked_players, replace=False)
        non_masked_idx = [idx for idx in range(len(match_input_player_id)) if idx not in masked_idx]

        match_input_player_id[masked_idx] = player_mask_token_id
        match_input_form_stats[masked_idx] = 0
        match_output_player_id[non_masked_idx] = -100

        return match_input_player_id, match_output_player_id, match_input_form_stats

    def __getitem__(self, idx):

        """
        idx is the idx of an element in the dataset, a number between 0 and len(dataset)
        """
        
        #print(f"idx: {idx}")
        match_id = self.df_input.match_id.unique()[idx]
        match_input = self.df_input[self.df_input.match_id == match_id]

        match_teams = match_input.team_name.unique()
        match_input = pd.concat([match_input[match_input.team_name == match_teams[i]] for i in range(2)], ignore_index=True) # ensure that the players name in the same order as the input

        if len(match_teams) != 2:
            print (f"Error: match {match_id} contains {len(match_teams)} teams !")
            return None

        # encode the player_name to player_id
        match_input_player_name = match_input.player_name
        match_input_player_id = np.array([self.player_name2label[player_name] for player_name in match_input_player_name])
        match_input_player_id = np.pad(match_input_player_id, (0, self.max_length-len(match_input_player_id)), mode='constant', constant_values=self.player_pad_token_id)

        match_output_player_name = match_input.player_name
        match_output_player_id = np.array([self.player_name2label[player_name] for player_name in match_output_player_name])
        match_output_player_id = np.pad(match_output_player_id, (0, self.max_length-len(match_output_player_id)), mode='constant', constant_values=self.player_pad_token_id)

        # encode the team_name to team_id
        match_input_team_name = match_input.team_name
        match_input_team_id = [self.team_name2label[team_name] for team_name in match_input_team_name]
        match_input_team_id = np.pad(match_input_team_id, (0, self.max_length-len(match_input_team_id)), mode='constant', constant_values=self.team_pad_token_id)

        # spatial position id
        match_input_position_id = np.array(match_input.position_id)
        match_input_position_id = np.pad(match_input_position_id, (0, self.max_length-len(match_input_position_id)), mode='constant', constant_values=self.position_pad_token_id)

        # remove the id columns
        match_input = match_input.drop(columns=config["ID_COLUMNS"], axis=1)

        # add the attention mask depending on if the player is playing or not (mainly for padding)
        match_attention_mask = np.array(match_input.is_aligned)
        match_attention_mask = np.pad(match_attention_mask, (0, self.max_length-len(match_attention_mask)), mode='constant', constant_values=0)

        match_input = match_input.drop(columns=['is_aligned'], axis=1)

        # prepare the players form stats (TPE) for each player
        match_input_form_stats = np.array(match_input)
        match_input_form_stats = np.pad(match_input_form_stats, ((0, self.max_length-match_input_form_stats.shape[0]), (0, 0)), mode='constant', constant_values=0)

        # masking strategy, 15% of the players that are playing are masked, means 4.2 players per match in average
        match_input_player_id, match_output_player_id, match_input_form_stats = self.mask_players(match_input_player_id, match_output_player_id, match_input_form_stats, match_attention_mask,
                                                                                                  self.player_mask_token_id, self.mask_percentage)
        # return the dict of input and output data
        sample = {
                  'input_ids': torch.tensor(match_input_player_id, dtype=torch.long),
                  'labels': torch.tensor(match_output_player_id, dtype=torch.long),
                  'position_id': torch.tensor(match_input_position_id, dtype=torch.long),
                  'team_id': torch.tensor(match_input_team_id, dtype=torch.long),
                  'form_stats': torch.tensor(match_input_form_stats).float(),
                  'attention_mask': torch.tensor(match_attention_mask, dtype=torch.long),
                  }

        return sample

class PreprocessedDataCollatorMaskedPM(Dataset):

    def __init__(self, all_batches):
      self.data = all_batches

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        """
        idx an already preprocessed batch
        """

        return self.data[idx]