import numpy as np
import torch

from prettytable import PrettyTable

def compute_metrics(eval_pred):

    model_output, labels = eval_pred 
    pred, players_embeddings, attention_matrices = model_output 

    #print(labels.shape, pred.shape)

    len_dev_batches, bs_sample, n_players = labels.shape

    labels = labels.reshape(len_dev_batches*bs_sample, n_players)
    labels = labels.reshape(len_dev_batches*bs_sample*n_players,)

    pred = pred.reshape(pred.shape[0]*n_players, -1)

    # remove the padding tokens
    mask_non_pad_players = labels != -100
    labels = labels[mask_non_pad_players]
    pred = pred[mask_non_pad_players]

    #print(labels.shape)
    #print(pred.shape)

    # find the most likely predicted player
    pred_top1_idx = np.argmax(pred, axis=1)

    # find the top 3 most likely predicted players
    pred_top3_idx = np.argsort(pred, axis=1)[:, -3:]

    # compute the model top 1 accuracy
    accuracy_top1 = (labels==pred_top1_idx).mean()

    # compute the model top3 accuracy
    accuracy_top3 = 0
    for label, pred_top3 in zip(labels, pred_top3_idx):
        if label in pred_top3:
            accuracy_top3 += 1
    accuracy_top3 /= len(labels)

    outputs = {'accuracy_top1': accuracy_top1,
               'accuracy_top3': accuracy_top3}

    return outputs

def custom_collate_fn(batch):

    # Filter None
    batch = [item for item in batch if item is not None]
    
    return torch.utils.data.dataloader.default_collate(batch)

def count_parameters(model, print_table = False):

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    
    if print_table:
        print(table)
    
    print(f"Total Trainable Params: {total_params}")
    
    return total_params

