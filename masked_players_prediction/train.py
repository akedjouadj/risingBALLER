import pandas as pd
import numpy as np
import json

from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from tqdm import tqdm

from data.modelling import DataCollatorMaskedPM, PreprocessedDataCollatorMaskedPM
from model.risingballer import TransformerForMaskedPM
from utils import custom_collate_fn, compute_metrics, count_parameters

config = json.load(open("config/masked_players_prediction/config.json", "r"))

#----------------- Load the data -----------------#
datafile, sample_batch_size, repeat = config["datafile"], config["sample_batch_size"], config["repeat"]
dev_ratio, seed = config["dev_ratio"], config["seed"]

df_input = pd.read_csv('dataset/statsbomb/df_raw_counts_players_matches.csv')
my_dataset = DataCollatorMaskedPM(df_input)
my_dataloader = DataLoader(my_dataset, batch_size=sample_batch_size, shuffle=True, collate_fn=custom_collate_fn, drop_last = True)

# upsample the matches data
print("\nUpsampling data...")
all_batches = []
for count in tqdm(range(repeat)):
  # there will be a shuffling at each repetition and a different masking for each batch in the dataloader
  for batch in my_dataloader:
      all_batches.append(batch)

dev_size = int(dev_ratio*len(all_batches))

np.random.seed(seed)
dev_batches_idx = np.random.choice(range(len(all_batches)), dev_size, replace=False)
train_batches_idx = [idx for idx in range(len(all_batches)) if idx not in dev_batches_idx]

print(f"\nNumber of training matches: {len(train_batches_idx)}")
print(f"Number of validation matches: {len(dev_batches_idx)}")

dev_batches = [all_batches[idx] for idx in dev_batches_idx]
train_batches = [all_batches[idx] for idx in train_batches_idx]

preprocessed_batch_size = 1

dataset_train = PreprocessedDataCollatorMaskedPM(train_batches)
dataset_val = PreprocessedDataCollatorMaskedPM(dev_batches)

dataloader_train = DataLoader(dataset_train, batch_size=preprocessed_batch_size, shuffle=False)
dataloader_val = DataLoader(dataset_val, batch_size=preprocessed_batch_size, shuffle=False)


#----------------- Load the model -----------------#
embed_size, num_layers, num_heads = config["embed_size"], config["num_layers"], config["num_heads"]
model = TransformerForMaskedPM(embed_size=embed_size,
                                num_layers=num_layers,
                                heads=num_heads,
                                forward_expansion=4,
                                dropout=0.05)

count_parameters(model, print_table=True)

#----------------- Train the model -----------------#
output_dir, num_train_epochs, learning_rate, warmup_ratio = config["output_dir"], config["num_train_epochs"], config["learning_rate"], config["warmup_ratio"]
evaluation_strategy, eval_steps, logging_strategy, logging_steps = config["evaluation_strategy"], config["eval_steps"], config["logging_strategy"], config["logging_steps"]
save_strategy, save_steps = config["save_strategy"], config["save_steps"]

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=preprocessed_batch_size,
    per_device_eval_batch_size=preprocessed_batch_size,
    report_to="tensorboard",
    learning_rate=learning_rate,
    warmup_ratio=warmup_ratio,
    evaluation_strategy=evaluation_strategy,
    eval_steps=eval_steps, 
    logging_strategy=logging_strategy,
    logging_steps=logging_steps, 
    save_strategy = save_strategy,
    save_steps = save_steps,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    compute_metrics=compute_metrics
)

trainer.train()
