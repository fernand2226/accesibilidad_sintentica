def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }

def process_data_to_model_inputs(batch):
  # Tokenize the input and target data
  inputs = tokenizer(batch["Product_Sf"], padding="max_length", truncation=True, max_length=encoder_max_length)
  outputs = tokenizer(batch["React01_Sf"], padding="max_length", truncation=True, max_length=decoder_max_length)

  batch["input_ids"] = inputs.input_ids
  batch["attention_mask"] = inputs.attention_mask
  batch["decoder_input_ids"] = outputs.input_ids
  batch["decoder_attention_mask"] = outputs.attention_mask
  batch["labels"] = outputs.input_ids.copy()

  batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

  return batch

import datasets
import transformers
import pandas as pd
import deepchem as dc
from datasets import Dataset

#Tokenizer
from transformers import RobertaTokenizerFast

#Encoder-Decoder Model
from transformers import EncoderDecoderModel


from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from dataclasses import dataclass, field
from typing import Optional

import os

# Loading the RoBERTa Tokenizer
from transformers import AutoTokenizer, AutoModelForMaskedLM, RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained("seyonec/BPE_SELFIES_PubChem_shard00_160k")
# Setting the BOS and EOS token
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token

#Set the path to the data folder, datafile and output folder and files
root_folder = '/mnt/c/Users/ferna/OneDrive/Escritorio/Proyecto/'
data_folder = os.path.abspath(os.path.join(root_folder, 'datasets/text_react'))
model_folder = os.path.abspath(os.path.join(root_folder, 'Projects/text_react/RoBERTa-FT-MLM'))
output_folder = os.path.abspath(os.path.join(root_folder, 'Projects/text_react'))
pretrainedmodel_folder = os.path.abspath(os.path.join(root_folder, 'Projects/text_react/RoBERTaMLM'))
tokenizer_folder = os.path.abspath(os.path.join(root_folder, 'Projects/text_react/TokRoBERTa'))

# Datafiles names containing training and test data
datafile= '/mnt/c/Users/ferna/OneDrive/Escritorio/Proyecto/USPTO_Complexity_50k.csv'
outputfile = 'prediction.csv'
datafile_path = os.path.abspath(os.path.join(data_folder,datafile))
outputfile_path = os.path.abspath(os.path.join(output_folder,outputfile))

# Load the dataset from a CSV file
df=pd.read_csv(datafile_path, usecols=[7,5])
print('Num Examples: ',len(df))
print('Null Values\n', df.isna().sum())


# Defining the train size. So 90% of the data will be used for training and the rest will be used for validation. 
train_size = 0.9
# Sampling 90% fo the rows from the dataset
train_dataset=df.sample(frac=train_size,random_state = 42)
# Reset the indexes
val_dataset=df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)
print('Length Train dataset: ', len(train_dataset))
print('Length Val dataset: ', len(val_dataset))

# To limit the training and validation dataset, for testing
max_train=28000
max_val=3200
# Create a Dataset from a pandas dataframe for training and validation
train_data=Dataset.from_pandas(train_dataset[:max_train])
val_data=Dataset.from_pandas(val_dataset[:max_val])

TRAIN_BATCH_SIZE = 16   # input batch size for training (default: 64)
VALID_BATCH_SIZE = 4    # input batch size for testing (default: 1000)
TRAIN_EPOCHS = 3       # number of epochs to train (default: 10)
VAL_EPOCHS = 1 
LEARNING_RATE = 1e-4    # learning rate (default: 0.01)
SEED = 42               # random seed (default: 42)
MAX_LEN = 128           # Max length for product description
SUMMARY_LEN = 7         # Max length for product names

# Loading the RoBERTa Tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained("seyonec/BPE_SELFIES_PubChem_shard00_160k",  max_len=MAX_LEN)
# Setting the BOS and EOS token
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token

batch_size=TRAIN_BATCH_SIZE  # change to 16 for full training
encoder_max_length=MAX_LEN
decoder_max_length=SUMMARY_LEN

# Preprocessing the training data
train_data = train_data.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["Product_Sf", "React01_Sf"]
)
train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)
# Preprocessing the validation data
val_data = val_data.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["Product_Sf", "React01_Sf"]
)
val_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)
# Shuffle the dataset when it is needed
#dataset = dataset.shuffle(seed=42, buffer_size=10, reshuffle_each_iteration=True)

# set encoder decoder tying to True
roberta_shared = EncoderDecoderModel.from_encoder_decoder_pretrained("seyonec/BPE_SELFIES_PubChem_shard00_160k", "seyonec/BPE_SELFIES_PubChem_shard00_160k", tie_encoder_decoder=True)
# Show the vocab size to check it has been loaded
print('Vocab Size: ',roberta_shared.config.encoder.vocab_size)

# set special tokens
roberta_shared.config.decoder_start_token_id = tokenizer.bos_token_id                                             
roberta_shared.config.eos_token_id = tokenizer.eos_token_id
roberta_shared.config.pad_token_id = tokenizer.pad_token_id

# sensible parameters for beam search
# set decoding params                               
roberta_shared.config.max_length = SUMMARY_LEN
roberta_shared.config.early_stopping = True
roberta_shared.config.no_repeat_ngram_size = 1
roberta_shared.config.length_penalty = 2.0
roberta_shared.config.repetition_penalty = 3.0
roberta_shared.config.num_beams = 10

# load rouge for validation
rouge = datasets.load_metric("rouge")

training_args = Seq2SeqTrainingArguments(
    output_dir=model_folder,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    predict_with_generate=True,
    #evaluate_during_training=True,
    evaluation_strategy="epoch",
    do_train=True,
    do_eval=True,
    logging_steps=1024,  
    save_steps=2048, 
    warmup_steps=1024,  
    #max_steps=1500, # delete for full training
    num_train_epochs = TRAIN_EPOCHS, #TRAIN_EPOCHS
    overwrite_output_dir=True,
    save_total_limit=1,
    fp16=True, 
)

# instantiate trainer
trainer = Seq2SeqTrainer(
    tokenizer=tokenizer,
    model=roberta_shared,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=val_data,
)

trainer.train()
trainer.save_model(model_folder)

# Load the dataset: sentence in english, sentence in spanish 
df=pd.read_csv(testfile_path, header=0)
print('Num Examples: ',len(df))
print('Null Values\n', df.isna().sum())
print(df.head(5))

test_data=Dataset.from_pandas(df)
print(test_data)

checkpoint_path = os.path.abspath(os.path.join(model_folder,'checkpoint-3072'))
print(checkpoint_path)

#Load the Tokenizer and the fine-tuned model
tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_folder)
model = EncoderDecoderModel.from_pretrained(model_folder)

model.to("cuda")


