import torch
import numpy
from transformers import AutoModel, AutoTokenizer, BertTokenizer
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

from datasets import load_dataset
from datasets import Dataset
import pandas as pd
# data = pd.read_excel("/gpfs/space/home/aral/mtProject/training_set_rel3.xlsx")
# data=data.dropna(subset=['domain1_score'])
# dataset = Dataset.from_pandas(data)


# ROWSTOTAKE=320

# testSet=pd.DataFrame()
# for i in range(1,9):
#   dataToSample=data[data.essay_set==i]
#   sampledData=dataToSample.sample(n=ROWSTOTAKE)
#   testSet = pd.concat([testSet, sampledData], axis=0)

# trainSet=data[~data.essay_id.isin(testSet.essay_id)]
# trainSet = Dataset.from_pandas(trainSet)
# testSet = Dataset.from_pandas(testSet)

trainSet = pd.read_csv("/gpfs/space/home/aral/mtProject/newTrainSet.csv")
testSet = pd.read_csv("/gpfs/space/home/aral/mtProject/newTestSet.csv")

trainSet = Dataset.from_pandas(trainSet)
testSet = Dataset.from_pandas(testSet)

def addTokenLenght(example):
  tokenLength=len(example["input_ids"])
  example['text_lenght']=tokenLength
  return example

import string
def charCleaning(example):
  # specialChars=[]
  fullText=example['text']
  newFullText=""
  # example['text']=fullText.encode('ascii',errors='ignore')
  printable = set(string.printable)
  for char in fullText:
    if(not (char in printable)):
      newFullText+=" "
    else:
      newFullText+=char
  example['text']=newFullText

  return example


from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
# from transformers import BigBirdTokenizer
# tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")
# from transformers import PegasusTokenizerFast
# tokenizer = PegasusTokenizerFast.from_pretrained("hf-internal-testing/tiny-random-bigbird_pegasus")
"""Train Set Tokenization"""
tokenizedDataset = trainSet.map(lambda examples: tokenizer(examples['essay'],truncation=True,padding=True), batched=True)
# tokenizedDataset = dataset.map(lambda examples: tokenizer(examples['essay'],truncation=True,padding=True), batched=True)
print(tokenizedDataset.features)
fullTrainSet=tokenizedDataset.map(addTokenLenght)

"""Test Set Tokenization"""
tokenizedDataset = testSet.map(lambda examples: tokenizer(examples['essay'],truncation=True,padding=True), batched=True)
# tokenizedDataset = dataset.map(lambda examples: tokenizer(examples['essay'],truncation=True,padding=True), batched=True)
print(tokenizedDataset.features)
fullTestSet=tokenizedDataset.map(addTokenLenght)
raterVals=[]
for text in fullTestSet:
  if(text['essay']!=None):

    raterVals.append(text['rater1_domain1'])
  else:
    print(text["essay_id"])
max(raterVals)
from transformers import TrainingArguments, Trainer
from datasets import load_metric
from transformers import AutoModel, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
import torch.nn as nn
from typing import Optional
import numpy as np
import torch.nn.functional as F

class EssayScorerConfig(PretrainedConfig):
    model_type = "essayscorer"

    def __init__(
            self,
            bert_model_name: str = 'distilbert-base-uncased',
            dropout_rate: float = 0.5,
            num_classes: int = 10,
            **kwargs) -> None:
        """Initialize the Essay Scorer Config.

        Args:
            bert_model_name (str, optional): Name of pretrained BERT model. Defaults to 'distilbert-base-uncased'.
            dropout_rate (float, optional): Dropout rate for the classification head. Defaults to 0.5.
            num_classes (int, optional): Number of classes to predict. Defaults to 2.
        """
        self.bert_model_name = bert_model_name
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        super().__init__(**kwargs)


class EssayScorerModel(PreTrainedModel):
    """DistilBERT based model for essay scoring."""

    config_class = EssayScorerConfig

    def __init__(self, config: PretrainedConfig) -> None:
        """Initialize the Essay Scorer Model.

        Args:
            config (PretrainedConfig): Config with model's hyperparameters.
        """
        super().__init__(config)

        # self.num_labels = config.num_labels
        self.num_labels = config.num_classes

        self.bert = AutoModel.from_pretrained(config.bert_model_name)
        self.clf = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size),
            nn.ELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(self.bert.config.hidden_size, config.num_classes)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,domain1_score) -> SequenceClassifierOutput:
        bert_output = self.bert(input_ids, attention_mask)

        # torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)
        last_hidden_state = bert_output[0]

        # torch.FloatTensor of shape (batch_size, hidden_size)
        pooled_output = last_hidden_state[:, 0]

        # has_company_logo=torch.reshape(has_company_logo,(has_company_logo.size(0),1))
        # telecommuting=torch.reshape(telecommuting,(telecommuting.size(0),1))
        # text_lenght=torch.reshape(text_lenght,(text_lenght.size(0),1))
        # has_questions=torch.reshape(has_questions,(has_questions.size(0),1))

        # torch.FloatTensor of shape (batch_size, num_labels)
        # print(pooled_output.size(),has_company_logo.size(),has_questions.size(),telecommuting.size(),text_lenght.size())
        logits = self.clf(pooled_output)

        loss = None
        # print(rater1_domain1.shape)
        # print(logits.shape)
        if domain1_score is not None:

            # print(logits.view(-1, self.num_labels).dtype)
            # print(rater1_domain1.view(-1).dtype)
            loss_fn = nn.CrossEntropyLoss()
            # print("")
            # print(logits.view(-1, self.num_labels))
            # print(rater1_domain1.long().to(logits.device))
            # print("")
            # loss = loss_fn(logits.view(-1, self.num_labels), F.one_hot(rater1_domain1.long(),num_classes=self.num_labels).view(-1,self.num_labels).float().to(logits.device))
            loss = loss_fn(logits.view(-1, self.num_labels), domain1_score.long().to(logits.device))
            # loss = loss_fn(logits.view(-1, self.num_labels), rater1_domain1.long().view(-1).to(logits.device))
            # loss=dice_loss(logits, F.one_hot(labels,num_classes=2))

        return SequenceClassifierOutput(loss=loss, logits=logits)

hyperparams = {
    'bert_model_name': 'distilbert-base-uncased',
    'dropout_rate': 0.15,
    'num_classes': 61
}
config = EssayScorerConfig(**hyperparams)
model = EssayScorerModel(config)

training_args = TrainingArguments(
    output_dir='/gpfs/space/home/aral/mtProject/results/distilbert-smarttrunc-classifier',
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=1298,
    num_train_epochs=100,
    weight_decay=0.01,
    evaluation_strategy='steps',
    metric_for_best_model='kappa',
    greater_is_better=True,
    label_names = ["domain1_score"]
)

metric = load_metric("accuracy")
import sklearn

labelsList=[i for i in range(61)]

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"eval_kappa":sklearn.metrics.cohen_kappa_score(predictions, labels,weights="quadratic",labels=labelsList)}



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=fullTrainSet,
    eval_dataset=fullTestSet,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()