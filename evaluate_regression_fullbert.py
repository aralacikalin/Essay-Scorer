from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from datasets import load_dataset, Dataset, load_metric
from transformers import AutoModel, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
import torch.nn as nn
import pandas as pd
from typing import Optional
import string
import re
from transformers import TrainingArguments, Trainer
import numpy as np


class FakeNewsClassifierConfig(PretrainedConfig):
    model_type = "fakenews"

    def __init__(
            self,
            bert_model_name: str = 'distilbert-base-uncased',
            dropout_rate: float = 0.5,
            num_classes: int = 10,
            **kwargs) -> None:
        """Initialize the Fake News Classifier Confing.

        Args:
            bert_model_name (str, optional): Name of pretrained BERT model. Defaults to 'distilbert-base-uncased'.
            dropout_rate (float, optional): Dropout rate for the classification head. Defaults to 0.5.
            num_classes (int, optional): Number of classes to predict. Defaults to 2.
        """
        self.bert_model_name = bert_model_name
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        super().__init__(**kwargs)


class FakeNewsClassifierModel(PreTrainedModel):
    """DistilBERT based model for fake news classification."""

    config_class = FakeNewsClassifierConfig

    def __init__(self, config: PretrainedConfig) -> None:
        """Initialize the Fake News Classifier Model.

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
        # predictions = torch.argmax(logits, dim=-1).float().to(logits.device).requires_grad_()
        if domain1_score is not None:

            # print(logits.view(-1, self.num_labels).dtype)
            # print(rater1_domain1.view(-1).dtype)
            # loss_fn = nn.CrossEntropyLoss()
            loss_fn = nn.MSELoss()
            # print("")
            # print(logits.view(-1, self.num_labels))
            # print(rater1_domain1.long().to(logits.device))
            # print("")
            # loss = loss_fn(logits.view(-1, self.num_labels), F.one_hot(rater1_domain1.long(),num_classes=self.num_labels).view(-1,self.num_labels).float().to(logits.device))
            # loss = loss_fn(logits.view(-1, self.num_labels), domain1_score.long().to(logits.device))
            # print(predictions)
            loss = loss_fn(logits.view(-1), domain1_score.to(logits.device))
            # loss = loss_fn(logits.view(-1, self.num_labels), rater1_domain1.long().view(-1).to(logits.device))
            # loss=dice_loss(logits, F.one_hot(labels,num_classes=2))

        # return SequenceClassifierOutput(loss=loss, logits=logits)
        return SequenceClassifierOutput(loss=loss, logits=logits)


AutoConfig.register("fakenews", FakeNewsClassifierConfig)
AutoModelForSequenceClassification.register(FakeNewsClassifierConfig, FakeNewsClassifierModel)

# TODO: INSERT BEST CHECKPOINT
modelPath='/gpfs/space/home/aral/mtProject/results/fullbert-regressor/checkpoint-50500'
model = AutoModelForSequenceClassification.from_pretrained(modelPath
    )
# tokenizer = AutoTokenizer.from_pretrained(
#     '/gpfs/space/home/aral/nlpProject/results/4/checkpoint-268000')

tokenizer = AutoTokenizer.from_pretrained(modelPath)



testSets=[]
testSet = pd.read_csv("/gpfs/space/home/aral/mtProject/testSet.csv")

for i in range(1,9):
    essaySet=testSet[testSet.essay_set==i]
    essaySet=Dataset.from_pandas(essaySet)
    tokenizedDataset = essaySet.map(lambda examples: tokenizer(examples['essay'],truncation=True,padding=True), batched=True)
    testSets.append(tokenizedDataset)






training_args = TrainingArguments(
    output_dir='/gpfs/space/home/aral/nlpProject/results/4-res',
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=300,
    weight_decay=0.01,
    evaluation_strategy='steps',
    metric_for_best_model='kappa',
    greater_is_better=True,
    label_names = ["domain1_score"]
)

# metric = load_metric('glue', 'mrpc')


import sklearn


labelList=[None,13,7,4,4,5,5,31,61]
labelsList8=[i for i in range(61)]
scoreSum=0
for i in range(1,9):
  labelsList=[j for j in range(labelList[i])]


  def compute_metrics(eval_pred):
      logits, labels = eval_pred
      # predictions = np.argmax(logits, axis=-1)
      predictions = np.round(logits)


      return {"eval_kappa":sklearn.metrics.cohen_kappa_score(predictions, labels,weights="quadratic",labels=labelsList)}



  trainer = Trainer(
      model=model,
      args=training_args,
      # train_dataset=fullDataset,
      # eval_dataset=fullDataset_dev,
      tokenizer=tokenizer,
      compute_metrics=compute_metrics
  )




  results=trainer.evaluate(eval_dataset=testSets[i-1])
  print(labelsList)
  print(results)
  print("----------------------------------------------------------")
  scoreSum+=results["eval_kappa"]

print("Average eval_Kappa: ",scoreSum/8)
