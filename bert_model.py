import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import pandas as pd

class BERTTextClassifier:
    def __init__(self, token, model_name, X_train, y_train, X_test, y_test):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.token = token
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)
        
        self.train_encodings = self.tokenizer(X_train, truncation=True, padding=True, max_length=256)
        self.test_encodings = self.tokenizer(X_test, truncation=True, padding=True, max_length=256)
        
        self.train_dataset = self.TextDataset(self.train_encodings, y_train)
        self.test_dataset = self.TextDataset(self.test_encodings, y_test)
        
        self.training_args = TrainingArguments(
            output_dir='./results_bert',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
        )
    
    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)
    
    def train(self):
        self.trainer.train()
    
    def evaluate(self):
        predictions = self.trainer.predict(self.test_dataset)
        probabilities = torch.nn.functional.softmax(torch.from_numpy(predictions.predictions), dim=-1)
        predicted_labels = torch.argmax(probabilities, dim=1)

        acc = accuracy_score(self.test_dataset.labels, predicted_labels)
        precision = precision_score(self.test_dataset.labels, predicted_labels)
        recall = recall_score(self.test_dataset.labels, predicted_labels)
        f1 = f1_score(self.test_dataset.labels, predicted_labels)
        
        return acc, precision, recall, f1

