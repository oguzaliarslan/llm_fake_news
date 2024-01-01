import torch
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoModelForSequenceClassification , AutoTokenizer, Trainer, TrainingArguments
import pandas as pd
from sklearn.model_selection import train_test_split

class BERTTextClassifier:
    def __init__(self, model_name, X_train, y_train, X_test, y_test,token="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"):
        '''
        :param model_name Name of the model that is going to be trained
        :param X_train Training text
        :param y_train Training label
        :param X_test Testing text
        :param y_test Testing label
        '''
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.token = token
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
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

def main():
    parser = argparse.ArgumentParser(description="Generate predictions using the specified dataset.")
    parser.add_argument('--input_data', type=str, help='Path to input data file (CSV)')
    parser.add_argument('--model_name', type=str, help='Model name on Huggingface, e.g. "bert-base-uncased"')
    args = parser.parse_args()
    
    input_data = pd.read_csv(args.input_data)
    input_data.dropna()
    try:
        input_data = pd.read_csv(args.input_data)
        X_col_name = 'clean_text' if 'clean_text' in input_data.columns else 'text'
        
        X = list(input_data.dropna(subset=[X_col_name])[X_col_name])
        y = list(input_data.dropna(subset=[X_col_name])['label'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        
        model = BERTTextClassifier(model_name=args.model_name, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        model.train()
        model_output_dir = "model_results/bert_liar"
        model.model.save_pretrained(model_output_dir)
        model.tokenizer.save_pretrained(model_output_dir)
    except Exception as e:
        print(f"An error occurred: {e}")
        
if __name__ == "__main__":
    main()
