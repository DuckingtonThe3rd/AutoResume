import pandas as pd

data = pd.read_csv('resume_sentences.csv')
casual_sentences = data['casual'].tolist()
professional_sentences = data['professional'].tolist()

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(casual_sentences, truncation=True, padding=True)
val_encodings = tokenizer(professional_sentences, truncation=True, padding=True)

class ResumeDataset(torch.utils.data.Dataset): # type: ignore
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ResumeDataset(train_encodings, professional_sentences)
val_dataset = ResumeDataset(val_encodings, professional_sentences)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model, # type: ignore
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

def professionalize_sentence(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs) # type: ignore
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return predicted_class

casual_sentence = "I worked on fixing bugs in the software."
professional_sentence = professionalize_sentence(casual_sentence)
print(professional_sentence)
