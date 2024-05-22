import pandas as pd

# Load your dataset
data = pd.read_csv('resume_sentences.csv')
casual_sentences = data['casual'].tolist()
professional_sentences = data['professional'].tolist()

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Tokenize sentences
train_encodings = tokenizer(casual_sentences, truncation=True, padding=True)
val_encodings = tokenizer(professional_sentences, truncation=True, padding=True)

# Create PyTorch datasets
class ResumeDataset(torch.utils.data.Dataset):
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

# Training arguments
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

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model
trainer.train()

def professionalize_sentence(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return predicted_class

# Example usage
casual_sentence = "I worked on fixing bugs in the software."
professional_sentence = professionalize_sentence(casual_sentence)
print(professional_sentence)
