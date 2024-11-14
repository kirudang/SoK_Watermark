import os
cache_dir = '.../cache'  # Change this to your desired cache directory
os.environ['HF_HOME'] = cache_dir

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import json
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import pandas as pd


# Ensure deterministic behavior
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(100)

# Define data load
def load_data(file_path, type_of_data, type_of_label):
    with open(file_path, 'r') as f:
        all_data = json.load(f)
    
    train_texts, train_labels, test_texts, test_labels = [], [], [], []
    
    for label in type_of_label:
        # Filter based on 'type_of_data' and current 'label'
        label_data = [item for item in all_data if type_of_data in item and item['label'] == label]
        
        # Splitting into training and testing based on 'train' flag
        train_texts += [item[type_of_data] for item in label_data if item['train'] == 1]
        train_labels += [item['label'] for item in label_data if item['train'] == 1]
        
        test_texts += [item[type_of_data] for item in label_data if item['train'] == 0]
        test_labels += [item['label'] for item in label_data if item['train'] == 0]
    
    return train_texts, train_labels, test_texts, test_labels


# Define the dataset
type_of_data =  "input_output"  #"output_only" #
# 2: llama2,                     3: mistral,               5: OPT 7B,               7: Falcon 7B                
# 4: ICML watermarked llama2,   21: ICML wm Mistral,      30: ICML wm OPT 7B,       22: ICML wm Falcon 7B
# 15: Distortion wm llama 2     19: Distortion wm Mistral 31: Distortion wm OPT 67  32: Distortion wm Falcon
# 33: SIR llama 2               34: SIR Mistral           35: SIR OPT 67            36: SIR Falcon 7B
# 37: Semstamp llama 2          38: Semstamp Mistral      39: Semstamp OPT 67       40: Semstamp Falcon 7B
# 41: Adaptive llama 2          42: Adaptive Mistral      43: Adaptive OPT          44: Adaptive Falcon
# 45: Uni llama 2               46: Uni Mistral           47: Uni OPT               48: Uni Falcon
# 49: Unbias llama 2            50: Unbias Mistral        51: Unbias OPT            52: Unbias Falcon
# 53: UPV llama 2               54: UPV Mistral           55: UPV OPT               56: UPV Falcon

type_of_label = [53,54,55,56]  # or any other labels you want to filter


# Load and prepare the data
data_file = ' your path to the data file'  # Change this to the path of your data file
train_texts, train_labels, test_texts, test_labels = load_data(data_file, type_of_data, type_of_label)

# Encode labels
def encode_labels(labels):
    encoder = LabelEncoder()
    return encoder.fit_transform(labels), encoder

encoded_train_labels, label_encoder = encode_labels(train_labels)
encoded_test_labels = label_encoder.transform(test_labels)
# Check the classes
print("Classes:", label_encoder.classes_)
print("Encoded classes:", label_encoder.transform(label_encoder.classes_))
# Check distribution of classes in training data
unique, counts = np.unique(train_labels, return_counts=True)
print("Train labels distribution:", dict(zip(unique, counts)))

# Check distribution of classes in test data
unique, counts = np.unique(test_labels, return_counts=True)
print("Test labels distribution:", dict(zip(unique, counts)))

# Tokenizer instantiation
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Tokenize training and testing texts
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=400, return_tensors="pt")
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=400, return_tensors="pt")

class TextClassificationDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        """
        Initializes the dataset.
        :param input_ids: Encoded input IDs from the tokenizer.
        :param attention_masks: Attention masks from the tokenizer to ignore padding.
        :param labels: Encoded labels for the classification task.
        """
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        """
        Returns the size of the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieves the ith item from the dataset.
        :param idx: The index of the item to retrieve.
        :return: A dictionary containing the input_ids, attention_mask, and labels for the item.
        """
        item = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)  # Ensuring labels are torch tensors
        }
        return item

# Create datasets
train_dataset = TextClassificationDataset(train_encodings['input_ids'], train_encodings['attention_mask'], encoded_train_labels)
test_dataset = TextClassificationDataset(test_encodings['input_ids'], test_encodings['attention_mask'], encoded_test_labels)

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16)


# Define the LSTM Classifier
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, bidirectional, dropout):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None):
        embedded = self.dropout(self.embedding(input_ids))
        lstm_output, (hidden, cell) = self.lstm(embedded)
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        else:
            hidden = hidden[-1,:,:]
        outputs = self.fc(self.dropout(hidden))
        return outputs


# Define training functions
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        predictions = torch.argmax(outputs, dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    return total_loss / len(dataloader), f1, accuracy


# Define evaluation functions
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    return total_loss / len(dataloader), f1, accuracy, all_predictions, all_labels

# Save the model and label encoder
def save_model_and_artifacts(model, label_encoder, model_path, encoder_path):
    torch.save(model.state_dict(), model_path)
    with open(encoder_path, 'wb') as le_file:
        pickle.dump(label_encoder, le_file)


# Instantiate the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
input_dim = tokenizer.vocab_size
hidden_dim = 128
output_dim = len(label_encoder.classes_)
print("Output dimension:", output_dim)
num_layers = 2
bidirectional = True
dropout = 0.1
model = LSTMClassifier(input_dim, hidden_dim, output_dim, num_layers, bidirectional, dropout).to(device)
#### model, optimizer, data, criteria to cuda

# Hyperparameters and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Training loop with evaluation and saving the best model
best_val_loss = float('inf')

# Initialize lists to store losses, F1 scores, and accuracies for each epoch
epochs = 50  # Define the number of epochs
train_losses = []
val_losses = []
train_f1s = []
val_f1s = []
train_accuracies = []
val_accuracies = []
best_model = None
best_label_encoder = None

for epoch in range(epochs):
    train_loss, train_f1, train_accuracy = train(model, train_dataloader, optimizer, criterion, device)
    val_loss, val_f1, val_accuracy, predictions, labels = evaluate(model, test_dataloader, criterion, device)
    
    # Append metrics for this epoch
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_f1s.append(train_f1)
    val_f1s.append(val_f1)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)
    
    # Convert classes to strings
    str_classes = [str(cls) for cls in label_encoder.classes_]
    
    # Print results of evaluation
    print(f'Epoch {epoch} Train loss: {train_loss:.4f} Val loss: {val_loss:.4f}')
    print(f'Train Accuracy: {train_accuracy*100:.2f}% Val Accuracy: {val_accuracy*100:.2f}%')
    print(f'Train F1 Score: {train_f1*100:.2f}% Val F1 Score: {val_f1*100:.2f}%')

    # Compute confusion matrix and classification report
    conf_matrix = confusion_matrix(labels, predictions)
    report = classification_report(labels, predictions, target_names=str_classes)

    # Print confusion matrix and classification report
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(report)

    # Save model and label encoder if there is an improvement
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
        best_label_encoder = label_encoder

# Create a new directory for this run
output_dir = f'{type_of_data}_{"_".join(map(str, type_of_label))}'
os.makedirs(output_dir, exist_ok=True)

# Save the best model, label encoder, and metrics after all epochs are complete
save_model_and_artifacts(best_model, best_label_encoder, os.path.join(output_dir, 'best_model.pth'), os.path.join(output_dir, 'label_encoder.pkl'))
print('Saved best model and label encoder.')

# Save classification report
report_txt_path = os.path.join(output_dir, 'classification_report.txt')
with open(report_txt_path, 'w') as f:
    f.write(report)
print(f'Saved classification report to {report_txt_path}.')

# Save confusion matrix
conf_matrix_txt_path = os.path.join(output_dir, 'confusion_matrix.txt')
with open(conf_matrix_txt_path, 'w') as f:
    for row in conf_matrix:
        f.write(' '.join(map(str, row)) + '\n')
print(f'Saved confusion matrix to {conf_matrix_txt_path}.')

# Plot losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(output_dir, 'loss_plot.png'))  # Save plot to file

# Plot F1 scores and accuracies
plt.figure(figsize=(10, 5))
plt.plot(train_f1s, label='Train F1')
plt.plot(val_f1s, label='Validation F1')
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Metric')
plt.legend()
plt.savefig(os.path.join(output_dir, 'metric_plot.png'))  # Save plot to file

# Save losses, F1 scores, and accuracies to a CSV file
metrics_df = pd.DataFrame({
    'epoch': list(range(epochs)),
    'train_loss': train_losses,
    'val_loss': val_losses,
    'train_f1': train_f1s,
    'val_f1': val_f1s,
    'train_accuracy': train_accuracies,
    'val_accuracy': val_accuracies
})
metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
print('Saved metrics.')