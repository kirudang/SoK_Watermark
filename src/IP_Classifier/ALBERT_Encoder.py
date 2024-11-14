import os
# Ensure the HF_HOME environment variable points to your desired cache location
os.environ["HF_TOKEN"] = "Your_Hugging_Face_API_Token"  # Change this to your Hugging Face API token
cache_dir = 'YOUR_CACHE_DIR'  # Change this to your desired cache directory
os.environ['HF_HOME'] = cache_dir

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AlbertForSequenceClassification, TrainingArguments, Trainer, TrainerCallback, AlbertTokenizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# All Parameters

lrate = 1e-5
epoches_num = 10
batch_size = 64
# Define the dataset
type_of_data =  "input_output"  #"output_only" #
# 2: llama2,                     3: mistral,               5: OPT 7B,               7: Falcon 7B                20: Bloom       6: Qwen          9: Vicuna     10: Gemma
# 4: ICML watermarked llama2,   21: ICML wm Mistral,      30: ICML wm OPT 7B,       22: ICML wm Falcon 7B
# 15: Distortion wm llama 2     19: Distortion wm Mistral 31: Distortion wm OPT 67  32: Distortion wm Falcon
# 33: SIR llama 2               34: SIR Mistral           35: SIR OPT 67            36: SIR Falcon 7B
# 37: Semstamp llama 2          38: Semstamp Mistral      39: Semstamp OPT 67       40: Semstamp Falcon 7B
# 41: Adaptive llama 2          42: Adaptive Mistral      43: Adaptive OPT          44: Adaptive Falcon
# 45: Uni llama 2               46: Uni Mistral           47: Uni OPT               48: Uni Falcon
# 49: Unbias llama 2            50: Unbias Mistral        51: Unbias OPT            52: Unbias Falcon
# 53: UPV llama 2               54: UPV Mistral           55: UPV OPT               56: UPV Falcon

type_of_label = [53,54,55,56]  # or any other labels you want to filter

base_model = "albert/albert-base-v2"

# Clear cache
torch.cuda.empty_cache()

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



# Load and prepare the data
data_file = ' Your_Data_File.json'
train_texts, train_labels, test_texts, test_labels = load_data(data_file, type_of_data, type_of_label)

unique, counts = np.unique(train_labels, return_counts=True)
print("Train labels distribution:", dict(zip(unique, counts)))
unique, counts = np.unique(test_labels, return_counts=True)
print("Test labels distribution:", dict(zip(unique, counts)))

label_mapping = {label: idx for idx, label in enumerate(type_of_label)}
def encode_labels(labels, mapping):
    return [mapping[label] for label in labels]
train_labels_encoded = encode_labels(train_labels, label_mapping)
test_labels_encoded = encode_labels(test_labels, label_mapping)

tokenizer = AlbertTokenizer.from_pretrained(base_model)
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=400, return_tensors="pt")
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=400, return_tensors="pt")

class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = CustomDataset(train_encodings, train_labels_encoded)
test_dataset = CustomDataset(test_encodings, test_labels_encoded)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

output_dim = len(label_mapping)
print("Output dimension:", output_dim)

model = AlbertForSequenceClassification.from_pretrained(base_model, num_labels=output_dim).to(device)

output_dir = f'{type_of_data}_{"_".join(map(str, type_of_label))}'
os.makedirs(output_dir, exist_ok=True)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average='weighted')
    accuracy = accuracy_score(labels, preds)
    return {'Accuracy': accuracy, 'F1': f1}

# Store loss for visualization
class LoggingCallback(TrainerCallback):                
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            output_log_file = os.path.join(args.output_dir, "train_results.json")
            with open(output_log_file, "a") as writer:
                writer.write(json.dumps(logs) + "\n")

loss_logger = LoggingCallback()

training_args = TrainingArguments(
    output_dir=output_dir,
    gradient_accumulation_steps=2,
    do_train=True,
    do_eval=True,
    learning_rate=lrate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epoches_num,
    weight_decay=0.01,
    evaluation_strategy="steps",
    logging_strategy="steps",
    logging_steps=250,
    save_steps=-1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[loss_logger]
)

trainer.train()

model_path = os.path.join(output_dir, 'model')
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)


# Create a reverse mapping from the encoded labels back to the original labels
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

def decode_labels(encoded_labels, mapping):
    return [mapping[label] for label in encoded_labels]

# Evaluate using the trainer
print("Starting evaluation...")
eval_results = trainer.predict(test_dataset)
predictions = eval_results.predictions
true_labels = eval_results.label_ids
eval_metrics = compute_metrics(eval_results)
print("Evaluation completed.")

# Decode the predictions and true labels
decoded_predictions = decode_labels(predictions.argmax(-1), reverse_label_mapping)
decoded_true_labels = decode_labels(true_labels, reverse_label_mapping)

# Save evaluation results
metrics_df = pd.DataFrame(eval_metrics.items(), columns=['Metric', 'Value'])
metrics_csv_path = os.path.join(output_dir, 'test_metrics.csv')
metrics_df.to_csv(metrics_csv_path, index=False)
print(f'Saved metrics to {metrics_csv_path}.')

# Generate confusion matrix and classification report using decoded labels
conf_matrix = confusion_matrix(decoded_true_labels, decoded_predictions)
report = classification_report(decoded_true_labels, decoded_predictions)

print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(report)

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


# Plot the train and test loss
train_results = os.path.join(output_dir, 'train_results.json')
# Initialize lists to store the metrics
epochs = []
train_losses = []
eval_losses = []

# Read the JSON file and parse each line
with open(train_results, "r") as f:
    for line in f:
        data = json.loads(line)
        if 'epoch' in data:
            epoch = data['epoch']
            if 'loss' in data:
                train_losses.append(data['loss'])
                if epoch not in epochs:
                    epochs.append(epoch)  # Append epoch only when train loss is present
            if 'eval_loss' in data:
                eval_losses.append(data['eval_loss'])
                if epoch not in epochs:
                    epochs.append(epoch)  # Append epoch only when eval loss is present

# Ensure the lengths of epochs, train_losses, and eval_losses are consistent
min_length = min(len(epochs), len(train_losses), len(eval_losses))
epochs = epochs[:min_length]
train_losses = train_losses[:min_length]
eval_losses = eval_losses[:min_length]

# Plotting
plt.figure(figsize=(10, 5))

plt.plot(epochs, train_losses, label='Train Loss', color='blue')
plt.plot(epochs, eval_losses, label='Eval Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Training and Evaluation Loss')
plt.legend()

plt.tight_layout()

# Save the plot in the output directory
plot_path = os.path.join(output_dir, 'training_evaluation_loss_plot.png')
plt.savefig(plot_path)
plt.close()

print(f"Plot saved in the current directory as 'training_evaluation_loss_plot.png'.")