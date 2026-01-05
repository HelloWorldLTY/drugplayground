import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
from torch.utils.data import TensorDataset, DataLoader
import os

# File paths configuration
ESM_PKL_PATH = "/data/zf/data/LTY-Proj/esm_emb/data_Drugbank.pkl"  # Key: protein
UNIMOL_PKL_PATH = "/data/zf/data/LTY-Proj/esm_unimol/unimol_data_Drugbank.pkl"  # Key: smile
CSV_PATH = "/data/zf/data/LTY-Proj/label/Drugbank.csv"

# 1. Load CSV file
csv_data = pd.read_csv(CSV_PATH)
print(f"CSV contains {len(csv_data)} samples")

# 2. Load PKL files
def load_pkl_dict(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

# Load embeddings dictionaries
print("Loading ESM embeddings...")
esm_embeddings = load_pkl_dict(ESM_PKL_PATH)  # {protein: embedding}
print("Loading UniMol embeddings...")
unimol_embeddings = load_pkl_dict(UNIMOL_PKL_PATH)  # {smile: embedding}

# Convert keys to string for consistent comparison
esm_embeddings = {str(k).strip(): v for k, v in esm_embeddings.items()}
unimol_embeddings = {str(k).strip(): v for k, v in unimol_embeddings.items()}

# 3. Process CSV rows and build dataset
X = []  # Combined embeddings
y = []  # Labels
valid_samples = []  # For tracking valid sample details
skipped_samples = []  # For tracking skipped samples

for i, row in csv_data.iterrows():
    # Extract values and clean
    compound_smiles = str(row['smile']).strip()
    target_seq = str(row['protein']).strip()
    affinity = row['affinity']
    
    # Check if embeddings exist for both
    smiles_emb_exists = compound_smiles in unimol_embeddings
    seq_emb_exists = target_seq in esm_embeddings
    
    if not (smiles_emb_exists and seq_emb_exists):
        skipped_samples.append({
            'index': i,
            'compound_smiles': compound_smiles,
            'target_seq': target_seq,
            'smiles_found': smiles_emb_exists,
            'seq_found': seq_emb_exists
        })
        continue
    
    # Extract embeddings
    def process_embedding(emb):
        if isinstance(emb, np.ndarray):
            return emb.flatten()
        elif torch.is_tensor(emb):
            return emb.cpu().numpy().flatten()
        else:
            return np.array(emb).flatten()
    
    unimol_emb = process_embedding(unimol_embeddings[compound_smiles])
    esm_emb = process_embedding(esm_embeddings[target_seq])
    
    # Combine embeddings
    combined_emb = np.concatenate([unimol_emb, esm_emb])
    X.append(combined_emb)
    y.append(affinity)
    
    # Record valid sample details
    valid_samples.append({
        'index': i,
        'compound_smiles': compound_smiles,
        'target_seq': target_seq,
        'affinity': affinity,
        'unimol_emb_shape': unimol_emb.shape,
        'esm_emb_shape': esm_emb.shape
    })

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Print dataset statistics
print(f"\nDataset statistics:")
print(f"  Total samples processed: {len(csv_data)}")
print(f"  Valid samples: {len(X)}")
print(f"  Skipped samples: {len(skipped_samples)}")
print(f"  Positive ratio: {np.mean(y == 1):.4f}")

if len(X) == 0:
    raise ValueError("No valid samples found! Check your data matching")

# Print reasons for skipped samples
if skipped_samples:
    print("\nSkipped sample reasons:")
    missing_smiles = sum(1 for s in skipped_samples if not s['smiles_found'])
    missing_seq = sum(1 for s in skipped_samples if not s['seq_found'])
    print(f"  Missing compound_smiles: {missing_smiles}")
    print(f"  Missing protein: {missing_seq}")
    print(f"  Both missing: {len(skipped_samples) - missing_smiles - missing_seq}")

# Check embedding shapes
if valid_samples:
    print("\nEmbedding dimension report:")
    print(f"  UniMol embedding shape: {valid_samples[0]['unimol_emb_shape']}")
    print(f"  ESM embedding shape: {valid_samples[0]['esm_emb_shape']}")
    feature_dim = len(X[0])
    print(f"  Combined feature dimension: {feature_dim}")

# Define MLP model architecture
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return self.softmax(x)

# Training parameters
BATCH_SIZE = 32
EPOCHS = 50
NUM_FOLDS = 5
RESULTS_FILE = "Drugbank_5_fold_results.txt"

# Setup GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")
if not torch.cuda.is_available():
    print("Warning: CUDA not available, using CPU instead")

# Initialize result file
with open(RESULTS_FILE, "w") as f:
    f.write("5-Fold Cross Validation Results\n")
    f.write("=" * 50 + "\n")
    f.write(f"Original CSV samples: {len(csv_data)}\n")
    f.write(f"Valid samples: {len(X)}\n")
    f.write(f"Skipped samples: {len(skipped_samples)}\n")
    f.write(f"Feature dimension: {feature_dim}\n")
    f.write(f"Positive ratio: {np.mean(y == 1):.4f}\n")
    f.write(f"Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}\n")
    f.write("=" * 50 + "\n\n")
    
    # Write matching logic details
    f.write("Data Matching Logic:\n")
    f.write("- For each CSV row:\n")
    f.write("  1. Use smile to find embedding in unimol_data_Drugbank.pkl\n")
    f.write("  2. Use protein to find embedding in esm_emb/data_Drugbank.pkl\n")
    f.write("  3. Only include samples where both embeddings exist\n\n")
    
    # Write skipped sample stats
    if skipped_samples:
        f.write("Skipped Sample Reasons:\n")
        missing_smiles = sum(1 for s in skipped_samples if not s['smiles_found'])
        missing_seq = sum(1 for s in skipped_samples if not s['seq_found'])
        f.write(f"  Missing compound_smiles: {missing_smiles}\n")
        f.write(f"  Missing protein: {missing_seq}\n")
        f.write(f"  Both missing: {len(skipped_samples) - missing_smiles - missing_seq}\n")
    f.write("=" * 50 + "\n\n")

# Set up 5-fold stratified cross-validation
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
fold_results = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n{'='*20} Fold {fold_idx+1}/{NUM_FOLDS} {'='*20}")
    
    # Split data
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Convert to PyTorch tensors and create datasets
    train_data = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), 
        torch.tensor(y_train, dtype=torch.long)
    )
    val_data = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32), 
        torch.tensor(y_val, dtype=torch.long)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model and optimizer
    model = MLP(input_dim=feature_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training variables
    best_val_acc = 0.0
    best_metrics = {}
    
    # Training loop
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_preds = []
        val_probs = []
        val_true = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                
                # Get predictions and probabilities
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_probs.extend(outputs[:, 1].cpu().numpy())  # Positive class probabilities
                val_true.extend(batch_y.cpu().numpy())
        
        # Calculate metrics
        val_acc = accuracy_score(val_true, val_preds)
        val_auc = roc_auc_score(val_true, val_probs)
        precision, recall, f1, _ = precision_recall_fscore_support(val_true, val_preds, average='binary')
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{EPOCHS}: "
              f"Loss: {total_loss/len(train_loader):.4f}, "
              f"Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}, "
              f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # Save best model in current fold
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_metrics = {
                'acc': val_acc,
                'auc': val_auc,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
    
    # Save fold results
    fold_result = {
        'fold': fold_idx + 1,
        **best_metrics
    }
    fold_results.append(fold_result)
    
    # Write fold results to file
    with open(RESULTS_FILE, "a") as f:
        f.write(f"Fold {fold_idx+1} Results:\n")
        f.write(f"  Best Accuracy: {best_metrics['acc']:.4f}\n")
        f.write(f"  Best AUC: {best_metrics['auc']:.4f}\n")
        f.write(f"  Best Precision: {best_metrics['precision']:.4f}\n")
        f.write(f"  Best Recall: {best_metrics['recall']:.4f}\n")
        f.write(f"  Best F1 Score: {best_metrics['f1']:.4f}\n")
        f.write("-" * 50 + "\n")

# Calculate average metrics across folds
avg_metrics = {
    'acc': np.mean([res['acc'] for res in fold_results]),
    'auc': np.mean([res['auc'] for res in fold_results]),
    'precision': np.mean([res['precision'] for res in fold_results]),
    'recall': np.mean([res['recall'] for res in fold_results]),
    'f1': np.mean([res['f1'] for res in fold_results])
}

# Calculate standard deviations
std_metrics = {
    'acc': np.std([res['acc'] for res in fold_results]),
    'auc': np.std([res['auc'] for res in fold_results]),
    'precision': np.std([res['precision'] for res in fold_results]),
    'recall': np.std([res['recall'] for res in fold_results]),
    'f1': np.std([res['f1'] for res in fold_results])
}

# Write summary to file
with open(RESULTS_FILE, "a") as f:
    f.write("\nCross-Validation Summary:\n")
    f.write("=" * 50 + "\n")
    f.write(f"Average Accuracy: {avg_metrics['acc']:.4f} ± {std_metrics['acc']:.4f}\n")
    f.write(f"Average AUC: {avg_metrics['auc']:.4f} ± {std_metrics['auc']:.4f}\n")
    f.write(f"Average Precision: {avg_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}\n")
    f.write(f"Average Recall: {avg_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}\n")
    f.write(f"Average F1 Score: {avg_metrics['f1']:.4f} ± {std_metrics['f1']:.4f}\n")
    f.write("=" * 50 + "\n\n")
    
    # Add dataset statistics
    f.write("Final Dataset Statistics:\n")
    f.write("=" * 50 + "\n")
    f.write(f"Original CSV samples: {len(csv_data)}\n")
    f.write(f"Valid samples with complete embeddings: {len(X)}\n")
    f.write(f"UniMol embeddings available: {len(unimol_embeddings)}\n")
    f.write(f"ESM embeddings available: {len(esm_embeddings)}\n")
    f.write(f"Feature dimension: {feature_dim}\n")
    f.write(f"Positive ratio: {np.mean(y == 1):.4f}\n")
    f.write("=" * 50 + "\n\n")

print("\nTraining completed! Results saved to:", RESULTS_FILE)
print(f"Final dataset size: {len(X)} samples")
print(f"Average Accuracy: {avg_metrics['acc']:.4f}, Average AUC: {avg_metrics['auc']:.4f}")

# Save valid sample details for reference
valid_samples_df = pd.DataFrame(valid_samples)
valid_samples_df.to_csv("valid_samples_details.csv", index=False)
print("Saved valid sample details to 'valid_samples_details.csv'")

if skipped_samples:
    skipped_samples_df = pd.DataFrame(skipped_samples)
    skipped_samples_df.to_csv("skipped_samples_details.csv", index=False)
    print("Saved skipped sample details to 'skipped_samples_details.csv'")