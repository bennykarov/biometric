import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold

import utils
# ----------------------------
# 1. Scanpath Encoder (LSTM)
# ----------------------------
class ScanpathEncoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths):
        # Pack for variable-length sequences
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h, c) = self.lstm(packed)
        # h: [1, batch, hidden_dim] (unidirectional)
        h_out = h[0]  # Take the single direction
        out = self.fc(h_out)
        return out  # [batch, output_dim]

# ----------------------------
# 2. Pairwise Comparison Model
# ----------------------------
class ScanpathPairModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, emb_dim=64):
        super().__init__()
        self.encoder = ScanpathEncoder(input_dim, hidden_dim, emb_dim)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, sp1, len1, sp2, len2):
        e1 = self.encoder(sp1, len1)
        e2 = self.encoder(sp2, len2)
        diff = e2 - e1
        out = self.classifier(diff)
        return torch.sigmoid(out).squeeze(1)

def calculate_accuracy(preds, labels, threshold=0.5):
    """
    Calculate binary classification accuracy
    """
    predictions = (preds > threshold).float()
    correct = (predictions == labels).float().sum()
    accuracy = correct / len(labels)
    return accuracy.item()

# ----------------------------
# 4. Example Training Loop
# ----------------------------
def collate_fn(batch):
    # batch = [(sp1, sp2, label), ...]
    sp1s, sp2s, labels = zip(*batch)

    # Check for empty scanpaths and provide informative error
    for i, (sp1, sp2) in enumerate(zip(sp1s, sp2s)):

        if len(sp1) == 0:
            raise ValueError(f"Empty scanpath found at batch index {i} for sp1")
        if len(sp2) == 0:
            raise ValueError(f"Empty scanpath found at batch index {i} for sp2")

        '''
        if len(sp1) == 0 and len(sp2) == 0:
            raise ValueError(f"Empty scanpath found at batch index {i} for both sp1 and sp2")
        '''

    len1 = torch.tensor([len(s) for s in sp1s])
    len2 = torch.tensor([len(s) for s in sp2s])

    sp1_padded = pad_sequence([torch.tensor(s, dtype=torch.float32) for s in sp1s],
                              batch_first=True)
    sp2_padded = pad_sequence([torch.tensor(s, dtype=torch.float32) for s in sp2s],
                              batch_first=True)
    labels = torch.tensor(labels, dtype=torch.float32)
    return sp1_padded, len1, sp2_padded, len2, labels

def train_fold(model, train_data, val_data, num_epochs, patience, results_dir, fold_num):
    """
    Train a single fold with the given train/val split
    """
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = nn.BCELoss()

    # Create data loaders
    train_loader = [collate_fn(train_data)]
    val_loader = [collate_fn(val_data)] if len(val_data) > 0 else None

    best_loss = float('inf')
    patience_counter = 0

    fold_stats = {
        'epochs': [],
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        num_train_batches = 0

        for sp1, len1, sp2, len2, labels in train_loader:
            optimizer.zero_grad()
            preds = model(sp1, len1, sp2, len2)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            accuracy = calculate_accuracy(preds, labels)
            train_loss += loss.item()
            train_accuracy += accuracy
            num_train_batches += 1

        avg_train_loss = train_loss / num_train_batches
        avg_train_accuracy = train_accuracy / num_train_batches

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        num_val_batches = 0

        if val_loader is not None:
            with torch.no_grad():
                for sp1, len1, sp2, len2, labels in val_loader:
                    preds = model(sp1, len1, sp2, len2)
                    loss = criterion(preds, labels)

                    accuracy = calculate_accuracy(preds, labels)
                    val_loss += loss.item()
                    val_accuracy += accuracy
                    num_val_batches += 1

            avg_val_loss = val_loss / num_val_batches
            avg_val_accuracy = val_accuracy / num_val_batches
        else:
            avg_val_loss = 0.0
            avg_val_accuracy = 0.0

        fold_stats['epochs'].append(epoch + 1)
        fold_stats['train_loss'].append(avg_train_loss)
        fold_stats['train_accuracy'].append(avg_train_accuracy)
        fold_stats['val_loss'].append(avg_val_loss)
        fold_stats['val_accuracy'].append(avg_val_accuracy)

        # Save best model
        if val_loader is not None and avg_val_loss < best_loss:
            best_loss = avg_val_loss
            checkpoints_dir = os.path.join(results_dir, f'fold_{fold_num}', 'checkpoints')
            os.makedirs(checkpoints_dir, exist_ok=True)
            utils.save_model(model, optimizer, epoch + 1, avg_val_loss,
                    save_dir=checkpoints_dir, filename='best_model.pth')
            patience_counter = 0
        elif val_loader is not None:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    return fold_stats, best_loss

def cross_validate(sample_data, n_folds=5, num_epochs=50, patience=10, results_dir='results'):
    """
    Perform k-fold cross-validation
    """
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    cv_results = {
        'fold_stats': [],
        'fold_best_losses': [],
        'fold_final_accuracies': []
    }

    print(f"\n{'='*60}")
    print(f"Starting {n_folds}-Fold Cross-Validation")
    print(f"Total samples: {len(sample_data)}")
    print(f"{'='*60}\n")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(sample_data), 1):
        print(f"\n{'='*60}")
        print(f"FOLD {fold}/{n_folds}")
        print(f"{'='*60}")

        # Split data
        train_data = [sample_data[i] for i in train_idx]
        val_data = [sample_data[i] for i in val_idx]

        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}\n")

        # Create new model for this fold
        model = ScanpathPairModel(input_dim=4, hidden_dim=16, emb_dim=16)

        # Train fold
        fold_stats, best_loss = train_fold(
            model, train_data, val_data, num_epochs, patience,
            results_dir, fold
        )

        # Store results
        cv_results['fold_stats'].append(fold_stats)
        cv_results['fold_best_losses'].append(best_loss)
        cv_results['fold_final_accuracies'].append(fold_stats['val_accuracy'][-1])

        print(f"\nFold {fold} Results:")
        print(f"  Best Val Loss: {best_loss:.4f}")
        print(f"  Final Val Accuracy: {fold_stats['val_accuracy'][-1]:.4f}")

    # Calculate aggregate statistics
    cv_results['mean_best_loss'] = np.mean(cv_results['fold_best_losses'])
    cv_results['std_best_loss'] = np.std(cv_results['fold_best_losses'])
    cv_results['mean_final_accuracy'] = np.mean(cv_results['fold_final_accuracies'])
    cv_results['std_final_accuracy'] = np.std(cv_results['fold_final_accuracies'])

    print(f"\n{'='*60}")
    print(f"Cross-Validation Summary")
    print(f"{'='*60}")
    print(f"Mean Best Val Loss: {cv_results['mean_best_loss']:.4f} ± {cv_results['std_best_loss']:.4f}")
    print(f"Mean Final Val Accuracy: {cv_results['mean_final_accuracy']:.4f} ± {cv_results['std_final_accuracy']:.4f}")
    print(f"{'='*60}\n")

    return cv_results

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train LSTM model with optional cross-validation')
    parser.add_argument('files', nargs='*', help='Scanpath data files')
    parser.add_argument('--cv', action='store_true', help='Enable k-fold cross-validation')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of folds for cross-validation (default: 5)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (default: 50)')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (default: 10)')

    args = parser.parse_args()
    file_names = args.files

    #sample_data = utils.createSyntheticData()
    if len(file_names) == 0:
        sample_data = utils.readScanpathFile()
        utils.writeScanpathTXTFile(sample_data)
    else:
        sample_data = []
        for file_name in file_names:
            print(f"Loading scanpath data from: {file_name}")
            sample_data += utils.readScanpathFile(file_name)

    # Shuffle data for reproducibility
    import random
    random.seed(42)
    random.shuffle(sample_data)

    print(f"Total samples: {len(sample_data)}")

    # Create timestamped result directories
    timestamp = datetime.now().strftime('%m%d_%H%M%S')
    results_dir = f'results/result_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)

    # Choose between cross-validation and regular train/val split
    if args.cv:
        print(f"\n{'='*60}")
        print(f"Running {args.n_folds}-Fold Cross-Validation")
        print(f"{'='*60}\n")

        # Perform cross-validation
        cv_results = cross_validate(
            sample_data,
            n_folds=args.n_folds,
            num_epochs=args.epochs,
            patience=args.patience,
            results_dir=results_dir
        )

        # Save cross-validation results
        cv_summary = {
            'n_folds': args.n_folds,
            'num_epochs': args.epochs,
            'patience': args.patience,
            'total_samples': len(sample_data),
            'mean_best_loss': cv_results['mean_best_loss'],
            'std_best_loss': cv_results['std_best_loss'],
            'mean_final_accuracy': cv_results['mean_final_accuracy'],
            'std_final_accuracy': cv_results['std_final_accuracy'],
            'fold_best_losses': cv_results['fold_best_losses'],
            'fold_final_accuracies': cv_results['fold_final_accuracies'],
            'timestamp': timestamp
        }

        # Save CV summary
        with open(os.path.join(results_dir, 'cv_summary.json'), 'w') as f:
            json.dump(cv_summary, f, indent=4)

        # Plot cross-validation results
        utils.plot_cv_results(cv_results, save_dir=results_dir)

        print(f"\nCross-validation completed!")
        print(f"Results saved to: {results_dir}")
        return

    # Regular train/val split
    print(f"\n{'='*60}")
    print(f"Running Regular Training with Train/Val Split")
    print(f"{'='*60}\n")

    train_size = int(0.8 * len(sample_data))
    train_data = sample_data[:train_size]
    val_data = sample_data[train_size:]

    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    # Create data loaders
    train_loader = [collate_fn(train_data)]
    val_loader = [collate_fn(val_data)] if len(val_data) > 0 else None

    # Model configuration optimized for ~200 samples
    # Smaller model to prevent overfitting on small dataset
    model = ScanpathPairModel(input_dim=4, hidden_dim=16, emb_dim=16)

    # Optimizer with weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = nn.BCELoss()

    # Training configuration
    num_epochs = args.epochs
    best_loss = float('inf')
    patience = args.patience
    patience_counter = 0

    # Create checkpoints directory
    checkpoints_dir = os.path.join(results_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    print(f"Results will be saved to: {results_dir}")
    print(f"Training for {num_epochs} epochs with patience {patience}\n")

    # Statistics tracking
    training_stats = {
        'model_config': {
            'input_dim': 4,
            'hidden_dim': 16,
            'emb_dim': 16,
            'learning_rate': 5e-4,
            'weight_decay': 1e-4,
            'num_samples': len(sample_data),
            'train_samples': len(train_data),
            'val_samples': len(val_data)
        },
        'training_history': {
            'epochs': [],
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        },
        'best_epoch': 0,
        'best_val_loss': float('inf'),
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'results_dir': results_dir
    }

    # Training loop with statistics
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        num_train_batches = 0

        for sp1, len1, sp2, len2, labels in train_loader:
            optimizer.zero_grad()
            preds = model(sp1, len1, sp2, len2)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            # Calculate metrics
            accuracy = calculate_accuracy(preds, labels)
            train_loss += loss.item()
            train_accuracy += accuracy
            num_train_batches += 1

        # Average training metrics
        avg_train_loss = train_loss / num_train_batches
        avg_train_accuracy = train_accuracy / num_train_batches

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        num_val_batches = 0

        if val_loader is not None:
            with torch.no_grad():
                for sp1, len1, sp2, len2, labels in val_loader:
                    preds = model(sp1, len1, sp2, len2)
                    loss = criterion(preds, labels)

                    # Calculate metrics
                    accuracy = calculate_accuracy(preds, labels)
                    val_loss += loss.item()
                    val_accuracy += accuracy
                    num_val_batches += 1

            avg_val_loss = val_loss / num_val_batches
            avg_val_accuracy = val_accuracy / num_val_batches
        else:
            avg_val_loss = 0.0
            avg_val_accuracy = 0.0

        # Update statistics
        training_stats['training_history']['epochs'].append(epoch + 1)
        training_stats['training_history']['train_loss'].append(avg_train_loss)
        training_stats['training_history']['train_accuracy'].append(avg_train_accuracy)
        training_stats['training_history']['val_loss'].append(avg_val_loss)
        training_stats['training_history']['val_accuracy'].append(avg_val_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss={avg_train_loss:.4f}, Train Acc={avg_train_accuracy:.4f} | "
              f"Val Loss={avg_val_loss:.4f}, Val Acc={avg_val_accuracy:.4f}")

        # Save best model based on validation loss with early stopping
        if val_loader is not None and avg_val_loss < best_loss:
            best_loss = avg_val_loss
            training_stats['best_epoch'] = epoch + 1
            training_stats['best_val_loss'] = best_loss
            utils.save_model(model, optimizer, epoch + 1, avg_val_loss,
                    save_dir=checkpoints_dir, filename='best_model.pth')
            patience_counter = 0  # Reset patience counter
            print(f"  --> New best model saved! (Val Loss: {best_loss:.4f})")
        elif val_loader is not None:
            patience_counter += 1
            print(f"  --> No improvement ({patience_counter}/{patience})")

        elif val_loader is None and avg_train_loss < best_loss:
            # Fallback to train loss if no validation set
            best_loss = avg_train_loss
            training_stats['best_epoch'] = epoch + 1
            training_stats['best_val_loss'] = best_loss
            utils.save_model(model, optimizer, epoch + 1, avg_train_loss,
                    save_dir=checkpoints_dir, filename='best_model.pth')
            patience_counter = 0

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs (patience={patience})")
            print(f"Best validation loss: {best_loss:.4f} at epoch {training_stats['best_epoch']}")
            break

    # Final statistics
    training_stats['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    training_stats['total_epochs'] = num_epochs

    # Save training statistics
    utils.save_statistics(len(sample_data), training_stats, save_dir=results_dir, filename='training_stats.json')

    # Plot training curves
    utils.plot_training_curves(training_stats, save_dir=results_dir)

    # Also save final model
    utils.save_model(model, optimizer, num_epochs, training_stats['training_history']['train_loss'][-1],
            save_dir=checkpoints_dir, filename='final_model.pth')

    print("\nTraining completed!")
    print(f"Best model at epoch {training_stats['best_epoch']} with val loss {training_stats['best_val_loss']:.4f}")
    print(f"Final train accuracy: {training_stats['training_history']['train_accuracy'][-1]:.4f}")
    if val_loader is not None:
        print(f"Final val accuracy: {training_stats['training_history']['val_accuracy'][-1]:.4f}")
    print(f"All results saved to: {results_dir}")



if __name__ == '__main__':
    main()
