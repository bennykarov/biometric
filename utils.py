from sklearn import utils
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import pickle
import numpy as np


def _print(msg):
    print(f"[utils] {msg}")



# ----------------------------
# 3. Model Saving and Statistics
# ----------------------------
def save_model(model, optimizer, epoch, loss, save_dir='checkpoints', filename='best_model.pth'):
    """
    Save model checkpoint with training state
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    save_path = os.path.join(save_dir, filename)
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")
    return save_path

def load_model(model, optimizer, checkpoint_path):
    """
    Load model checkpoint
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Model loaded from {checkpoint_path} (Epoch: {epoch}, Loss: {loss:.4f})")
    return model, optimizer, epoch, loss

def save_statistics(sampleLen, stats, save_dir='results', filename='training_stats.json'):
    """
    Save training statistics to JSON file
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    
    # Add sample length info
    stats['num_samples'] = sampleLen

    with open(save_path, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"Statistics saved to {save_path}")
    return save_path


def plot_training_curves(stats, save_dir='results'):
    """
    Plot and save training loss and accuracy curves with train/val split
    """
    os.makedirs(save_dir, exist_ok=True)

    epochs = stats['training_history']['epochs']

    # Check if we have new format (train/val split) or old format
    has_val_data = 'train_loss' in stats['training_history']

    if has_val_data:
        train_loss = stats['training_history']['train_loss']
        train_accuracy = stats['training_history']['train_accuracy']
        val_loss = stats['training_history']['val_loss']
        val_accuracy = stats['training_history']['val_accuracy']
    else:
        # Fallback to old format for backward compatibility
        train_loss = stats['training_history'].get('loss', [])
        train_accuracy = stats['training_history'].get('accuracy', [])
        val_loss = []
        val_accuracy = []

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot loss
    ax1.plot(epochs, train_loss, 'b-o', linewidth=2, markersize=6, label='Train')
    if has_val_data and any(v > 0 for v in val_loss):
        ax1.plot(epochs, val_loss, 'r-o', linewidth=2, markersize=6, label='Validation')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Mark best epoch on loss plot
    if 'best_epoch' in stats:
        best_epoch = stats['best_epoch']
        best_val_loss = stats.get('best_val_loss', stats.get('best_loss', 0))
        if best_val_loss > 0:
            ax1.plot(best_epoch, best_val_loss, 'g*', markersize=15,
                    label=f'Best Val (Epoch {best_epoch})')

    # Plot accuracy
    ax2.plot(epochs, train_accuracy, 'b-o', linewidth=2, markersize=6, label='Train')
    if has_val_data and any(v > 0 for v in val_accuracy):
        ax2.plot(epochs, val_accuracy, 'r-o', linewidth=2, markersize=6, label='Validation')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    # Save figure
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close()

    return save_path

def plot_cv_results(cv_results, save_dir='results'):
    """
    Plot cross-validation results across all folds
    """
    os.makedirs(save_dir, exist_ok=True)

    n_folds = len(cv_results['fold_stats'])

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))

    # Plot 1: Loss curves for all folds
    ax1 = plt.subplot(2, 3, 1)
    for fold_idx, fold_stats in enumerate(cv_results['fold_stats']):
        epochs = fold_stats['epochs']
        val_loss = fold_stats['val_loss']
        ax1.plot(epochs, val_loss, '-o', linewidth=2, markersize=4,
                label=f'Fold {fold_idx+1}', alpha=0.7)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Validation Loss', fontsize=12)
    ax1.set_title('Validation Loss Across Folds', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Accuracy curves for all folds
    ax2 = plt.subplot(2, 3, 2)
    for fold_idx, fold_stats in enumerate(cv_results['fold_stats']):
        epochs = fold_stats['epochs']
        val_accuracy = fold_stats['val_accuracy']
        ax2.plot(epochs, val_accuracy, '-o', linewidth=2, markersize=4,
                label=f'Fold {fold_idx+1}', alpha=0.7)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Accuracy', fontsize=12)
    ax2.set_title('Validation Accuracy Across Folds', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Best loss per fold (bar chart)
    ax3 = plt.subplot(2, 3, 3)
    fold_numbers = list(range(1, n_folds + 1))
    best_losses = cv_results['fold_best_losses']
    ax3.bar(fold_numbers, best_losses, color='steelblue', alpha=0.7)
    ax3.axhline(y=cv_results['mean_best_loss'], color='r', linestyle='--',
                linewidth=2, label=f"Mean: {cv_results['mean_best_loss']:.4f}")
    ax3.set_xlabel('Fold', fontsize=12)
    ax3.set_ylabel('Best Validation Loss', fontsize=12)
    ax3.set_title('Best Loss per Fold', fontsize=14, fontweight='bold')
    ax3.set_xticks(fold_numbers)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend()

    # Plot 4: Final accuracy per fold (bar chart)
    ax4 = plt.subplot(2, 3, 4)
    final_accuracies = cv_results['fold_final_accuracies']
    ax4.bar(fold_numbers, final_accuracies, color='forestgreen', alpha=0.7)
    ax4.axhline(y=cv_results['mean_final_accuracy'], color='r', linestyle='--',
                linewidth=2, label=f"Mean: {cv_results['mean_final_accuracy']:.4f}")
    ax4.set_xlabel('Fold', fontsize=12)
    ax4.set_ylabel('Final Validation Accuracy', fontsize=12)
    ax4.set_title('Final Accuracy per Fold', fontsize=14, fontweight='bold')
    ax4.set_xticks(fold_numbers)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.legend()

    # Plot 5: Average training curves across folds
    ax5 = plt.subplot(2, 3, 5)
    # Calculate average train/val loss across folds
    max_epochs = max(len(fold['epochs']) for fold in cv_results['fold_stats'])
    avg_train_loss = []
    avg_val_loss = []

    for epoch in range(max_epochs):
        train_losses = [fold['train_loss'][epoch] for fold in cv_results['fold_stats']
                       if epoch < len(fold['epochs'])]
        val_losses = [fold['val_loss'][epoch] for fold in cv_results['fold_stats']
                     if epoch < len(fold['epochs'])]
        if train_losses:
            avg_train_loss.append(np.mean(train_losses))
        if val_losses:
            avg_val_loss.append(np.mean(val_losses))

    epochs_range = list(range(1, len(avg_train_loss) + 1))
    ax5.plot(epochs_range, avg_train_loss, 'b-o', linewidth=2, markersize=6, label='Avg Train')
    ax5.plot(epochs_range, avg_val_loss, 'r-o', linewidth=2, markersize=6, label='Avg Val')
    ax5.set_xlabel('Epoch', fontsize=12)
    ax5.set_ylabel('Loss', fontsize=12)
    ax5.set_title('Average Loss Across Folds', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    # Plot 6: Average accuracy curves across folds
    ax6 = plt.subplot(2, 3, 6)
    avg_train_acc = []
    avg_val_acc = []

    for epoch in range(max_epochs):
        train_accs = [fold['train_accuracy'][epoch] for fold in cv_results['fold_stats']
                     if epoch < len(fold['epochs'])]
        val_accs = [fold['val_accuracy'][epoch] for fold in cv_results['fold_stats']
                   if epoch < len(fold['epochs'])]
        if train_accs:
            avg_train_acc.append(np.mean(train_accs))
        if val_accs:
            avg_val_acc.append(np.mean(val_accs))

    ax6.plot(epochs_range, avg_train_acc, 'b-o', linewidth=2, markersize=6, label='Avg Train')
    ax6.plot(epochs_range, avg_val_acc, 'r-o', linewidth=2, markersize=6, label='Avg Val')
    ax6.set_xlabel('Epoch', fontsize=12)
    ax6.set_ylabel('Accuracy', fontsize=12)
    ax6.set_title('Average Accuracy Across Folds', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend()

    plt.tight_layout()

    # Save figure
    save_path = os.path.join(save_dir, 'cv_results.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Cross-validation results saved to {save_path}")
    plt.close()

    return save_path

'''
------------------------------------------------
    Data creation / reading utils
------------------------------------------------
'''
def readDummyData():
    return [
        # ([x,y,t,d], [x,y,t,d], label)
        ([[0,0,1,0.1],[1,1,2,0.2]], [[0,0,1,0.1],[1,1,2,0.3],[2,2,3,0.4]], 1),
        ([[0,0,1,0.2],[1,0,2,0.2]], [[0,0,1,0.1],[0.5,1,2,0.3]], 0)
    ]

# create synthetic RANDOM data for testing
def createSyntheticData(num_samples=100):
    data = []
    for _ in range(num_samples):
        len1 = np.random.randint(5, 15)
        len2 = np.random.randint(5, 15)
        sp1 = np.random.rand(len1, 4).tolist()  # [x,y,t,d]
        sp2 = np.random.rand(len2, 4).tolist()
        label = np.random.randint(0, 2)
        data.append((sp1, sp2, label))
    return data


def readScanpathFile(file_name=None):
    # open folder dialog:
    from tkinter import Tk
    from tkinter.filedialog import askopenfilenames, askdirectory

    if file_name is None:
        root = Tk()
        root.withdraw()  # Hide the root window
        # change to ask for A folder
        root.update()
        # Set default directory - you can change this path to your preferred default
        default_dir = "/media/bennyk/SSD_DATA2/DATA/emotionML/scanpath/"
        
        file_path = askopenfilenames(title="Select scanpath Pickle data files", 
                                 initialdir=default_dir,
                                 filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")])
        file_name = file_path[0]
        
        #folder_path = askdirectory(title="Select folder containing scanpath Pickle data files", initialdir=default_dir)
    
    if False: # DDEBUG DDEBUG
        convertPickleToJSON(file_name, file_name.replace(".pkl", ".json"))

    # displayScanpathData(file_name)

    with open(file_name, "rb") as f:
        data = pickle.load(f)

        # remove all empty lists from data
        data = [scanpath for scanpath in data if len(scanpath) > 0]

        
        scanpathPairs = [] 
        
        for scanpath_line in data:
            # convert file format [x,y,d,t...] to [x,y, t, d]
            scanpathL = np.array([[fixation[0], fixation[1], fixation[3], fixation[2]] for fixation in scanpath_line[0]])
            scanpathR = np.array([[fixation[0], fixation[1], fixation[3], fixation[2]] for fixation in scanpath_line[1]])

            # Skip pairs where either scanpath is empty
            if len(scanpathL) == 0 or len(scanpathR) == 0:
            #if len(scanpathL) == 0 and len(scanpathR) == 0:
                print(f"Warning: Skipping scanpath pair with empty scanpath on both sides.")
                continue

            if len(scanpath_line) == 3:
                scanpathPairs.append([scanpathL, scanpathR, scanpath_line[2][0]])
            else:
                print("bad data in Pickle file, missing label info?")

            
            '''
            if '_neutral' in os.path.basename(file_path).lower():
                labels.extend([0] * len(data))
            else:
                labels.extend([1] * len(data))
            '''

    # reset X values to left image edge (500 on left, 1050 on right)
    '''
    minPosX = 1920/2 # DDEBUG CONST 
    for scanpath in scanpaths:
        if scanpath[:, 0].mean() < minPosX:
            scanpath[:, 0] = scanpath[:, 0] - 500  # shift left image to start at 0
        else:
            scanpath[:, 0] = scanpath[:, 0] - 1050  # shift right image to start at 0
    '''

    '''
    #--------------------------------
    # _print data  statistice:
    #-----------------------------
    _print(f"Total scanpaths loaded: {len(scanpaths)}")
    n_neutral = sum(1 for label in labels if label == 0)
    n_emotional = sum(1 for label in labels if label == 1)
    _print(f"  Neutral scanpaths: {n_neutral}")
    _print(f"  Emotional scanpaths: {n_emotional}")  

    # calc mean lengthg of scanpaths
    lengths = [len(sp) for sp in scanpaths]
    mean_length = np.mean(lengths)
    _print(f"Mean scanpath length: {mean_length:.2f} fixations")
    '''
    # count how many scanpath pairs were loaded
    _print(f"Total scanpath pairs loaded: {len(scanpathPairs)}")
    # count how many 0 and how many 1 labels are in the data
    n_label0 = sum(1 for pair in scanpathPairs if pair[2] == 0)
    n_label1 = sum(1 for pair in scanpathPairs if pair[2] == 1)
    _print(f"  Label Left vs Right : {n_label0} : {n_label1}")
    
    return scanpathPairs


def displayScanpathData(file_path): 
    import pickle
    import matplotlib.pyplot as plt

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    for scanpath_pair in data:
        sp1 = np.array(scanpath_pair[0])
        sp2 = np.array(scanpath_pair[1])
        label = scanpath_pair[2][0]
        print (sp1, " - ", sp2, f"Label: {label}")


# Function that convert Picle file to a text (JSON) file for easier reading
def convertPickleToJSON(pickle_path, json_path):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    # Convert numpy arrays to lists for JSON serialization
    json_data = []
    for item in data:
        sp1 = item[0].tolist() if hasattr(item[0], 'tolist') else item[0]
        sp2 = item[1].tolist() if hasattr(item[1], 'tolist') else item[1]
        label = item[2]
        json_data.append((sp1, sp2, label))

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=4)

    print(f"Converted {pickle_path} to {json_path}")
    return json_path



def writeScanpathTXTFile(sample_data):
    """
    Write scanpath data to a text file for easier reading

    Args:
        sample_data: List of tuples (sp1, sp2, label)
    """
    save_path = 'scanpath_data.txt'
    with open(save_path, 'w') as f:
        for idx, (sp1, sp2, label) in enumerate(sample_data):
            f.write(f"Scanpath Pair {idx+1}:\n")
            f.write(f"  Scanpath 1 ({len(sp1)} fixations):\n")
            for fixation in sp1:
                f.write(f"    {fixation}\n")
            f.write(f"  end_sp\n")
            f.write(f"  Scanpath 2 ({len(sp2)} fixations):\n")
            for fixation in sp2:
                f.write(f"    {fixation}\n")
            f.write(f"  end_sp\n")
            f.write(f"  Label: {label}\n\n")
    print(f"Scanpath data written to {save_path}")
    return save_path