"""
Simple inference script for BiLSTM Scanpath Pair Model

Usage:
    python inference.py --checkpoint path/to/model.pth --sp1 scanpath1.json --sp2 scanpath2.json

    Or use Python API:
    from inference import load_model, predict_pair
    model = load_model('checkpoint.pth')
    prob = predict_pair(model, sp1_data, sp2_data)

Input format:
    Each scanpath should be a list of fixations: [[x,y,t,d], [x,y,t,d], ...]
    Scanpaths can have different lengths.
"""

import torch
import torch.nn as nn
import argparse
import json
import pickle
import numpy as np
from LSTM import ScanpathPairModel

'''
print("\033[31mThis is red text\033[0m")
print("\033[32mThis is green text\033[0m")
print("\033[33mThis is yellow text\033[0m")
print("\033[34mThis is blue text\033[0m")
print("\033[35mThis is magenta text\033[0m")
print("\033[36mThis is cyan text\033[0m")
print("\033[37mThis is white text\033[0m")
'''

RED_START = "\033[31m"
GREEN_START = "\033[32m"
YELLOW_START = "\033[33m"
BLUE_START = "\033[34m"
MAGENTA_START = "\033[35m"
CYAN_START = "\033[36m"
WHITE_START = "\033[37m"
COLOR_END = "\033[0m"

def load_model(checkpoint_path, device='cpu'):
    """
    Load trained BiLSTM model from checkpoint

    Args:
        checkpoint_path: Path to .pth checkpoint file
        device: 'cpu' or 'cuda'

    Returns:
        model: Loaded model in eval mode
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Try to detect model dimensions from checkpoint, otherwise use defaults
    # New models (after tuning): hidden_dim=16, emb_dim=16
    # Old models (before tuning): hidden_dim=32/64, emb_dim=32/64
    try:
        # Try to load with new dimensions
        model = ScanpathPairModel(input_dim=4, hidden_dim=16, emb_dim=16)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded with hidden_dim=16, emb_dim=16 (tuned for small dataset)")
    except RuntimeError:
        try:
            # Try with previous dimensions
            model = ScanpathPairModel(input_dim=4, hidden_dim=32, emb_dim=32)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded with hidden_dim=32, emb_dim=32 (previous configuration)")
        except RuntimeError:
            # Fallback to original dimensions
            model = ScanpathPairModel(input_dim=4, hidden_dim=64, emb_dim=64)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded with hidden_dim=64, emb_dim=64 (original configuration)")

    model.to(device)
    model.eval()

    print(f"Loaded from: {checkpoint_path}")
    print(f"  Trained epoch: {checkpoint['epoch']}")
    print(f"  Training loss: {checkpoint['loss']:.4f}")

    return model


def predict_pair(model, sp1, sp2, device='cpu'):
    """
    Predict comparison result for a pair of scanpaths

    Args:
        model: Trained ScanpathPairModel
        sp1: First scanpath as list/array of shape [N, 4] where each row is [x,y,t,d]
        sp2: Second scanpath as list/array of shape [M, 4] where each row is [x,y,t,d]
        device: 'cpu' or 'cuda'

    Returns:
        probability: Float between 0-1 indicating prediction confidence
        prediction: Binary prediction (0 or 1)
    """
    # Convert to numpy if needed
    if not isinstance(sp1, np.ndarray):
        sp1 = np.array(sp1, dtype=np.float32)
    if not isinstance(sp2, np.ndarray):
        sp2 = np.array(sp2, dtype=np.float32)

    # Ensure only first 4 features are used
    sp1 = sp1[:, :4]
    sp2 = sp2[:, :4]

    # Check for empty scanpaths
    if len(sp1) == 0 or len(sp2) == 0:
        raise ValueError(f"Empty scanpath detected: sp1 length={len(sp1)}, sp2 length={len(sp2)}")

    # Convert to tensors and add batch dimension
    sp1_tensor = torch.tensor(sp1, dtype=torch.float32).unsqueeze(0).to(device)  # [1, N, 4]
    sp2_tensor = torch.tensor(sp2, dtype=torch.float32).unsqueeze(0).to(device)  # [1, M, 4]
    len1 = torch.tensor([len(sp1)], dtype=torch.long).to(device)
    len2 = torch.tensor([len(sp2)], dtype=torch.long).to(device)

    # Run inference
    with torch.no_grad():
        prob = model(sp1_tensor, len1, sp2_tensor, len2)
        prob = prob.item()
        prediction = 1 if prob > 0.5 else 0

    return prob, prediction

'''
def load_scanpath_from_file(file_path):
    """
    Load a single scanpath from file

    Args:
        file_path: Path to .json, .pkl, .npy, or .txt file

    Returns:
        scanpath: numpy array of shape [N, 4] or [N, k] where k >= 4
    """
    if file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
    elif file_path.endswith('.pkl'):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    elif file_path.endswith('.npy'):
        data = np.load(file_path)
    elif file_path.endswith('.txt'):
        data = np.loadtxt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    # Convert to numpy array
    scanpath = np.array(data, dtype=np.float32)

    # Ensure 2D array
    if scanpath.ndim == 1:
        scanpath = scanpath.reshape(-1, 4)

    print(f"Loaded scanpath from {file_path}: shape={scanpath.shape}")

    return scanpath
'''

# is float - (integers work too) 
def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False
    

def readScanPathTxt(file_path):
    scanpathPair = []
    scanpath = []
    spIndex = 0
    with open(file_path, 'r') as f:
        for line in f:
            # remove '[' and ']' if present
            line = line.replace('[', '').replace(']', '')

            parts = line.strip().split()
            # check that all parts are floatable
            if len(parts) == 4 and all(is_float(part) for part in parts):
                    fixation = [float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])]
                    scanpath.append(fixation)
            else:
                # check if the string 'scanpath' exist in string
                if len(scanpath) > 0 and 'end_sp' in line.lower():
                    scanpathPair.append(np.array(scanpath, dtype=np.float32))
                    scanpath = []
                    spIndex += 1
                    continue

    return scanpathPair[0], scanpathPair[1]

    return np.array(scanpath, dtype=np.float32)

def readRawScanPath(file_path):
    with open(file_path, 'r') as f:
        raw_data = json.load(f)

    ScanPair=[]
    #sp1, sp2  = load_scanpath_from_file(args.sp1)
    for item in raw_data:
        # convert file format [x,y,d,t...] to [x,y, t, d]
        ScanPair.append(np.array([[fixation[0], fixation[1], fixation[3], fixation[2]] for fixation in item]))

    sp1 = ScanPair[0]
    sp2 = ScanPair[1]
    return sp1, sp2


def main():
    parser = argparse.ArgumentParser(description='BiLSTM Scanpath Pair Inference')
    parser.add_argument('--checkpoint', type=str, 
                        default='results/result_1106_093614/checkpoints/best_model.pth',
                        help='Path to model checkpoint (.pth file)')
    '''
    parser.add_argument('--sp1', type=str, required=True,
                        help='Path to first scanpath file (.json, .pkl, .npy, .txt)')
    parser.add_argument('--sp2', type=str, required=True,
                        help='Path to second scanpath file (.json, .pkl, .npy, .txt)')
    '''
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to run inference on')

    parser.add_argument('--scantxt', type=str, default='fixation2.txt',help=' image to inference ')

    args = parser.parse_args()

    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    print("="*60)
    print("BiLSTM Scanpath Pair - Inference")
    print("="*60)

    # Load model
    print(GREEN_START, "\n[1/3] Loading model : ", args.checkpoint, "...", COLOR_END)
    model = load_model(args.checkpoint, args.device)

    # Load scanpaths
    print("\n[2/3] Loading scanpaths...")

    #sp1, sp2 = readRawScanPath("fixation1.json")
    #sp1, sp2 = readScanPathTxt("fixation2.txt")
    sp1, sp2 = readScanPathTxt(args.scantxt)
    print(GREEN_START,'inference image : ', args.scantxt, COLOR_END)
    
    # Run inference
    print("\n[3/3] Running inference...")
    probability, prediction = predict_pair(model, sp1, sp2, args.device)

    # Print results
    print("\n" + "="*60)
    print("INFERENCE RESULTS")
    print("="*60)
    print(f"Probability:  {probability:.4f}")
    print(GREEN_START,f"Prediction:   {prediction} ({'Positive' if prediction == 1 else 'Negative'})", COLOR_END)
    print(f"Confidence:   {abs(probability - 0.5) * 200:.1f}%")
    print("="*60)

    return probability, prediction


if __name__ == '__main__':
    main()
