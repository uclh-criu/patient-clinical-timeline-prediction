import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, classification_report, 
                             precision_recall_curve)
import torch
from tqdm import tqdm

def load_and_extract_entities(directory: Path, label: int) -> list:
    """Loads all timeline JSON files from a directory and extracts entities."""
    timeline_files = sorted(directory.glob("*.json"))
    print(f"Found {len(timeline_files)} timeline files in {directory.name}")
    
    patient_data = []
    for timeline_file in timeline_files:
        with open(timeline_file, 'r') as f:
            timeline = json.load(f)
            patient_id = timeline['patient_id']
            
            # Extract all entity preferred names from events (lowercase)
            entities = []
            for event in timeline['events']:
                entity_name = event.get('entity_preferred_name')
                if entity_name is not None:
                    entities.append(entity_name.lower().strip())
            
            # Store patient data
            patient_data.append({
                'patient_id': patient_id,
                'entities': entities,
                'num_entities': len(entities),
                'label': label
            })
    return patient_data

def encode_and_pad(entities: list, vocab: dict, max_len: int) -> list:
    """Encodes a list of entities into a padded sequence of integer IDs."""
    # Convert entities to their corresponding vocab IDs, use <unk> for out-of-vocab
    encoded = [vocab.get(word, vocab['<unk>']) for word in entities]
    
    # Truncate if longer than max_len
    if len(encoded) > max_len:
        encoded = encoded[:max_len]
    
    # Pad with <pad> token ID if shorter
    else:
        padding_needed = max_len - len(encoded)
        encoded.extend([vocab['<pad>']] * padding_needed)
        
    return encoded

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Function to handle the training of the model for one epoch."""
    model.train()
    total_loss = 0
    
    # Wrap dataloader for a progress bar
    progress_bar = tqdm(dataloader, desc='Training', leave=False)
    
    for batch in progress_bar:
        sequences = batch['sequence'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        predictions = model(sequences).squeeze(1)
        loss = criterion(predictions, labels)
        
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    """Function to evaluate the model on a dataset."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    # Wrap dataloader for a progress bar
    progress_bar = tqdm(dataloader, desc='Evaluating', leave=False)
    
    with torch.no_grad():
        for batch in progress_bar:
            sequences = batch['sequence'].to(device)
            labels = batch['label'].to(device)
            
            predictions = model(sequences).squeeze(1)
            loss = criterion(predictions, labels)
            total_loss += loss.item()
            
            preds_binary = torch.round(torch.sigmoid(predictions))
            all_predictions.extend(preds_binary.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    
    return avg_loss, accuracy

def get_predictions(model, dataloader, device):
    """Function to get model predictions and actual labels from a dataloader."""
    model.eval()
    all_predictions = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            sequences = batch['sequence'].to(device)
            labels = batch['label'].to(device)
            
            # Get raw logits
            predictions_logits = model(sequences).squeeze(1)
            
            # Get probabilities
            predictions_probs = torch.sigmoid(predictions_logits)
            
            # Get binary predictions (at 0.5 threshold for now)
            predictions_binary = torch.round(predictions_probs)

            all_predictions.extend(predictions_binary.cpu().numpy())
            all_probs.extend(predictions_probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return np.array(all_labels), np.array(all_predictions), np.array(all_probs)

def plot_training_curves(train_losses, test_losses, train_accuracies, test_accuracies, title_suffix=""):
    """
    Plot training and validation loss and accuracy curves.
    
    Args:
        train_losses: List of training losses per epoch
        test_losses: List of test losses per epoch
        train_accuracies: List of training accuracies per epoch
        test_accuracies: List of test accuracies per epoch
        title_suffix: Optional string to append to plot titles
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plotting loss
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(test_losses, label='Test Loss')
    ax1.set_ylabel('Loss (BCEWithLogits)')
    ax1.set_title(f'Training and Test Loss Over Epochs{title_suffix}')
    ax1.legend()
    ax1.grid(True)
    
    # Plotting accuracy
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(test_accuracies, label='Test Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'Training and Test Accuracy Over Epochs{title_suffix}')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def find_optimal_threshold(y_true, y_probs, plot=True):
    """
    Find the optimal threshold that maximizes F1 score.
    
    Args:
        y_true: True binary labels
        y_probs: Predicted probabilities for positive class
        plot: Whether to plot the precision-recall-F1 curve
        
    Returns:
        best_threshold: The threshold that maximizes F1
        best_f1: The F1 score at that threshold
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    thresholds_for_plot = np.append(thresholds, 1)
    
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    f1_scores = np.nan_to_num(f1_scores)
    
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx]
    best_f1 = f1_scores[best_f1_idx]
    
    print(f"Best Threshold (maximizes F1 score): {best_threshold:.4f}")
    print(f"F1 Score at this threshold: {best_f1:.4f}")
    
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds_for_plot, precisions, "b--", label="Precision")
        plt.plot(thresholds_for_plot, recalls, "g-", label="Recall")
        plt.plot(thresholds_for_plot, f1_scores, "r-", label="F1 Score", linewidth=2)
        plt.axvline(x=best_threshold, color='k', linestyle='--', 
                   label=f'Best Threshold ({best_threshold:.2f})')
        plt.title('Precision, Recall, and F1 Score by Decision Threshold')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()
    
    return best_threshold, best_f1

def evaluate_model(y_true, y_probs, threshold=0.5, title_suffix=""):
    """
    Comprehensive model evaluation with metrics, classification report, and confusion matrix.
    
    Args:
        y_true: True binary labels
        y_probs: Predicted probabilities for positive class
        threshold: Decision threshold for binary predictions
        title_suffix: Optional string to append to plot titles
    """
    y_pred = (y_probs >= threshold).astype(int)
    
    # Core metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_probs)
    
    print(f"\n=== Test Set Performance (threshold={threshold:.4f}){title_suffix} ===")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-score : {f1:.3f}")
    print(f"ROC-AUC  : {roc_auc:.3f}")
    
    # Classification report
    print("\nDetailed classification report:")
    print(classification_report(y_true, y_pred, digits=3))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion matrix:")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Control (0)', 'Treatment (1)'], 
                yticklabels=['Control (0)', 'Treatment (1)'])
    plt.title(f'Confusion Matrix{title_suffix}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }