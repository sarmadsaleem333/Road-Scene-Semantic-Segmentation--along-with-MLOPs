import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, device, scheduler=None, save_path="best_model.pth"):
    model.to(device)
    best_val_acc = 0.0  # Track the best validation accuracy

    # Store metrics for plotting
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        total_correct = 0
        total_pixels = 0

        print(f"Epoch [{epoch+1}/{num_epochs}]")

        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device).long()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_pixels += torch.numel(labels)

        avg_train_loss = train_loss / len(train_loader)
        train_pixel_acc = total_correct / total_pixels
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_pixel_acc)

        print(f"Train Loss: {avg_train_loss:.4f} | Pixel Acc: {train_pixel_acc:.4f}")

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Val Loss: {val_loss:.4f} | Val Pixel Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"Validation accuracy improved! Saving model...")
            torch.save(model.state_dict(), save_path)

        if scheduler:
            scheduler.step()

    print("Training complete!")

    # Plot training and validation curves
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)


def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    total_correct = 0
    total_pixels = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device).long()
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_pixels += torch.numel(labels)

    avg_loss = val_loss / len(val_loader)
    pixel_acc = total_correct / total_pixels

    return avg_loss, pixel_acc


def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Pixel Acc')
    plt.plot(epochs, val_accuracies, label='Val Pixel Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Pixel Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()
