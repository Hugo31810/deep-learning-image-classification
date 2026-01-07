import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, train_loader, loss_fn, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return running_loss / len(train_loader), correct / total


def evaluate(model, loader, loss_fn, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / len(loader), correct / total


def train_and_evaluate(model, train_loader, val_loader, loss_fn, optimizer, device, num_epochs=10, test_loader=None,
                       name='None'):
    results = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    for epoch in range(num_epochs):
        t_loss, t_acc = train(model, train_loader, loss_fn, optimizer, device)
        v_loss, v_acc = evaluate(model, val_loader, loss_fn, device)
        results['train_loss'].append(t_loss)
        results['val_loss'].append(v_loss)
        results['train_acc'].append(t_acc)
        results['val_acc'].append(v_acc)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {t_loss:.4f}, Val Acc: {v_acc:.4f}")

    pd.DataFrame(results).to_csv(f'results_{name}.csv', index=False)  # Simplificado para el ejemplo
    # Guardar archivos específicos como pide el enunciado
    pd.DataFrame({'Epoch': range(1, num_epochs + 1), 'Train Loss': results['train_loss']}).to_csv(
        f'train_loss_{name}.csv', index=False)
    pd.DataFrame({'Epoch': range(1, num_epochs + 1), 'Validation Loss': results['val_loss']}).to_csv(
        f'valid_loss_{name}.csv', index=False)
    pd.DataFrame({'Epoch': range(1, num_epochs + 1), 'Train Accuracy': results['train_acc']}).to_csv(
        f'train_accuracy_{name}.csv', index=False)
    pd.DataFrame({'Epoch': range(1, num_epochs + 1), 'Validation Accuracy': results['val_acc']}).to_csv(
        f'valid_accuracy_{name}.csv', index=False)

    test_acc = None
    if test_loader:
        _, test_acc = evaluate(model, test_loader, loss_fn, device)
        pd.DataFrame({'Test Accuracy': [test_acc]}).to_csv(f'test_accuracy_{name}.csv', index=False)
    return results


def save_full_model(model, file_name):
    torch.save(model, file_name)


def load_full_model(file_name):
    # Usamos weights_only=False para evitar el error de seguridad en PyTorch 2.6+
    return torch.load(file_name, weights_only=False)


def plot_loss_accuracy(train_loss_file, valid_loss_file, train_accuracy_file, valid_accuracy_file):
    t_loss = pd.read_csv(train_loss_file)
    v_loss = pd.read_csv(valid_loss_file)
    t_acc = pd.read_csv(train_accuracy_file)
    v_acc = pd.read_csv(valid_accuracy_file)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(t_loss.iloc[:, 0], t_loss.iloc[:, 1], label='Train');
    ax1.plot(v_loss.iloc[:, 0], v_loss.iloc[:, 1], label='Val')
    ax1.set_title('Loss');
    ax1.legend()
    ax2.plot(t_acc.iloc[:, 0], t_acc.iloc[:, 1], label='Train');
    ax2.plot(v_acc.iloc[:, 0], v_acc.iloc[:, 1], label='Val')
    ax2.set_title('Accuracy');
    ax2.legend()
    plt.show()


def plot_confusion_matrix(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy());
            all_labels.extend(labels.cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8));
    sns.heatmap(cm, cmap='Blues');
    plt.show()
    return cm


def plot_error_per_class(cm):
    error_per_class = 1 - (cm.diagonal() / cm.sum(axis=1))
    plt.figure(figsize=(10, 5));
    plt.bar(range(len(cm)), error_per_class * 100, color='orange')
    plt.title('Error por clase (%)');
    plt.show()