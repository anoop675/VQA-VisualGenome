import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from tqdm import tqdm
from config import EPOCHS, DEVICE


def evaluate(model, val_dataloader, criterion, device=DEVICE):
    model.eval()
    total_val_loss = 0.0
    predictions = []
    true_vals = []
    conf = []

    with torch.no_grad():
        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            inputs = {
                'image_embedding': batch['image_embedding'],
                'text_embedding': batch['text_embedding'],
                'image_mask': batch.get('image_mask', None),
                'text_mask': batch.get('text_mask', None)
            }

            outputs = model(**inputs)
            labels = batch['label']
            loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
            total_val_loss += loss.item()
            probs = torch.max(outputs.softmax(dim=1), dim=-1)[0].detach().cpu().numpy()
            outputs = outputs.argmax(-1)
            predictions.append(outputs.detach().cpu().numpy())
            true_vals.append(labels.cpu().numpy())
            conf.append(probs)

    loss_val_avg = total_val_loss / len(val_dataloader)
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    conf = np.concatenate(conf, axis=0)

    return loss_val_avg, predictions, true_vals, conf


def train_loop(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, epochs=EPOCHS, device=DEVICE):
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        train_predictions = []
        train_true_vals = []

        for batch in tqdm(train_dataloader, desc=f"Train epoch {epoch}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            inputs = {
                'image_embedding': batch['image_embedding'],
                'text_embedding': batch['text_embedding'],
                'image_mask': batch.get('image_mask', None),
                'text_mask': batch.get('text_mask', None)
            }

            labels = batch['label']
            outputs = model(**inputs)
            loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()
            train_predictions.append(outputs.argmax(-1).detach().cpu().numpy())
            train_true_vals.append(labels.cpu().numpy())

        train_preds = np.concatenate(train_predictions, axis=0)
        train_trues = np.concatenate(train_true_vals, axis=0)
        train_acc = accuracy_score(train_preds, train_trues)
        val_loss, val_preds, val_trues, _ = evaluate(model, val_dataloader, criterion, device=device)
        val_acc = accuracy_score(val_preds, val_trues)
        print(
            f"Epoch {epoch}: train_loss={total_train_loss / len(train_dataloader):.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model_vg_fusion.pth")
            print("Saved best model with val_acc:", best_val_acc)

        return best_val_acc


