import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
try:
    from sklearn.metrics import f1_score
except ImportError:
    def f1_score(y_true, y_pred, average='macro'):
        y_true = list(y_true)
        y_pred = list(y_pred)
        labels = sorted(set(y_true) | set(y_pred))
        f1s = []
        for lbl in labels:
            tp = sum(1 for t,p in zip(y_true,y_pred) if t==lbl and p==lbl)
            fp = sum(1 for t,p in zip(y_true,y_pred) if t!=lbl and p==lbl)
            fn = sum(1 for t,p in zip(y_true,y_pred) if t==lbl and p!=lbl)
            precision = tp / (tp+fp) if (tp+fp)>0 else 0.0
            recall = tp / (tp+fn) if (tp+fn)>0 else 0.0
            f1s.append(0.0 if precision+recall==0 else 2*precision*recall/(precision+recall))
        return sum(f1s)/len(f1s) if f1s else 0.0
from tqdm import tqdm

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_dataloaders(data_dir: Path, image_size=224, batch_size=32, val_split=0.15, balanced=True):
    tfm_train = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    tfm_val = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    full_ds = datasets.ImageFolder(str(data_dir), transform=tfm_train)
    class_names = full_ds.classes
    n_total = len(full_ds)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])
    val_ds.dataset.transform = tfm_val
    if balanced:
        counts = [0]*len(class_names)
        for _, idx in train_ds:
            counts[idx] += 1
        total = sum(counts)
        weights_per_class = [total/c for c in counts]
        sample_weights = [weights_per_class[label] for _, label in train_ds]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=2)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, class_names

def build_model(num_classes: int, arch: str = "efficientnet_b0", pretrained=True):
    if arch.startswith("efficientnet"):
        model_fn = getattr(models, arch)
        model = model_fn(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained and arch=="efficientnet_b0" else None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    return model

def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds = []
    gts = []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="val", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            running_loss += loss.item() * imgs.size(0)
            pred = out.argmax(dim=1).cpu().numpy()
            preds.extend(pred.tolist())
            gts.extend(labels.cpu().numpy().tolist())
    avg_loss = running_loss / len(loader.dataset)
    f1 = f1_score(gts, preds, average='macro')
    return avg_loss, f1, gts, preds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Directorio raiz de imagenes (species/disease/img)')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--arch', type=str, default='efficientnet_b0')
    parser.add_argument('--out_dir', type=str, default='outputs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--balanced', action='store_true')
    parser.add_argument('--log_every', type=int, default=50, help='Frecuencia (batches) para imprimir progreso de entrenamiento')
    args = parser.parse_args()

    seed_everything(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, class_names = build_dataloaders(data_dir, balanced=args.balanced)
    model = build_model(num_classes=len(class_names), arch=args.arch).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = 0
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs+1):
        scaler = torch.cuda.amp.GradScaler(enabled=device=='cuda')
        model.train()
        running_loss = 0.0
        seen = 0
        total_batches = len(train_loader)
        pbar = tqdm(enumerate(train_loader), total=total_batches, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for i, (imgs, labels) in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device=='cuda'):
                out = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_size = imgs.size(0)
            running_loss += loss.item() * batch_size
            seen += batch_size
            if (i+1) % args.log_every == 0 or (i+1)==total_batches:
                current_avg = running_loss / seen
                lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({'loss': f"{current_avg:.4f}", 'lr': f"{lr:.2e}"})
        train_loss = running_loss / seen
        val_loss, val_f1, gts, preds = eval_epoch(model, val_loader, criterion, device)
        print(f"[Epoch {epoch}/{args.epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f} lr={optimizer.param_groups[0]['lr']:.2e}")
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'val_f1': val_f1})
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({'model_state': model.state_dict(), 'classes': class_names}, out_dir / 'best.pt')
            patience_counter = 0
        else:
            patience_counter += 1
        scheduler.step()
        if patience_counter >= args.patience:
            print('Early stopping por paciencia alcanzada.')
            break
    with open(out_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    model.eval()
    example = torch.randn(1,3,224,224).to(device)
    traced = torch.jit.trace(model, example)
    traced.save(str(out_dir / 'model_ts.pt'))
    print('Entrenamiento finalizado. Mejor F1:', best_f1)

if __name__ == '__main__':
    main()
