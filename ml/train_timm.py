import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms, datasets
from sklearn.metrics import f1_score
from tqdm import tqdm
import timm

"""Entrenamiento avanzado para alta precisión usando timm."""

def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class MixupCutmixWrapper:
    def __init__(self, num_classes, mixup_alpha=0.4, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        self.num_classes = num_classes
    def _sample_lambda(self, alpha):
        if alpha <= 0:
            return 1.0
        lam = np.random.beta(alpha, alpha)
        return lam
    def __call__(self, x, y):
        if np.random.rand() > self.prob:
            return x, y, 1.0
        use_cutmix = np.random.rand() < self.switch_prob
        if use_cutmix and self.cutmix_alpha > 0:
            lam = self._sample_lambda(self.cutmix_alpha)
            batch = x.size(0)
            idx = torch.randperm(batch, device=x.device)
            H, W = x.size(2), x.size(3)
            cut_w = int(W * (1 - lam) ** 0.5)
            cut_h = int(H * (1 - lam) ** 0.5)
            cx = np.random.randint(W)
            cy = np.random.randint(H)
            x1 = np.clip(cx - cut_w // 2, 0, W)
            y1 = np.clip(cy - cut_h // 2, 0, H)
            x2 = np.clip(cx + cut_w // 2, 0, W)
            y2 = np.clip(cy + cut_h // 2, 0, H)
            x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
            lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
            y_a, y_b = y, y[idx]
            return (x, (y_a, y_b, lam), lam)
        else:
            lam = self._sample_lambda(self.mixup_alpha)
            batch = x.size(0)
            idx = torch.randperm(batch, device=x.device)
            mixed = lam * x + (1 - lam) * x[idx]
            y_a, y_b = y, y[idx]
            return (mixed, (y_a, y_b, lam), lam)

def get_loaders(data_dir: Path, img_size=224, batch_size=32, val_split=0.15, balanced=True):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.25,0.25,0.25,0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size*1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    full_ds = datasets.ImageFolder(str(data_dir), transform=train_tf)
    classes = full_ds.classes
    n_val = int(len(full_ds) * val_split)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])
    val_ds.dataset.transform = val_tf
    if balanced:
        counts = [0]*len(classes)
        for _, lbl in train_ds: counts[lbl]+=1
        total = sum(counts)
        weights_per_class = [total/c for c in counts]
        sample_weights = [weights_per_class[lbl] for _, lbl in train_ds]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, classes

def smooth_loss(logits, target, smoothing=0.1):
    if smoothing <= 0:
        return nn.functional.cross_entropy(logits, target)
    n = logits.size(1)
    log_probs = nn.functional.log_softmax(logits, dim=1)
    with torch.no_grad():
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(smoothing / (n - 1))
        true_dist.scatter_(1, target.unsqueeze(1), 1 - smoothing)
    return torch.mean(torch.sum(-true_dist * log_probs, dim=1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--model', default='efficientnetv2_rw_s')
    ap.add_argument('--img_size', type=int, default=384)
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--mixup', type=float, default=0.4)
    ap.add_argument('--cutmix', type=float, default=1.0)
    ap.add_argument('--smoothing', type=float, default=0.1)
    ap.add_argument('--patience', type=int, default=7)
    ap.add_argument('--val_split', type=float, default=0.15)
    ap.add_argument('--balanced', action='store_true')
    ap.add_argument('--out_dir', default='outputs_adv')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--export_onnx', action='store_true')
    args = ap.parse_args()

    seed_all(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, classes = get_loaders(data_dir, args.img_size, args.batch_size, args.val_split, balanced=args.balanced)
    num_classes = len(classes)
    model = timm.create_model(args.model, pretrained=True, num_classes=num_classes).to(device)

    mixup_fn = MixupCutmixWrapper(num_classes, mixup_alpha=args.mixup, cutmix_alpha=args.cutmix) if (args.mixup>0 or args.cutmix>0) else None

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=device=='cuda')

    best_f1 = 0.0
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs+1):
        model.train()
        train_loss_accum = 0.0
        for imgs, labels in tqdm(train_loader, desc=f'train {epoch}', leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            if mixup_fn:
                imgs, mix_tuple, lam = mixup_fn(imgs, labels)
            with torch.cuda.amp.autocast(enabled=device=='cuda'):
                logits = model(imgs)
                if mixup_fn and isinstance(mix_tuple, tuple):
                    y_a, y_b, lam = mix_tuple
                    loss = lam * smooth_loss(logits, y_a, args.smoothing) + (1-lam)*smooth_loss(logits, y_b, args.smoothing)
                else:
                    loss = smooth_loss(logits, labels, args.smoothing)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_accum += loss.item() * imgs.size(0)
        train_loss = train_loss_accum / len(train_loader.dataset)

        model.eval()
        val_loss_accum = 0.0
        preds = []
        gts = []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f'val {epoch}', leave=False):
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.cuda.amp.autocast(enabled=device=='cuda'):
                    logits = model(imgs)
                    loss = nn.functional.cross_entropy(logits, labels)
                val_loss_accum += loss.item()*imgs.size(0)
                pred = logits.argmax(1).cpu().numpy().tolist()
                preds.extend(pred)
                gts.extend(labels.cpu().numpy().tolist())
        val_loss = val_loss_accum / len(val_loader.dataset)
        val_f1 = f1_score(gts, preds, average='macro')
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'val_f1': val_f1})
        print(f'Epoch {epoch}/{args.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f} best_f1={best_f1:.4f}')

        improved = val_f1 > best_f1
        if improved:
            best_f1 = val_f1
            patience_counter = 0
            torch.save({'model_state': model.state_dict(), 'classes': classes, 'f1': best_f1}, out_dir/'best_adv.pt')
            ts = torch.jit.trace(model, torch.randn(1,3,args.img_size,args.img_size, device=device))
            ts.save(str(out_dir/'model_ts.pt'))
            if args.export_onnx:
                onnx_path = out_dir/'model.onnx'
                torch.onnx.export(model, torch.randn(1,3,args.img_size,args.img_size, device=device), str(onnx_path),
                                  input_names=['input'], output_names=['logits'], opset_version=17)
        else:
            patience_counter += 1

        scheduler.step()
        with open(out_dir/'history_adv.json', 'w') as f:
            json.dump(history, f, indent=2)
        if patience_counter >= args.patience:
            print('Early stopping (macro F1).')
            break

    print('Entrenamiento finalizado. Mejor macro F1:', best_f1)

if __name__ == '__main__':
    main()
