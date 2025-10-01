import argparse
from pathlib import Path
import json
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

"""Evalúa un modelo guardado (best.pt + pesos) sobre un directorio de test."""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ts_model', type=str, required=True)
    ap.add_argument('--classes_json', type=str, required=True)
    ap.add_argument('--test_dir', type=str, required=True)
    ap.add_argument('--img_size', type=int, default=384)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--out_dir', type=str, default='eval_outputs')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.classes_json, 'r') as f:
        classes = json.load(f)

    tfm = transforms.Compose([
        transforms.Resize(int(args.img_size*1.15)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    ds = datasets.ImageFolder(args.test_dir, transform=tfm)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = torch.jit.load(args.ts_model, map_location=device)
    model.eval()

    preds = []
    gts = []
    with torch.inference_mode():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            out = model(imgs)
            prob = F.softmax(out, dim=1)
            pred = prob.argmax(1).cpu().numpy()
            preds.extend(pred.tolist())
            gts.extend(labels.numpy().tolist())

    report = classification_report(gts, preds, target_names=classes, output_dict=True)
    f1_macro = f1_score(gts, preds, average='macro')
    with open(out_dir/'report.json', 'w') as f:
        json.dump({'f1_macro': f1_macro, 'report': report}, f, indent=2)

    cm = confusion_matrix(gts, preds, labels=list(range(len(classes))))
    fig, ax = plt.subplots(figsize=(max(8, len(classes)*0.3), max(6, len(classes)*0.3)))
    sns.heatmap(cm, ax=ax, cmap='Blues', cbar=True, xticklabels=classes, yticklabels=classes, fmt='d', annot=len(classes)<=40)
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Real')
    ax.set_title(f'Matriz de confusión (F1 macro={f1_macro:.3f})')
    plt.tight_layout()
    fig.savefig(out_dir/'confusion_matrix.png', dpi=200)
    print('Evaluación completa. F1 macro:', f1_macro)

if __name__ == '__main__':
    main()
