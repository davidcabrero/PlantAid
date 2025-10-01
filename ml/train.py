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
# Intentamos importar la función f1_score de sklearn.metrics
try:
    from sklearn.metrics import f1_score
# Si no está disponible, definimos una implementación personalizada de f1_score
except ImportError:
    # Definimos la función f1_score personalizada
    def f1_score(y_true, y_pred, average='macro'):
        # Convertimos las entradas a listas
        y_true = list(y_true)
        y_pred = list(y_pred)
        # Obtenemos las etiquetas únicas presentes en y_true o y_pred
        labels = sorted(set(y_true) | set(y_pred))
        f1s = []  # Lista para almacenar los valores F1 por clase
        # Iteramos sobre cada etiqueta
        for lbl in labels:
            # Calculamos los verdaderos positivos (tp)
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p == lbl)
            # Calculamos los falsos positivos (fp)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != lbl and p == lbl)
            # Calculamos los falsos negativos (fn)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p != lbl)
            # Calculamos la precisión (precision)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            # Calculamos el recall
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            # Calculamos el F1 para la etiqueta actual
            f1s.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
        # Devolvemos el promedio de los valores F1 si existen, de lo contrario devolvemos 0.0
        return sum(f1s) / len(f1s) if f1s else 0.0

# Importamos tqdm para mostrar barras de progreso en los bucles
from tqdm import tqdm

def seed_everything(seed=42):
    # Establece la semilla para generar números aleatorios en Python, NumPy y PyTorch
    random.seed(seed)  # Semilla para el módulo random
    np.random.seed(seed)  # Semilla para NumPy
    torch.manual_seed(seed)  # Semilla para PyTorch en CPU
    torch.cuda.manual_seed_all(seed)  # Semilla para PyTorch en GPU

def build_dataloaders(data_dir: Path, image_size=224, batch_size=32, val_split=0.15, balanced=True):
    # Define las transformaciones para los datos de entrenamiento
    tfm_train = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Redimensiona las imágenes
        transforms.RandomHorizontalFlip(),  # Voltea horizontalmente de forma aleatoria
        transforms.RandomVerticalFlip(),  # Voltea verticalmente de forma aleatoria
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),  # Ajusta brillo, contraste, saturación y tono
        transforms.RandomRotation(15),  # Rota aleatoriamente hasta 15 grados
        transforms.ToTensor(),  # Convierte la imagen a tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normaliza con valores predefinidos
    ])
    # Define las transformaciones para los datos de validación
    tfm_val = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Redimensiona las imágenes
        transforms.ToTensor(),  # Convierte la imagen a tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normaliza con valores predefinidos
    ])
    # Carga el conjunto de datos completo desde el directorio
    full_ds = datasets.ImageFolder(str(data_dir), transform=tfm_train)
    class_names = full_ds.classes  # Obtiene los nombres de las clases
    n_total = len(full_ds)  # Número total de imágenes
    n_val = int(n_total * val_split)  # Calcula el tamaño del conjunto de validación
    n_train = n_total - n_val  # Calcula el tamaño del conjunto de entrenamiento
    # Divide el conjunto de datos en entrenamiento y validación
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])
    val_ds.dataset.transform = tfm_val  # Aplica las transformaciones de validación al conjunto de validación
    if balanced:
        # Calcula la cantidad de imágenes por clase en el conjunto de entrenamiento
        counts = [0] * len(class_names)
        for _, idx in train_ds:
            counts[idx] += 1
        total = sum(counts)  # Total de imágenes
        # Calcula los pesos por clase para balancear el muestreo
        weights_per_class = [total / c for c in counts]
        # Asigna un peso a cada muestra en el conjunto de entrenamiento
        sample_weights = [weights_per_class[label] for _, label in train_ds]
        # Crea un sampler ponderado para el DataLoader
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        # Crea el DataLoader para el conjunto de entrenamiento con el sampler
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=2)
    else:
        # Crea el DataLoader para el conjunto de entrenamiento con barajado
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    # Crea el DataLoader para el conjunto de validación sin barajado
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, class_names  # Devuelve los DataLoaders y los nombres de las clases

def build_model(num_classes: int, arch: str = "efficientnet_b0", pretrained=True):
    # Construye un modelo basado en la arquitectura especificada
    if arch.startswith("efficientnet"):
        # Obtiene la función del modelo de EfficientNet
        model_fn = getattr(models, arch)
        # Carga el modelo preentrenado si corresponde
        model = model_fn(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained and arch == "efficientnet_b0" else None)
        in_features = model.classifier[1].in_features  # Obtiene el número de características de entrada
        model.classifier[1] = nn.Linear(in_features, num_classes)  # Reemplaza la capa final con una nueva para clasificación
    else:
        # Carga un modelo ResNet50 como alternativa
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_features = model.fc.in_features  # Obtiene el número de características de entrada
        model.fc = nn.Linear(in_features, num_classes)  # Reemplaza la capa final con una nueva para clasificación
    return model  # Devuelve el modelo

def eval_epoch(model, loader, criterion, device):
    # Evalúa el modelo en un conjunto de datos
    model.eval()  # Cambia el modelo al modo de evaluación
    running_loss = 0.0  # Acumulador para la pérdida
    preds = []  # Lista para almacenar las predicciones
    gts = []  # Lista para almacenar las etiquetas reales
    with torch.no_grad():  # Desactiva el cálculo de gradientes
        for imgs, labels in tqdm(loader, desc="val", leave=False):  # Itera sobre el DataLoader
            imgs, labels = imgs.to(device), labels.to(device)  # Mueve los datos al dispositivo (CPU/GPU)
            out = model(imgs)  # Realiza una pasada hacia adelante
            loss = criterion(out, labels)  # Calcula la pérdida
            running_loss += loss.item() * imgs.size(0)  # Acumula la pérdida
            pred = out.argmax(dim=1).cpu().numpy()  # Obtiene las predicciones
            preds.extend(pred.tolist())  # Agrega las predicciones a la lista
            gts.extend(labels.cpu().numpy().tolist())  # Agrega las etiquetas reales a la lista
    avg_loss = running_loss / len(loader.dataset)  # Calcula la pérdida promedio
    f1 = f1_score(gts, preds, average='macro')  # Calcula el F1-score
    return avg_loss, f1, gts, preds  # Devuelve la pérdida promedio, el F1-score, las etiquetas reales y las predicciones

def main():
    # Crear un analizador de argumentos para la línea de comandos
    parser = argparse.ArgumentParser()
    # Agregar argumentos para el directorio de datos, número de épocas, tamaño de batch, etc.
    parser.add_argument('--data_dir', type=str, required=True, help='Directorio raiz de imagenes (species/disease/img)')
    parser.add_argument('--epochs', type=int, default=10)  # Número de épocas de entrenamiento
    parser.add_argument('--batch_size', type=int, default=32)  # Tamaño de batch
    parser.add_argument('--lr', type=float, default=3e-4)  # Tasa de aprendizaje
    parser.add_argument('--arch', type=str, default='efficientnet_b0')  # Arquitectura del modelo
    parser.add_argument('--out_dir', type=str, default='outputs')  # Directorio de salida para guardar resultados
    parser.add_argument('--seed', type=int, default=42)  # Semilla para reproducibilidad
    parser.add_argument('--patience', type=int, default=5)  # Paciencia para early stopping
    parser.add_argument('--balanced', action='store_true')  # Bandera para usar muestreo balanceado
    parser.add_argument('--log_every', type=int, default=50, help='Frecuencia (batches) para imprimir progreso de entrenamiento')
    # Parsear los argumentos proporcionados por el usuario
    args = parser.parse_args()

    # Establecer la semilla para reproducibilidad
    seed_everything(args.seed)

    # Determinar el dispositivo a usar (GPU si está disponible, de lo contrario CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Convertir los directorios de datos y salida a objetos Path
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    # Crear el directorio de salida si no existe
    out_dir.mkdir(parents=True, exist_ok=True)

    # Construir los DataLoaders para entrenamiento y validación
    train_loader, val_loader, class_names = build_dataloaders(data_dir, balanced=args.balanced)
    # Construir el modelo con el número de clases y arquitectura especificados
    model = build_model(num_classes=len(class_names), arch=args.arch).to(device)

    # Definir la función de pérdida
    criterion = nn.CrossEntropyLoss()
    # Configurar el optimizador AdamW con la tasa de aprendizaje especificada
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Configurar el programador de tasa de aprendizaje con CosineAnnealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Inicializar variables para el mejor F1-score y el contador de paciencia
    best_f1 = 0
    patience_counter = 0
    # Lista para almacenar el historial de entrenamiento
    history = []

    # Bucle principal de entrenamiento por épocas
    for epoch in range(1, args.epochs+1):
        # Configurar el escalador para entrenamiento con precisión mixta
        scaler = torch.cuda.amp.GradScaler(enabled=device=='cuda')
        model.train()  # Cambiar el modelo al modo de entrenamiento
        running_loss = 0.0  # Inicializar la pérdida acumulada
        seen = 0  # Contador de muestras vistas
        total_batches = len(train_loader)  # Número total de batches
        # Crear una barra de progreso para el entrenamiento
        pbar = tqdm(enumerate(train_loader), total=total_batches, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for i, (imgs, labels) in pbar:
            # Mover las imágenes y etiquetas al dispositivo
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)  # Reiniciar los gradientes
            # Hacer una pasada hacia adelante con precisión mixta
            with torch.cuda.amp.autocast(enabled=device=='cuda'):
                out = model(imgs)  # Obtener las predicciones del modelo
                loss = criterion(out, labels)  # Calcular la pérdida
            # Escalar la pérdida y hacer la pasada hacia atrás
            scaler.scale(loss).backward()
            scaler.step(optimizer)  # Actualizar los pesos del modelo
            scaler.update()  # Actualizar el escalador
            batch_size = imgs.size(0)  # Tamaño del batch actual
            running_loss += loss.item() * batch_size  # Acumular la pérdida
            seen += batch_size  # Incrementar el contador de muestras vistas
            # Actualizar la barra de progreso cada cierto número de batches
            if (i+1) % args.log_every == 0 or (i+1)==total_batches:
                current_avg = running_loss / seen  # Calcular la pérdida promedio
                lr = optimizer.param_groups[0]['lr']  # Obtener la tasa de aprendizaje actual
                pbar.set_postfix({'loss': f"{current_avg:.4f}", 'lr': f"{lr:.2e}"})  # Mostrar métricas en la barra
        # Calcular la pérdida promedio de entrenamiento
        train_loss = running_loss / seen
        # Evaluar el modelo en el conjunto de validación
        val_loss, val_f1, gts, preds = eval_epoch(model, val_loader, criterion, device)
        # Imprimir métricas de la época actual
        print(f"[Epoch {epoch}/{args.epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f} lr={optimizer.param_groups[0]['lr']:.2e}")
        # Guardar las métricas en el historial
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'val_f1': val_f1})
        # Guardar el modelo si el F1-score de validación mejora
        if val_f1 > best_f1:
            best_f1 = val_f1  # Actualizar el mejor F1-score
            torch.save({'model_state': model.state_dict(), 'classes': class_names}, out_dir / 'best.pt')  # Guardar el modelo
            patience_counter = 0  # Reiniciar el contador de paciencia
        else:
            patience_counter += 1  # Incrementar el contador de paciencia
        scheduler.step()  # Actualizar la tasa de aprendizaje
        # Detener el entrenamiento si se alcanza la paciencia máxima
        if patience_counter >= args.patience:
            print('Early stopping por paciencia alcanzada.')
            break
    # Guardar el historial de entrenamiento en un archivo JSON
    with open(out_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    # Cambiar el modelo al modo de evaluación
    model.eval()
    # Crear un ejemplo de entrada para trazar el modelo
    example = torch.randn(1,3,224,224).to(device)
    traced = torch.jit.trace(model, example)  # Trazar el modelo
    traced.save(str(out_dir / 'model_ts.pt'))  # Guardar el modelo trazado
    print('Entrenamiento finalizado. Mejor F1:', best_f1)  # Imprimir el mejor F1-score

if __name__ == '__main__':
    main()
