"""Normaliza nombres de clases combinando múltiples datasets en formato species/disease.

Entrada esperada: directorios originales variados.
Salida: carpeta destino con estructura: root/species/disease/img.jpg

Estrategia:
1. Mapear nombres crudos a (species, disease_clean)
2. Unificar sinónimos (ej: early_blight == alternaria_early_blight)
3. Opción para filtrar clases con < N imágenes.
"""
from pathlib import Path
import shutil
import re
import argparse
from collections import defaultdict

# Reglas ampliadas para PlantVillage y PlantDoc
CLASS_MAP = {
    'Tomato___Early_blight': ('tomato', 'early_blight'),
    'Tomato___Late_blight': ('tomato', 'late_blight'),
    'Tomato___Leaf_Mold': ('tomato', 'leaf_mold'),
    'Tomato___Bacterial_spot': ('tomato', 'bacterial_spot'),
    'Tomato___healthy': ('tomato', 'healthy'),
    'Tomato_healthy': ('tomato', 'healthy'),
    'Tomato_Early_blight': ('tomato', 'early_blight'),
    'Tomato_Late_blight': ('tomato', 'late_blight'),
    'Tomato_Leaf_Mold': ('tomato', 'leaf_mold'),
    'Tomato_Bacterial_spot': ('tomato', 'bacterial_spot'),
    'Tomato_Septoria_leaf_spot': ('tomato', 'septoria_leaf_spot'),
    'Tomato_Spider_mites_Two_spotted_spider_mite': ('tomato', 'spider_mites'),
    'Tomato__Target_Spot': ('tomato', 'target_spot'),
    'Tomato__Tomato_mosaic_virus': ('tomato', 'mosaic_virus'),
    'Tomato__Tomato_YellowLeaf__Curl_Virus': ('tomato', 'yellow_leaf_curl_virus'),
    'Potato___Early_blight': ('potato', 'early_blight'),
    'Potato___Late_blight': ('potato', 'late_blight'),
    'Potato___healthy': ('potato', 'healthy'),
    'Pepper__bell___Bacterial_spot': ('bell_pepper', 'bacterial_spot'),
    'Pepper__bell___healthy': ('bell_pepper', 'healthy'),
    'Grape___Black_rot': ('grape', 'black_rot'),
    'Grape___healthy': ('grape', 'healthy'),
    'Cassava___Bacterial_Blight': ('cassava', 'bacterial_blight'),
    'Cassava___Mosaic_Disease': ('cassava', 'mosaic_virus'),
    'Cassava___Healthy': ('cassava', 'healthy'),
    'Apple leaf': ('apple', 'healthy'),
    'Apple rust leaf': ('apple', 'rust'),
    'Apple Scab Leaf': ('apple', 'scab'),
    'Bell_pepper leaf': ('bell_pepper', 'healthy'),
    'Bell_pepper leaf spot': ('bell_pepper', 'leaf_spot'),
    'Blueberry leaf': ('blueberry', 'healthy'),
    'Cherry leaf': ('cherry', 'healthy'),
    'Corn Gray leaf spot': ('corn', 'gray_leaf_spot'),
    'Corn leaf blight': ('corn', 'leaf_blight'),
    'Corn rust leaf': ('corn', 'rust'),
    'grape leaf': ('grape', 'healthy'),
    'grape leaf black rot': ('grape', 'black_rot'),
    'Peach leaf': ('peach', 'healthy'),
    'Potato leaf early blight': ('potato', 'early_blight'),
    'Potato leaf late blight': ('potato', 'late_blight'),
    'Raspberry leaf': ('raspberry', 'healthy'),
    'Soyabean leaf': ('soybean', 'healthy'),
    'Squash Powdery mildew leaf': ('squash', 'powdery_mildew'),
    'Strawberry leaf': ('strawberry', 'healthy'),
    'Tomato Early blight leaf': ('tomato', 'early_blight'),
    'Tomato leaf': ('tomato', 'healthy'),
    'Tomato leaf bacterial spot': ('tomato', 'bacterial_spot'),
    'Tomato leaf late blight': ('tomato', 'late_blight'),
    'Tomato leaf mosaic virus': ('tomato', 'mosaic_virus'),
    'Tomato leaf yellow virus': ('tomato', 'yellow_leaf_curl_virus'),
    'Tomato mold leaf': ('tomato', 'leaf_mold'),
    'Tomato Septoria leaf spot': ('tomato', 'septoria_leaf_spot'),
    'Tomato two spotted spider mites leaf': ('tomato', 'spider_mites'),
}

ALIAS = {
    'early_blight': 'early_blight',
    'alternaria': 'early_blight',
    'late_blight': 'late_blight',
    'phytophthora': 'late_blight',
    'septoria_leaf_spot': 'septoria_leaf_spot',
    'spider_mites': 'spider_mites',
    'target_spot': 'target_spot',
    'mosaic_virus': 'mosaic_virus',
    'yellow_leaf_curl_virus': 'yellow_leaf_curl_virus',
    'bacterial_spot': 'bacterial_spot',
    'leaf_spot': 'leaf_spot',
    'leaf_blight': 'leaf_blight',
    'powdery_mildew': 'powdery_mildew',
    'gray_leaf_spot': 'gray_leaf_spot',
    'black_rot': 'black_rot',
    'leaf_mold': 'leaf_mold',
    'rust': 'rust',
    'scab': 'scab',
    'bacterial_blight': 'bacterial_blight'
}

def normalize(label: str):
    if label in CLASS_MAP:
        sp, dis = CLASS_MAP[label]
        return sp, ALIAS.get(dis, dis)
    norm = label.replace('__', '_').replace('___', '_').replace('-', '_').strip()
    if norm in CLASS_MAP:
        sp, dis = CLASS_MAP[norm]
        return sp, ALIAS.get(dis, dis)
    return None

def process(src: Path, dst: Path, min_per_class: int = 20):
    counts = defaultdict(int)
    unknown = defaultdict(int)
    for img in src.rglob('*'):
        if img.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            continue
        class_name = img.parent.name
        norm = normalize(class_name)
        if not norm:
            unknown[class_name] += 1
            continue
        sp, dis = norm
        out_dir = dst / sp / dis
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / img.name
        if not target.exists():
            shutil.copy2(img, target)
            counts[(sp, dis)] += 1
    removed = []
    for (sp, dis), c in list(counts.items()):
        if c < min_per_class:
            folder = dst / sp / dis
            for f in folder.glob('*'):
                f.unlink()
            folder.rmdir()
            removed.append((sp, dis, c))
            del counts[(sp, dis)]
    print('\nResumen clases finales (>= min_per_class):')
    for (sp, dis), c in sorted(counts.items()):
        print(f'  {sp}/{dis}: {c}')
    if removed:
        print('\nClases eliminadas por pocas imágenes:')
        for sp, dis, c in removed:
            print(f'  {sp}/{dis}: {c}')
    if unknown:
        print('\nClases desconocidas (no mapeadas) encontradas:')
        for k, v in sorted(unknown.items(), key=lambda x: -x[1])[:30]:
            print(f'  {k}: {v}')
        print('-> Añade las que necesites a CLASS_MAP.')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--min_per_class', type=int, default=20)
    args = ap.parse_args()
    process(Path(args.src), Path(args.dst), args.min_per_class)
