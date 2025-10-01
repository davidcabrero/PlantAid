# PlantAid

Aplicación para detectar enfermedades de plantas a partir de fotografías con un modelo de ML entrenado.

## Componentes
- `ml/`: Scripts de entrenamiento del modelo, evaluación y normalizar datasets.
- `backend/`: API (FastAPI) para inferencia, gestión de metadatos y tratamiento.
- `frontend/`: Interfaz de la aplicación (React + Vite + Tailwind).

## Flujo
1. Entrenar modelo con transferencia usando datasets combinados normalizados de enfermedades.
2. Exportar a formato TorchScript u ONNX para inferencia rápida.
3. Servir modelo con FastAPI (endpoint `/predict`).
4. La aplicación permite subir una imagen y muestra especie, prob. enfermedad, diagnóstico, tratamientos sugeridos y links.

## Instrucciones detalladas

### 1. Preparar entorno (Windows PowerShell)
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r ml/requirements.txt
pip install -r backend/requirements.txt
cd frontend; npm install; cd ..
```

### 2. Descargar datasets

- PlantVillage (https://www.kaggle.com/datasets/emmarex/plantdisease)

- PlantDoc (https://github.com/pratikkayal/PlantDoc-Dataset)

### 3. Unificar clases
Coloca todo el contenido crudo en `ml/data/raw` y ejecuta:
```powershell
python ml/class_normalizer.py --src ml/data/raw --dst ml/data/clean --min_per_class 30
```
Estructura resultante esperada: `ml/data/clean/<species>/<disease>/imagen.jpg`.

### 4. Entrenamiento (baseline)
```powershell
python ml/train.py --data_dir ml/data/clean --epochs 12 --balanced --out_dir ml/outputs_baseline
```
Salida clave: `best.pt` (pesos) y `model_ts.pt` (TorchScript).

### 5. Entrenamiento avanzado (mayor precisión)
```powershell
python ml/train_timm.py --data_dir ml/data/clean --model efficientnetv2_rw_s --img_size 384 --epochs 40 --balanced --mixup 0.4 --cutmix 1.0 --smoothing 0.1 --out_dir ml/outputs_adv --export_onnx
```
Resultados: `best_adv.pt`, `model_ts.pt`, `model.onnx` (opcional).

### 6. Evaluación adicional
Crear `classes.json` con el array de clases en el mismo orden que entrenamiento (ya está incluido dentro de `best*.pt`; exporta manualmente si lo necesitas):
```powershell
# ejemplo rápido (desde Python interactivo):
python - <<'PY'
import torch, json
ck=torch.load('ml/outputs_adv/best_adv.pt', map_location='cpu')
json.dump(ck['classes'], open('ml/outputs_adv/classes.json','w'))
PY
```
Evaluar TorchScript:
```powershell
python ml/evaluate.py --ts_model ml/outputs_adv/model_ts.pt --classes_json ml/outputs_adv/classes.json --test_dir ml/data/clean --out_dir ml/eval
```

### 7. Servir backend
Copiar el modelo a `backend/model/model_ts.pt` o exportar variable:
```powershell
mkdir backend\model
copy ml\outputs_adv\model_ts.pt backend\model\model_ts.pt
setx PLANTAID_MODEL_PATH "backend\model\model_ts.pt"
uvicorn backend.main:app --reload --port 8000
```

### 8. Ejecutar frontend
```powershell
cd frontend
npm run dev
```
Abrir http://localhost:5173 y probar subida de imagen.

### 9. Métricas objetivo sugeridas
- Macro F1 >= 0.90 en especies y >= 0.85 en enfermedades minoritarias.
- Balanced accuracy >= 0.88.
- Latencia inferencia (CPU) < 200 ms / imagen (224px) o < 350 ms (384px).

## Endpoints (backend)
- `POST /predict` -> imagen -> especies y enfermedades principales.
- `GET /health` -> estado.

## Variables de entorno
- `PLANTAID_MODEL_PATH`: ruta al `model_ts.pt` a cargar.

## Tratamientos
Definidos en `backend/disease_info.json`; ampliar con nuevas enfermedades tras normalizar mapeos.

## Requerimientos iniciales
Ver `ml/requirements.txt` y `backend/requirements.txt`.
