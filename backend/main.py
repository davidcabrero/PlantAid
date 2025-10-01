from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image
import io
import torch
import torchvision.transforms as T
import time
import json
import os
from pathlib import Path
from .model_loader import InferenceModel

app = FastAPI(title="PlantAid API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DiseasePrediction(BaseModel):
    species: str
    disease: str
    confidence: float
    healthy: bool
    treatment: Optional[str] = None

class PredictResponse(BaseModel):
    predictions: List[DiseasePrediction]
    model_version: str
    elapsed_ms: float

# Placeholder label maps
SPECIES = ["tomato", "apple", "cassava", "grape"]
DISEASES = ["healthy", "leaf_blight", "leaf_spot", "mosaic", "rust"]

# Intentar cargar modelo real si existe
env_model = os.getenv('PLANTAID_MODEL_PATH')
MODEL_PATH = Path(env_model) if env_model else (Path("model") / "model_ts.pt")
try:
    if MODEL_PATH.exists():
        model_wrapper = InferenceModel(str(MODEL_PATH))
        model = model_wrapper.model
        loaded_real = True
    else:
        raise FileNotFoundError
except Exception:
    class DummyModel:
        def __call__(self, x):
            batch = x.shape[0]
            import torch
            return torch.randn(batch, len(SPECIES)*len(DISEASES))
    model = DummyModel()
    loaded_real = False

# Cargar información de enfermedades
DISEASE_INFO = {}
info_path = Path(__file__).parent / 'disease_info.json'
if info_path.exists():
    with open(info_path, 'r', encoding='utf-8') as f:
        DISEASE_INFO = json.load(f)
transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def _split_class_name(name: str):
    # acepta formatos 'tomato/early_blight', 'tomato\\early_blight', 'tomato__early_blight'
    if '__' in name:
        parts = name.split('__', 1)
    elif '/' in name:
        parts = name.split('/', 1)
    elif '\\' in name:
        parts = name.split('\\', 1)
    else:
        parts = [name, 'unknown']
    if len(parts) == 1:
        parts.append('unknown')
    return parts[0], parts[1]

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    start = time.time()
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        raw_out = model(x)
    import torch.nn.functional as F
    preds = []
    if loaded_real and hasattr(model, 'forward') and 'model_wrapper' in globals() and model_wrapper.classes:
        probs = F.softmax(raw_out, dim=1)[0]
        species_best = {}
        for idx, cname in enumerate(model_wrapper.classes):
            sp, dis = _split_class_name(cname)
            p = probs[idx].item()
            if sp not in species_best or p > species_best[sp]['p']:
                species_best[sp] = {'disease': dis, 'p': p}
        for sp, info in species_best.items():
            dis = info['disease']
            healthy = dis in ('healthy','normal')
            treat = None
            if not healthy:
                norm_dis = dis.replace('leaf_', '').replace('___','_')
                if sp in DISEASE_INFO and norm_dis in DISEASE_INFO[sp]:
                    treat = DISEASE_INFO[sp][norm_dis].get('tratamiento')
            preds.append(DiseasePrediction(
                species=sp,
                disease=dis,
                confidence=info['p'],
                healthy=healthy,
                treatment=None if healthy else (treat or f"Tratamiento base para {dis}")))
        preds = sorted(preds, key=lambda x: x.confidence, reverse=True)[:5]
    else:
        # placeholder multi-head dummy
        try:
            expected = len(SPECIES) * len(DISEASES)
            # Caso 1: coincide con multi-head plano
            if raw_out.ndim == 2 and raw_out.shape[1] == expected:
                out = raw_out.view(1, len(SPECIES), len(DISEASES))
                probs = F.softmax(out, dim=-1)
                for i, sp in enumerate(SPECIES):
                    sp_probs = probs[0, i]
                    conf, idx = torch.max(sp_probs, dim=0)
                    disease = DISEASES[idx]
                    if disease != "healthy":
                        treat = None
                        norm_dis = disease.replace('leaf_', '').replace('___','_')
                        if sp in DISEASE_INFO and norm_dis in DISEASE_INFO[sp]:
                            treat = DISEASE_INFO[sp][norm_dis].get('tratamiento')
                        preds.append(DiseasePrediction(
                            species=sp,
                            disease=disease,
                            confidence=conf.item(),
                            healthy=False,
                            treatment=treat or f"Tratamiento base para {disease}"))
            else:
                # Caso 2: salida single-head con N clases combinadas desconocidas (ej: 13)
                combined_classes = [f"{sp}__{dis}" for sp in SPECIES for dis in DISEASES]
                probs = F.softmax(raw_out, dim=1)[0]
                topk = min(5, probs.shape[0])
                values, indices = torch.topk(probs, k=topk)
                for v, idx in zip(values.tolist(), indices.tolist()):
                    if idx < len(combined_classes):
                        cname = combined_classes[idx]
                        sp, dis = cname.split('__', 1)
                    else:
                        # índice desconocido; asignar placeholder
                        sp, dis = 'unknown', f'class_{idx}'
                    healthy = dis in ('healthy','normal')
                    treat = None
                    if not healthy:
                        norm_dis = dis.replace('leaf_', '').replace('___','_')
                        if sp in DISEASE_INFO and norm_dis in DISEASE_INFO[sp]:
                            treat = DISEASE_INFO[sp][norm_dis].get('tratamiento')
                    preds.append(DiseasePrediction(
                        species=sp,
                        disease=dis,
                        confidence=v,
                        healthy=healthy,
                        treatment=None if healthy else (treat or f"Tratamiento base para {dis}")))
            if not preds:
                preds.append(DiseasePrediction(species="unknown", disease="healthy", confidence=1.0, healthy=True))
        except Exception as e:
            # fallback de emergencia
            preds = [DiseasePrediction(species="unknown", disease="healthy", confidence=1.0, healthy=True, treatment=None)]
    elapsed_ms = (time.time()-start)*1000
    return PredictResponse(predictions=preds, model_version="dummy-0.1", elapsed_ms=elapsed_ms)

@app.get("/health")
async def health():
    return {"status": "ok"}
