import torch
from pathlib import Path
from typing import List

class InferenceModel:
    def __init__(self, model_path: str, device: str='cpu'):
        self.device = device
        mp = Path(model_path)
        if mp.suffix == '.pt':
            self.model = torch.jit.load(str(mp), map_location=device)
        else:
            raise ValueError('Formato de modelo no soportado aún')
        self.model.eval()
        meta_file = mp.with_name('best.pt')
        if meta_file.exists():
            meta = torch.load(meta_file, map_location='cpu')
            self.classes: List[str] = meta['classes']
        else:
            self.classes = []

    @torch.inference_mode()
    def predict_logits(self, tensor):
        return self.model(tensor)
