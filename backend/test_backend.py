from fastapi.testclient import TestClient
from .main import app
from pathlib import Path
from PIL import Image
import io

client = TestClient(app)

def test_health():
    r = client.get('/health')
    assert r.status_code == 200


def test_predict_dummy():
    # crear imagen en memoria
    img = Image.new('RGB', (256,256), color=(100,180,90))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    files = {'file': ('test.png', buf, 'image/png')}
    r = client.post('/predict', files=files)
    assert r.status_code == 200
    data = r.json()
    assert 'predictions' in data
