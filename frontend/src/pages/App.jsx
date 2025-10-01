import React, { useState } from 'react';
import axios from 'axios';

export default function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [preds, setPreds] = useState(null);
  const [error, setError] = useState(null);

  const handleUpload = async () => {
    if(!file) return;
    setLoading(true); setError(null); setPreds(null);
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await axios.post('http://localhost:8000/predict', formData, { headers: { 'Content-Type': 'multipart/form-data' }});
      setPreds(res.data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen px-4 py-8 max-w-3xl mx-auto">
      <h1 className="text-3xl font-bold text-brand-600 mb-4">PlantAid</h1>
      <p className="text-slate-600 mb-6">Sube una foto de la hoja o planta para detectar especie y enfermedad.</p>
      <div className="space-y-4">
        <input type="file" accept="image/*" onChange={e => setFile(e.target.files[0])} />
        <button onClick={handleUpload} disabled={!file || loading} className="bg-brand-600 text-white px-4 py-2 rounded disabled:opacity-40">
          {loading ? 'Analizando...' : 'Detectar'}
        </button>
      </div>
      {error && <div className="mt-4 text-red-600">Error: {error}</div>}
      {preds && (
        <div className="mt-8 space-y-4">
          <h2 className="text-xl font-semibold">Resultados</h2>
          {preds.predictions.map((p,i)=>(
            <div key={i} className="border rounded p-4 bg-white shadow-sm">
              <div className="flex justify-between">
                <span className="font-medium capitalize">{p.species}</span>
                <span className="text-sm text-slate-500">{(p.confidence*100).toFixed(1)}%</span>
              </div>
              <div className="mt-1 capitalize">{p.healthy ? 'Saludable' : p.disease.replace(/_/g,' ')}</div>
              {p.treatment && <div className="mt-2 text-sm text-slate-600">Tratamiento sugerido: {p.treatment}</div>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
