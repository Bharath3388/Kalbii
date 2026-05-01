import { useState, useCallback } from 'react';
import { FiUploadCloud, FiSend, FiEye, FiAlertTriangle, FiCpu, FiFileText, FiShield } from 'react-icons/fi';
import { encrypt, ingest, getResult, RecordData } from '../api';
import ResultPanel from '../components/ResultPanel';

const CV_OPTIONS = [
  { id: 'opencv', title: 'OpenCV', desc: 'Statistical (Laplacian, edge, color)' },
  { id: 'autoencoder', title: 'Autoencoder', desc: 'PyTorch Conv-AE (advanced)' },
  { id: 'clip', title: 'CLIP', desc: 'Zero-shot VIT (bonus)' },
] as const;

export default function AnalyzePage() {
  const [text, setText] = useState('');
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [cvBackend, setCvBackend] = useState('opencv');
  const [loading, setLoading] = useState(false);
  const [step, setStep] = useState('');
  const [result, setResult] = useState<RecordData | null>(null);
  const [error, setError] = useState('');

  const handleImage = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setImageFile(f);
    const reader = new FileReader();
    reader.onload = () => setImagePreview(reader.result as string);
    reader.readAsDataURL(f);
  }, []);

  const fileToBase64 = (file: File): Promise<string> =>
    new Promise((resolve, reject) => {
      const r = new FileReader();
      r.onload = () => {
        const s = r.result as string;
        resolve(s.split(',')[1]);
      };
      r.onerror = reject;
      r.readAsDataURL(file);
    });

  const handleSubmit = async () => {
    if (!text.trim()) { setError('Please enter some text.'); return; }
    setError('');
    setResult(null);
    setLoading(true);

    try {
      // Step 1: encrypt
      setStep('Encrypting data...');
      const imgB64 = imageFile ? await fileToBase64(imageFile) : null;
      const enc = await encrypt(text, imgB64);

      // Step 2: ingest (decrypt + NLP + CV + Risk)
      setStep('Processing pipeline (decrypt → NLP → CV → Risk)...');
      const ing = await ingest(enc.ciphertext_text, enc.ciphertext_image, cvBackend, {
        source: 'react-frontend',
        cv_backend_selected: cvBackend,
      });

      // Step 3: fetch result
      if (ing.status === 'done') {
        setStep('Fetching result...');
        const rec = await getResult(ing.job_id);
        setResult(rec);
      } else {
        // Poll if queued (Celery mode)
        setStep('Queued... polling for result...');
        let attempts = 0;
        while (attempts < 30) {
          await new Promise(r => setTimeout(r, 2000));
          try {
            const rec = await getResult(ing.job_id);
            if (rec.status === 'done') { setResult(rec); break; }
          } catch { /* not ready yet */ }
          attempts++;
        }
        if (!result && attempts >= 30) setError('Timeout waiting for result.');
      }
    } catch (e: any) {
      setError(e.message || 'Something went wrong.');
    } finally {
      setLoading(false);
      setStep('');
    }
  };

  return (
    <div>
      <h1 style={{ fontSize: 24, fontWeight: 700, marginBottom: 4 }}>
        <FiShield style={{ marginRight: 8, verticalAlign: -2 }} />
        Risk Analysis
      </h1>
      <p style={{ color: 'var(--text2)', marginBottom: 24, fontSize: 14 }}>
        Submit text and/or an image. Data is encrypted, processed through NLP + CV pipelines, and scored by the ML model.
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
        {/* Left: inputs */}
        <div className="card">
          <div className="card-title"><FiFileText style={{ marginRight: 6 }} /> Input</div>

          <div className="form-group">
            <label>Text to analyze</label>
            <textarea
              placeholder="e.g. URGENT: crack detected near reactor weld — potential gas leak"
              value={text}
              onChange={e => setText(e.target.value)}
            />
          </div>

          <div className="form-group">
            <label>Image (optional)</label>
            <div className="file-upload">
              <input type="file" accept="image/*" onChange={handleImage} />
              {imagePreview ? (
                <img src={imagePreview} alt="preview" className="file-upload-preview" />
              ) : (
                <>
                  <div className="file-upload-icon"><FiUploadCloud /></div>
                  <div className="file-upload-text">Click or drag to upload an image</div>
                </>
              )}
            </div>
          </div>

          <div className="form-group">
            <label><FiCpu style={{ marginRight: 4 }} /> CV Backend</label>
            <div className="cv-selector">
              {CV_OPTIONS.map(o => (
                <div
                  key={o.id}
                  className={`cv-option ${cvBackend === o.id ? 'selected' : ''}`}
                  onClick={() => setCvBackend(o.id)}
                >
                  <div className="cv-option-title">{o.title}</div>
                  <div className="cv-option-desc">{o.desc}</div>
                </div>
              ))}
            </div>
          </div>

          <button className="btn btn-primary" onClick={handleSubmit} disabled={loading}>
            {loading ? <><span className="spinner" /> {step}</> : <><FiSend /> Analyze</>}
          </button>

          {error && (
            <div style={{ color: 'var(--red)', marginTop: 12, fontSize: 13 }}>
              <FiAlertTriangle style={{ marginRight: 4 }} /> {error}
            </div>
          )}
        </div>

        {/* Right: result */}
        <div>
          {result ? (
            <ResultPanel data={result} />
          ) : (
            <div className="card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 400, color: 'var(--text2)' }}>
              <div style={{ textAlign: 'center' }}>
                <FiEye style={{ fontSize: 48, marginBottom: 12 }} />
                <div>Results will appear here</div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
