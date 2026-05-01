const API_BASE = import.meta.env.VITE_API_URL || '';
const API_KEY = import.meta.env.VITE_API_KEY || 'demo-key';

const headers = (): Record<string, string> => ({
  'Content-Type': 'application/json',
  'X-API-Key': API_KEY,
});

export interface EncryptResponse {
  ciphertext_text: string;
  ciphertext_image: string | null;
}

export interface IngestResponse {
  job_id: string;
  status: string;
  status_url: string;
}

export interface NlpData {
  sentiment: { label: string; score: number };
  entities: { text: string; label: string }[];
  keywords: string[];
  risk_terms: { term: string; category: string }[];
  n_risk_terms_critical: number;
  n_risk_terms_high: number;
  text_length: number;
  text_risk_sub: number;
}

export interface CvData {
  anomaly_score: number;
  method: string;
  image_path: string | null;
  heatmap_path: string | null;
  edge_density: number;
  hotpix_ratio: number;
  error?: string;
}

export interface RecordData {
  job_id: string;
  status: string;
  decrypted_text: string;
  image_path: string | null;
  heatmap_path: string | null;
  nlp: NlpData;
  cv: CvData;
  risk_score: number;
  risk_label: 'LOW' | 'MEDIUM' | 'HIGH';
  created_at: string;
  metadata?: Record<string, unknown>;
}

export interface HealthData {
  status: string;
  mongo: boolean;
  celery: boolean;
  cv_backend: string;
  nlp_backend: string;
}

export async function encrypt(text: string, image_b64: string | null): Promise<EncryptResponse> {
  const res = await fetch(`${API_BASE}/encrypt`, {
    method: 'POST',
    headers: headers(),
    body: JSON.stringify({ text, image_b64 }),
  });
  if (!res.ok) throw new Error(`Encrypt failed: ${res.status}`);
  return res.json();
}

export async function ingest(
  ciphertext_text: string,
  ciphertext_image: string | null,
  cv_backend: string,
  metadata?: Record<string, unknown>,
): Promise<IngestResponse> {
  const res = await fetch(`${API_BASE}/ingest`, {
    method: 'POST',
    headers: headers(),
    body: JSON.stringify({ ciphertext_text, ciphertext_image, cv_backend, metadata }),
  });
  if (!res.ok) throw new Error(`Ingest failed: ${res.status}`);
  return res.json();
}

export async function getResult(job_id: string): Promise<RecordData> {
  const res = await fetch(`${API_BASE}/result/${job_id}`, { headers: headers() });
  if (!res.ok) throw new Error(`Result failed: ${res.status}`);
  return res.json();
}

export async function getRecords(limit = 50): Promise<{ count: number; items: RecordData[] }> {
  const res = await fetch(`${API_BASE}/records?limit=${limit}`, { headers: headers() });
  if (!res.ok) throw new Error(`Records failed: ${res.status}`);
  return res.json();
}

export async function getHealth(): Promise<HealthData> {
  const res = await fetch(`${API_BASE}/healthz`);
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}
