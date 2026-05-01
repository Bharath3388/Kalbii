import { RecordData } from '../api';
import { FiShield, FiFileText, FiImage, FiTag, FiAlertCircle } from 'react-icons/fi';

function ScoreBar({ value, max = 1, color }: { value: number; max?: number; color: string }) {
  const pct = Math.min(100, (value / max) * 100);
  return (
    <div className="progress-bar">
      <div className="progress-bar-fill" style={{ width: `${pct}%`, background: color }} />
    </div>
  );
}

export default function ResultPanel({ data }: { data: RecordData }) {
  const { nlp, cv, risk_score, risk_label, decrypted_text } = data;

  const riskColor = risk_label === 'HIGH' ? 'var(--red)' : risk_label === 'MEDIUM' ? 'var(--yellow)' : 'var(--green)';

  return (
    <div className="result-panel">
      {/* Risk Score Hero */}
      <div className="card" style={{ textAlign: 'center', borderColor: riskColor, borderWidth: 2 }}>
        <div className="card-title"><FiShield style={{ marginRight: 6 }} /> Unified Risk Score</div>
        <div className={`risk-score-big ${risk_label}`}>{risk_score}</div>
        <span className={`risk-badge ${risk_label}`}>{risk_label} RISK</span>
        <div style={{ marginTop: 8, fontSize: 12, color: 'var(--text2)' }}>
          Model: {data.metadata?.model_backend as string || 'xgboost'} | CV: {cv.method}
        </div>
      </div>

      {/* Decrypted text */}
      <div className="card" style={{ marginTop: 16 }}>
        <div className="card-title"><FiFileText style={{ marginRight: 6 }} /> Decrypted Text</div>
        <div className="decrypted-box">{decrypted_text}</div>
      </div>

      <div className="detail-grid">
        {/* NLP Analysis */}
        <div className="card">
          <div className="card-title">NLP Analysis</div>

          <div className="section-label">Sentiment</div>
          <div style={{ marginBottom: 12 }}>
            <span className={`risk-badge ${nlp.sentiment.label === 'NEGATIVE' ? 'HIGH' : nlp.sentiment.label === 'POSITIVE' ? 'LOW' : 'MEDIUM'}`}>
              {nlp.sentiment.label}
            </span>
            <span style={{ marginLeft: 8, fontSize: 13, color: 'var(--text2)' }}>
              confidence: {(nlp.sentiment.score * 100).toFixed(1)}%
            </span>
          </div>

          <div className="section-label">Text Risk Sub-score</div>
          <div style={{ fontSize: 20, fontWeight: 700, color: riskColor }}>{(nlp.text_risk_sub * 100).toFixed(1)}%</div>
          <ScoreBar value={nlp.text_risk_sub} color={riskColor} />

          <div className="section-label" style={{ marginTop: 12 }}>Risk Terms</div>
          <div>
            {nlp.risk_terms.length === 0 && <span style={{ fontSize: 13, color: 'var(--text2)' }}>None detected</span>}
            {nlp.risk_terms.map((t, i) => (
              <span key={i} className={`tag ${t.category}`}>
                <FiAlertCircle style={{ fontSize: 10, marginRight: 3 }} />
                {t.term} ({t.category})
              </span>
            ))}
          </div>

          <div className="section-label" style={{ marginTop: 12 }}>Keywords</div>
          <div>
            {nlp.keywords.map((k, i) => <span key={i} className="tag">{k}</span>)}
          </div>

          {nlp.entities.length > 0 && (
            <>
              <div className="section-label" style={{ marginTop: 12 }}>Entities</div>
              <div>
                {nlp.entities.map((e, i) => (
                  <span key={i} className="tag"><FiTag style={{ fontSize: 10, marginRight: 3 }} />{e.text} ({e.label})</span>
                ))}
              </div>
            </>
          )}
        </div>

        {/* CV Analysis */}
        <div className="card">
          <div className="card-title"><FiImage style={{ marginRight: 6 }} /> CV Analysis</div>

          {cv.error ? (
            <div style={{ color: 'var(--text2)', fontSize: 13 }}>No image provided — CV skipped</div>
          ) : (
            <>
              <div className="section-label">Anomaly Score</div>
              <div style={{ fontSize: 20, fontWeight: 700, color: cv.anomaly_score > 0.5 ? 'var(--red)' : cv.anomaly_score > 0.25 ? 'var(--yellow)' : 'var(--green)' }}>
                {(cv.anomaly_score * 100).toFixed(1)}%
              </div>
              <ScoreBar value={cv.anomaly_score} color={cv.anomaly_score > 0.5 ? 'var(--red)' : cv.anomaly_score > 0.25 ? 'var(--yellow)' : 'var(--green)'} />

              <div className="section-label" style={{ marginTop: 12 }}>Method</div>
              <span className="tag">{cv.method}</span>

              <div className="section-label" style={{ marginTop: 12 }}>Features</div>
              <div style={{ fontSize: 13, color: 'var(--text2)' }}>
                Edge density: {cv.edge_density?.toFixed(4)}<br />
                Hot-pixel ratio: {cv.hotpix_ratio?.toFixed(4)}
              </div>
            </>
          )}
        </div>
      </div>

      {/* Meta */}
      <div className="card" style={{ marginTop: 16, fontSize: 12, color: 'var(--text2)' }}>
        <strong>Job ID:</strong> {data.job_id} &nbsp; | &nbsp;
        <strong>Processed:</strong> {data.created_at} &nbsp; | &nbsp;
        <strong>Text length:</strong> {nlp.text_length} chars
      </div>
    </div>
  );
}
