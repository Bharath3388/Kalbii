import { useState, useEffect } from 'react';
import { FiActivity, FiRefreshCw } from 'react-icons/fi';
import { getRecords, RecordData } from '../api';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, BarChart, Bar, XAxis, YAxis, CartesianGrid } from 'recharts';

const COLORS: Record<string, string> = { LOW: '#10b981', MEDIUM: '#f59e0b', HIGH: '#ef4444' };

export default function HistoryPage() {
  const [records, setRecords] = useState<RecordData[]>([]);
  const [loading, setLoading] = useState(false);

  const load = async () => {
    setLoading(true);
    try {
      const { items } = await getRecords(100);
      setRecords(items);
    } catch (e: any) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  // Aggregated KPIs
  const total = records.length;
  const high = records.filter(r => r.risk_label === 'HIGH').length;
  const medium = records.filter(r => r.risk_label === 'MEDIUM').length;
  const low = records.filter(r => r.risk_label === 'LOW').length;
  const avgScore = total > 0 ? (records.reduce((s, r) => s + r.risk_score, 0) / total).toFixed(1) : '—';

  const pieData = [
    { name: 'LOW', value: low },
    { name: 'MEDIUM', value: medium },
    { name: 'HIGH', value: high },
  ].filter(d => d.value > 0);

  // Group scores in 10-bucket histogram
  const buckets = Array.from({ length: 10 }, (_, i) => ({
    range: `${i * 10}-${i * 10 + 9}`,
    count: records.filter(r => r.risk_score >= i * 10 && r.risk_score < (i + 1) * 10).length,
  }));

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 28 }}>
        <h1 className="page-heading">
          <FiActivity />
          Analysis History
        </h1>
        <button className="btn btn-outline" onClick={load} disabled={loading}>
          <FiRefreshCw className={loading ? 'spinning' : ''} /> Refresh
        </button>
      </div>

      {/* KPI cards */}
      <div className="kpi-row">
        <div className="kpi-card">
          <div className="kpi-label">Total Records</div>
          <div className="kpi-value">{total}</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">High Risk</div>
          <div className="kpi-value" style={{ color: COLORS.HIGH }}>{high}</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Medium Risk</div>
          <div className="kpi-value" style={{ color: COLORS.MEDIUM }}>{medium}</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Avg Score</div>
          <div className="kpi-value">{avgScore}</div>
        </div>
      </div>

      {/* Charts */}
      {total > 0 && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: 16, marginBottom: 24 }}>
          <div className="card">
            <div className="card-title">Risk Distribution</div>
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={pieData} cx="50%" cy="50%" innerRadius={45} outerRadius={75} dataKey="value" label={({ name, value }) => `${name}: ${value}`}>
                  {pieData.map((d) => <Cell key={d.name} fill={COLORS[d.name]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="card">
            <div className="card-title">Score Distribution</div>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={buckets}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2a35" />
                <XAxis dataKey="range" tick={{ fill: '#8888a0', fontSize: 11 }} />
                <YAxis tick={{ fill: '#8888a0', fontSize: 11 }} />
                <Tooltip contentStyle={{ background: '#131316', border: '1px solid #2a2a35', borderRadius: 8 }} />
                <Bar dataKey="count" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Records table */}
      <div className="card">
        <div className="card-title">Records</div>
        {total === 0 ? (
          <div style={{ padding: 32, textAlign: 'center', color: 'var(--text2)' }}>
            No records yet. Run an analysis first.
          </div>
        ) : (
          <table className="records-table">
            <thead>
              <tr>
                <th>Job ID</th>
                <th>Risk</th>
                <th>Score</th>
                <th>Sentiment</th>
                <th>CV Score</th>
                <th>CV Backend</th>
                <th>Date</th>
              </tr>
            </thead>
            <tbody>
              {records.map(r => (
                <tr key={r.job_id}>
                  <td style={{ fontFamily: 'monospace', fontSize: 12 }}>{r.job_id.slice(0, 12)}…</td>
                  <td><span className={`risk-badge ${r.risk_label}`}>{r.risk_label}</span></td>
                  <td style={{ fontWeight: 600 }}>{r.risk_score}</td>
                  <td>{r.nlp?.sentiment?.label || '—'}</td>
                  <td>{r.cv?.anomaly_score != null ? (r.cv.anomaly_score * 100).toFixed(1) + '%' : '—'}</td>
                  <td><span className="tag">{r.cv?.method || '—'}</span></td>
                  <td style={{ fontSize: 12, color: 'var(--text2)' }}>{r.created_at ? new Date(r.created_at).toLocaleString() : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
