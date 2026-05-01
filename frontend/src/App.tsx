import { Routes, Route, NavLink } from 'react-router-dom';
import { FiShield, FiActivity, FiSearch, FiServer } from 'react-icons/fi';
import AnalyzePage from './pages/AnalyzePage';
import HistoryPage from './pages/HistoryPage';

export default function App() {
  return (
    <div className="app-layout">
      <nav className="sidebar">
        <div className="sidebar-logo">
          <span>🛡️</span> Kalbii
        </div>
        <NavLink to="/" end className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
          <FiSearch /> Analyze
        </NavLink>
        <NavLink to="/history" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
          <FiActivity /> History
        </NavLink>
        <div style={{ flex: 1 }} />
        <div className="nav-item" style={{ cursor: 'default', fontSize: 11, color: 'var(--text2)' }}>
          <FiServer /> <span>Encrypted Multi-Modal Intelligence</span>
        </div>
      </nav>
      <main className="main-content">
        <Routes>
          <Route path="/" element={<AnalyzePage />} />
          <Route path="/history" element={<HistoryPage />} />
        </Routes>
      </main>
    </div>
  );
}
