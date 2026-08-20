import React, { useState } from 'react';
import MainPage from './pages/MainPage';
import GeneratorPage from './pages/GeneratorPage';
import AdlPage from './pages/AdlPage';
import EnsemblePage from './pages/EnsemblePage';
import './App.css';

type Page = 'main' | 'generator' | 'adl' | 'ensemble';

const Icons = {
  main: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="3"/><path d="M12 1v2m0 18v2M4.22 4.22l1.42 1.42m12.72 12.72l1.42 1.42M1 12h2m18 0h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/>
    </svg>
  ),
  generator: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
    </svg>
  ),
  adl: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
    </svg>
  ),
  ensemble: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="2" y="2" width="6" height="6" rx="1"/><rect x="16" y="2" width="6" height="6" rx="1"/><rect x="9" y="9" width="6" height="6" rx="1"/><rect x="2" y="16" width="6" height="6" rx="1"/><rect x="16" y="16" width="6" height="6" rx="1"/>
    </svg>
  ),
  shield: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
    </svg>
  ),
};

const NAV: { id: Page; label: string; icon: React.ReactNode }[] = [
  { id: 'ensemble',  label: 'Ensemble Detector', icon: Icons.ensemble },
  { id: 'adl',       label: 'Adaptive Defense',  icon: Icons.adl },
  { id: 'generator', label: 'Fraud Generator',   icon: Icons.generator },
  { id: 'main',      label: 'Research Framework', icon: Icons.main },
];

export default function App() {
  const [page, setPage] = useState<Page>('ensemble');

  return (
    <div className="app-root">
      <nav className="sidebar">
        <div className="sidebar-brand">
          <div className="brand-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
              <path d="M9 12l2 2 4-4"/>
            </svg>
          </div>
          <div className="brand-text">
            FraudGuard
            <span>Detection Platform</span>
          </div>
        </div>

        <div className="nav-section">
          <div className="nav-section-label">Modules</div>
          {NAV.map((n) => (
            <button
              key={n.id}
              className={`nav-btn ${page === n.id ? 'active' : ''}`}
              onClick={() => setPage(n.id)}
            >
              <span className="nav-icon" style={page === n.id ? { color: '#3fb950' } : { color: '#5a7a62' }}>{n.icon}</span>
              <span className="nav-label">{n.label}</span>
            </button>
          ))}
        </div>

        <div className="sidebar-footer">
          <div className="sidebar-footer-inner">
            <div className="label">System</div>
            <div className="value">4 backends + React</div>
          </div>
        </div>
      </nav>
      <main className="content">
        {page === 'main' && <MainPage />}
        {page === 'generator' && <GeneratorPage />}
        {page === 'adl' && <AdlPage />}
        {page === 'ensemble' && <EnsemblePage />}
      </main>
    </div>
  );
}
