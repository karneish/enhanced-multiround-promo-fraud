import React from 'react';

interface Props { total: number; current: number; }
export default function RoundTracker({ total, current }: Props) {
  return (
    <div className="round-track">
      <span className="rt-label">Rounds</span>
      {Array.from({ length: total }, (_, i) => (
        <div key={i} className={`round-chip ${i < current ? 'done' : i === current ? 'active' : ''}`}>
          <span className="ball" />r{i}
        </div>
      ))}
    </div>
  );
}
