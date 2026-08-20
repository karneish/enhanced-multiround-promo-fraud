import React from 'react';

interface Props { label: string; value: string | number; cls?: string; sub?: string; }
export default function MetricCard({ label, value, cls = 'green', sub }: Props) {
  return (
    <div className={`metric ${cls}`}>
      <div className="k">{label}</div>
      <div className="v">{value}</div>
      {sub && <div className="s">{sub}</div>}
    </div>
  );
}
