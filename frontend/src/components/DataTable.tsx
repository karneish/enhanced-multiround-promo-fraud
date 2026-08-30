import React from 'react';

interface Props { columns: string[]; rows: Record<string, any>[]; renderCell?: (col: string, row: Record<string, any>, i: number) => React.ReactNode; }
export default function DataTable({ columns, rows, renderCell }: Props) {
  return (
    <div className="table-wrap">
      <table className="table">
        <thead><tr>{columns.map((c) => <th key={c}>{c}</th>)}</tr></thead>
        <tbody>
          {rows.length === 0 && <tr><td colSpan={columns.length} className="empty">No data</td></tr>}
          {rows.map((row, i) => (
            <tr key={i}>{columns.map((c) => (
              <td key={c}>{renderCell ? renderCell(c, row, i) : (row[c] ?? '--')}</td>
            ))}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
