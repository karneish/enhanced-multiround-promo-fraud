import React from 'react';

interface Props { lines: { t: number; text: string }[]; autoScroll?: boolean; }
export default function LogPanel({ lines, autoScroll = true }: Props) {
  const ref = React.useRef<HTMLDivElement>(null);
  React.useEffect(() => {
    if (autoScroll && ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [lines.length, autoScroll]);
  return (
    <div className="log-panel" ref={ref}>
      {lines.length === 0 && <div className="empty">Waiting for logs...</div>}
      {lines.map((l, i) => {
        const cls = /\[error\]/.test(l.text) ? 'err' : /\[warn\]/.test(l.text) ? 'warn' : /\[done\]/.test(l.text) ? 'ok' : '';
        return <div key={i} className={`log-line ${cls}`}><span className="t">{l.t.toFixed(1)}s</span><span className="tx">{l.text}</span></div>;
      })}
    </div>
  );
}
