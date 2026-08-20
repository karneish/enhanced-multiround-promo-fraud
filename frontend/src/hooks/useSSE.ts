import { useEffect, useRef, useState, useCallback } from 'react';

export interface SSEEvent {
  type: string;
  _idx?: number;
  [key: string]: any;
}

export function useSSE(url: string | null) {
  const [events, setEvents] = useState<SSEEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const esRef = useRef<EventSource | null>(null);
  const counterRef = useRef(0);

  useEffect(() => {
    if (!url) return;
    counterRef.current = 0;
    setEvents([]);
    const es = new EventSource(url);
    esRef.current = es;
    es.onopen = () => setConnected(true);
    es.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        data._idx = counterRef.current++;
        setEvents((prev) => [...prev, data]);
      } catch { /* ignore */ }
    };
    es.onerror = () => setConnected(false);
    return () => { es.close(); esRef.current = null; setConnected(false); };
  }, [url]);

  const reset = useCallback(() => { counterRef.current = 0; setEvents([]); }, []);

  return { events, connected, reset };
}
