import { useEffect, useRef, useState } from 'react';

export function useCountUp(target, { duration = 700, decimals = 0 } = {}) {
  const [val, setVal] = useState(target);
  const fromRef = useRef(target);
  useEffect(() => {
    const from = fromRef.current;
    if (from === target) return;
    const start = performance.now();
    let raf;
    const ease = (t) => 1 - Math.pow(1 - t, 3);
    const step = (now) => {
      const p = Math.min(1, (now - start) / duration);
      const v = from + (target - from) * ease(p);
      setVal(Number(v.toFixed(decimals)));
      if (p < 1) raf = requestAnimationFrame(step);
      else fromRef.current = target;
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [target, duration, decimals]);
  return val;
}

export function usePrevious(value) {
  const ref = useRef();
  useEffect(() => {
    ref.current = value;
  }, [value]);
  return ref.current;
}