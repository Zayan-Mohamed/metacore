/**
 * An "i" that explains a term on hover or keyboard focus. The explanation is also the button's
 * accessible label, so screen readers get it without the hover. The bubble is positioned against
 * the window (fixed), so a scrolling table around it cannot clip it.
 */
import { useRef, useState } from "react";

import "./info.css";

export default function Info({ text, label }: { text: string; label?: string }) {
  const ref = useRef<HTMLSpanElement>(null);
  const [pos, setPos] = useState<{ left: number; top: number; below: boolean } | null>(null);

  function show() {
    const r = ref.current?.getBoundingClientRect();
    if (!r) return;
    const below = r.top < 140; // near the top of the window: open downwards
    const left = Math.min(Math.max(r.left + r.width / 2, 150), window.innerWidth - 150);
    setPos({ left, top: below ? r.bottom + 8 : r.top - 8, below });
  }

  return (
    <span ref={ref} className="info" tabIndex={0} role="button"
      aria-label={label ? `${label}: ${text}` : text}
      onMouseEnter={show} onFocus={show} onMouseLeave={() => setPos(null)} onBlur={() => setPos(null)}>
      <span aria-hidden="true">i</span>
      {pos ? (
        <span className="info__bubble" role="tooltip"
          style={{ left: pos.left, top: pos.top, transform: `translate(-50%, ${pos.below ? "0" : "-100%"})` }}>
          {text}
        </span>
      ) : null}
    </span>
  );
}
