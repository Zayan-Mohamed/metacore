/**
 * One-axis line chart with a crosshair tooltip, a threshold line and an optional "now" marker.
 * Hand-rolled SVG like the rest of the dashboard (no chart library). Series colours come from the
 * --series-* roles in styles/page.css; text always stays in text ink.
 */
import { useMemo, useRef, useState } from "react";

export interface LineSeries {
  id: string;
  label: string;
  color: string; // a CSS colour, normally var(--series-*)
  dashed?: boolean;
  width?: number;
  /** One value per x position; null leaves a gap. */
  values: (number | null)[];
}

interface Props {
  xLabels: string[];
  series: LineSeries[];
  threshold?: { value: number; label: string };
  nowIndex?: number;
  yDomain?: [number, number];
  height?: number;
  format?: (v: number) => string;
  ariaLabel: string;
}

const W = 760;
const PAD = { l: 44, r: 14, t: 12, b: 26 };

export default function LineChart({
  xLabels, series, threshold, nowIndex, yDomain, height = 220, format = (v) => v.toFixed(3), ariaLabel,
}: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<number | null>(null);
  const n = xLabels.length;
  const [lo, hi] = useMemo<[number, number]>(() => {
    if (yDomain) return yDomain;
    const all = series.flatMap((s) => s.values.filter((v): v is number => v !== null));
    if (threshold) all.push(threshold.value);
    const mn = Math.min(...all), mx = Math.max(...all);
    const pad = (mx - mn) * 0.08 || 0.05;
    return [mn - pad, mx + pad];
  }, [series, threshold, yDomain]);

  const x = (i: number) => PAD.l + (n <= 1 ? 0 : (i / (n - 1)) * (W - PAD.l - PAD.r));
  const y = (v: number) => PAD.t + (1 - (v - lo) / (hi - lo || 1)) * (height - PAD.t - PAD.b);
  const ticks = [lo, (lo + hi) / 2, hi];
  const xTicks = n > 1 ? [0, Math.floor((n - 1) / 2), n - 1] : [0];

  function path(values: (number | null)[]) {
    let d = "";
    let pen = false;
    values.forEach((v, i) => {
      if (v === null) {
        pen = false;
        return;
      }
      d += `${pen ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`;
      pen = true;
    });
    return d;
  }

  function onMove(e: React.MouseEvent<SVGSVGElement>) {
    const box = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - box.left) / box.width) * W;
    const i = Math.round(((px - PAD.l) / (W - PAD.l - PAD.r)) * (n - 1));
    setHover(i >= 0 && i < n ? i : null);
  }

  const tipLeft = hover !== null ? (x(hover) / W) * 100 : 0;

  return (
    <div className="pg__chart" ref={ref}>
      <svg viewBox={`0 0 ${W} ${height}`} role="img" aria-label={ariaLabel}
        onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={PAD.l} x2={W - PAD.r} y1={y(t)} y2={y(t)} stroke="var(--border)" strokeWidth={1} />
            <text x={PAD.l - 6} y={y(t) + 3.5} textAnchor="end" fontSize={10} fill="var(--text-dim)"
              fontFamily="IBM Plex Mono, monospace">{format(t)}</text>
          </g>
        ))}
        {xTicks.map((i) => (
          <text key={i} x={x(i)} y={height - 8} textAnchor={i === 0 ? "start" : i === n - 1 ? "end" : "middle"}
            fontSize={10} fill="var(--text-dim)" fontFamily="IBM Plex Mono, monospace">
            {xLabels[i]?.replace("T", " ")}
          </text>
        ))}
        {threshold ? (
          <g>
            <line x1={PAD.l} x2={W - PAD.r} y1={y(threshold.value)} y2={y(threshold.value)}
              stroke="var(--amber)" strokeWidth={1.5} strokeDasharray="5 4" />
            <text x={W - PAD.r} y={y(threshold.value) - 5} textAnchor="end" fontSize={10}
              fill="var(--text-muted)">{threshold.label}</text>
          </g>
        ) : null}
        {nowIndex !== undefined ? (
          <g>
            <line x1={x(nowIndex)} x2={x(nowIndex)} y1={PAD.t} y2={height - PAD.b}
              stroke="var(--text-dim)" strokeWidth={1} />
            <text x={x(nowIndex) + 4} y={PAD.t + 10} fontSize={10} fill="var(--text-muted)">now</text>
          </g>
        ) : null}
        {series.map((s) => (
          <path key={s.id} d={path(s.values)} fill="none" stroke={s.color} strokeWidth={s.width ?? 2}
            strokeDasharray={s.dashed ? "6 4" : undefined} strokeLinejoin="round" strokeLinecap="round" />
        ))}
        {hover !== null ? (
          <g>
            <line x1={x(hover)} x2={x(hover)} y1={PAD.t} y2={height - PAD.b} stroke="var(--text-muted)"
              strokeWidth={1} />
            {series.map((s) => {
              const v = s.values[hover];
              return v === null || v === undefined ? null : (
                <circle key={s.id} cx={x(hover)} cy={y(v)} r={4} fill={s.color} stroke="var(--surface)" strokeWidth={2} />
              );
            })}
          </g>
        ) : null}
      </svg>
      {hover !== null ? (
        <div className="pg__tip" style={{ left: `calc(${tipLeft}% + ${tipLeft > 60 ? -12 : 12}px)`,
          top: 8, transform: tipLeft > 60 ? "translateX(-100%)" : undefined }}>
          <div className="mono" style={{ color: "var(--text-muted)" }}>{xLabels[hover]?.replace("T", " ")}</div>
          {series.map((s) => {
            const v = s.values[hover];
            return v === null || v === undefined ? null : (
              <div key={s.id} className="pg__tip-row">
                <span className="pg__swatch" style={{ background: s.color }} />
                <span style={{ color: "var(--text-muted)" }}>{s.label}</span>
                <span style={{ marginLeft: "auto", paddingLeft: 10 }}>{format(v)}</span>
              </div>
            );
          })}
        </div>
      ) : null}
      <div className="pg__legend">
        {series.map((s) => (
          <span key={s.id} className="pg__legend-item">
            <span className={`pg__swatch${s.dashed ? " pg__swatch--dash" : ""}`}
              style={{ background: s.color, color: s.color }} />
            {s.label}
          </span>
        ))}
        {threshold ? (
          <span className="pg__legend-item">
            <span className="pg__swatch pg__swatch--dash" style={{ color: "var(--amber)" }} />
            {threshold.label}
          </span>
        ) : null}
      </div>
    </div>
  );
}
