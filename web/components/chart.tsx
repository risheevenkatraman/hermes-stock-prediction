"use client";
import { useEffect, useRef, useState } from "react";
import type { Research } from "@/lib/types";

export default function PriceChart({ research, range, horizon, showForecast }: { research: Research; range: string; horizon: number; showForecast: boolean }) {
  const [hover, setHover] = useState<number | null>(null);
  const container = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(820);
  useEffect(() => {
    if (!container.current) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.max(260, entry.contentRect.width)));
    observer.observe(container.current);
    return () => observer.disconnect();
  }, []);
  const count = ({ "1W": 5, "1M": 22, "3M": 66, "1Y": 252, ALL: Infinity } as Record<string, number>)[range];
  const rows = research.prices.slice(-count);
  const forecast = showForecast ? research.forecasts.find(item => item.horizon === horizon) : undefined;
  if (!rows.length) return <div className="empty-state">No price history available.</div>;
  const values = rows.map(row => row.close).concat(forecast ? [forecast.expected_price] : []);
  const min = Math.min(...values), max = Math.max(...values);
  const padding = (max - min) * .22 || 1;
  const low = min - padding, high = max + padding;
  const height = 270, left = 8, right = width - 62;
  const end = forecast ? right - (width < 500 ? 50 : 90) : right;
  const x = (index: number) => left + index / Math.max(rows.length - 1, 1) * (end - left);
  const y = (value: number) => 15 + (high - value) / (high - low) * (height - 40);
  const points = rows.map((row, i) => `${x(i)},${y(row.close)}`).join(" ");
  const selected = hover === null ? null : rows[Math.min(hover, rows.length - 1)];
  return <div className="chart-wrap" ref={container}>
    <div className="chart-readout" aria-live="polite">{selected ? `${selected.date} · $${selected.close.toFixed(2)}` : "Daily closing prices · USD"}</div>
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${research.ticker} price chart. ${rows[0].date} to ${rows.at(-1)!.date}. Latest close $${rows.at(-1)!.close.toFixed(2)}.`} onMouseLeave={() => setHover(null)} onMouseMove={event => {
      const rect = event.currentTarget.getBoundingClientRect();
      const px = (event.clientX - rect.left) / rect.width * width;
      setHover(Math.max(0, Math.min(rows.length - 1, Math.round((px - left) / (end - left) * (rows.length - 1)))));
    }}>
      <defs><linearGradient id="chart-fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#245b48" stopOpacity=".16"/><stop offset="100%" stopColor="#245b48" stopOpacity="0"/></linearGradient></defs>
      {[0, 1, 2, 3, 4].map(i => { const price = high - (high - low) * i / 4; return <g key={i}><line x1={left} x2={right} y1={y(price)} y2={y(price)} stroke="#e8e7df" strokeDasharray="3 5"/><text x={right + 12} y={y(price) + 4} fill="#6f7968" fontSize="10">{price.toFixed(2)}</text></g>; })}
      <polygon points={`${left},${height - 20} ${points} ${end},${height - 20}`} fill="url(#chart-fill)"/>
      <polyline points={points} fill="none" stroke="#245b48" strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round"/>
      {forecast && <g><rect x={end + 8} y="10" width={right - end - 8} height={height - 30} fill="#e9b591" opacity=".13" rx="8"/><line x1={end} x2={right} y1={y(rows.at(-1)!.close)} y2={y(forecast.expected_price)} stroke="#c97e48" strokeWidth="2.5" strokeDasharray="6 5"/><circle cx={right} cy={y(forecast.expected_price)} r="4" fill="#c97e48"/><text x={end + 12} y="25" fill="#95613c" fontSize="8">{width < 500 ? "EST." : "FORECAST"}</text></g>}
      {selected && hover !== null && <g><line x1={x(Math.min(hover, rows.length - 1))} x2={x(Math.min(hover, rows.length - 1))} y1="15" y2={height - 20} stroke="#8c9a8f" strokeDasharray="3 3"/><circle cx={x(Math.min(hover, rows.length - 1))} cy={y(selected.close)} r="5" fill="#245b48" stroke="#fff" strokeWidth="2"/></g>}
    </svg>
    <div className="chart-dates"><span>{rows[0].date}</span><span>{rows[Math.floor(rows.length / 2)].date}</span><span>{rows.at(-1)!.date}</span></div>
    <details className="price-table"><summary>View chart data</summary><div><table><thead><tr><th>Date</th><th>Close (USD)</th></tr></thead><tbody>{rows.map(row => <tr key={row.date}><td>{row.date}</td><td>{row.close.toFixed(2)}</td></tr>)}</tbody></table></div></details>
  </div>;
}
