import { useEffect, useRef } from "react";

interface Props {
  before: number[];
  after: number[];
  height?: number;
}

/**
 * Before and after, mirrored across a shared centre line.
 *
 * Original above, cleaned below. Drawing them back to back rather than in
 * two separate strips is what makes the result legible at a glance: the
 * speech peaks line up across the axis, so if the cleaned side has lost a
 * transient the asymmetry is obvious. Two stacked charts hide exactly
 * that, because the eye cannot track a small difference across a gap.
 *
 * The quiet stretches are where the real answer lives. Noise reduction is
 * judged in the gaps between words.
 */
export function Waveform({ before, after, height = 132 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const parent = canvas.parentElement;
    const cssWidth = parent ? parent.clientWidth : 800;
    const ratio = window.devicePixelRatio || 1;

    canvas.width = Math.max(1, Math.floor(cssWidth * ratio));
    canvas.height = Math.floor(height * ratio);
    canvas.style.width = "100%";
    canvas.style.height = `${height}px`;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(ratio, ratio);
    ctx.clearRect(0, 0, cssWidth, height);

    const mid = height / 2;
    const half = mid - 8;
    const points = Math.max(before.length, after.length, 1);
    const step = cssWidth / points;

    // A square-root curve compresses the loud peaks and lifts the quiet
    // floor into view. On a linear scale the hiss between words sits at a
    // couple of pixels and the whole comparison becomes invisible.
    const scale = (v: number) => Math.sqrt(Math.min(Math.abs(v), 1)) * half;

    const drawSide = (data: number[], up: boolean, color: string) => {
      ctx.beginPath();
      ctx.moveTo(0, mid);
      for (let i = 0; i < points; i += 1) {
        const value = scale(data[i] ?? 0);
        const y = up ? mid - value : mid + value;
        ctx.lineTo(i * step, y);
      }
      ctx.lineTo(cssWidth, mid);
      ctx.closePath();
      ctx.fillStyle = color;
      ctx.fill();
    };

    drawSide(before, true, "#97A1AE");
    drawSide(after, false, "#17694A");

    ctx.strokeStyle = "#C9CFD6";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, mid + 0.5);
    ctx.lineTo(cssWidth, mid + 0.5);
    ctx.stroke();
  }, [before, after, height]);

  return (
    <div className="w-full">
      <canvas
        ref={canvasRef}
        role="img"
        aria-label="Waveform comparison. The original is drawn above the centre line and the cleaned version below it."
      />
      <div className="mt-1.5 flex justify-between text-xs text-muted">
        <span className="flex items-center gap-1.5">
          <span
            className="inline-block h-2 w-2 rounded-sm"
            style={{ background: "#97A1AE" }}
          />
          Original, above the line
        </span>
        <span className="flex items-center gap-1.5">
          <span
            className="inline-block h-2 w-2 rounded-sm"
            style={{ background: "#17694A" }}
          />
          Cleaned, below the line
        </span>
      </div>
    </div>
  );
}
