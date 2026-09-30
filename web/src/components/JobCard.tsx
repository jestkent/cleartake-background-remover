import { useState } from "react";
import { api, formatBytes, formatDuration } from "../api";
import type { Job } from "../types";
import { Waveform } from "./Waveform";

interface Props {
  job: Job;
  onCancel: (id: string) => void;
}

const STATUS_STYLES: Record<string, string> = {
  queued: "bg-paper text-muted",
  running: "bg-signalsoft text-signal",
  done: "bg-signalsoft text-signal",
  failed: "bg-[#F6E2DE] text-alert",
  cancelled: "bg-paper text-muted",
};

const STATUS_WORDS: Record<string, string> = {
  queued: "Waiting",
  running: "Working",
  done: "Cleaned",
  failed: "Failed",
  cancelled: "Cancelled",
};

export function JobCard({ job, onCancel }: Props) {
  const [comparing, setComparing] = useState(false);
  const result = job.result;
  const active = job.status === "running" || job.status === "queued";

  return (
    <article className="rounded border border-hairline bg-surface">
      <header className="flex items-start justify-between gap-4 p-4">
        <div className="min-w-0 flex-1">
          <h3 className="truncate font-medium" title={job.filename}>
            {job.filename}
          </h3>
          <p className="readout mt-0.5 text-xs text-muted">
            {formatBytes(job.source_size_bytes)}
            {job.duration_seconds > 0 &&
              ` · ${formatDuration(job.duration_seconds)}`}
            {result && ` · ${result.engine_used}`}
          </p>
        </div>
        <span
          className={`shrink-0 rounded px-2 py-0.5 text-xs font-medium ${
            STATUS_STYLES[job.status] ?? "bg-paper text-muted"
          }`}
        >
          {STATUS_WORDS[job.status] ?? job.status}
        </span>
      </header>

      {active && (
        <div className="px-4 pb-4">
          <div className="h-1 w-full overflow-hidden rounded-full bg-paper">
            <div
              className="h-full bg-signal transition-[width] duration-300"
              style={{ width: `${Math.max(2, job.progress * 100)}%` }}
            />
          </div>
          <div className="mt-2 flex items-center justify-between">
            <p className="text-xs text-muted">{job.message}</p>
            <button
              type="button"
              onClick={() => onCancel(job.id)}
              className="text-xs text-muted underline-offset-2 hover:text-ink hover:underline"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {job.status === "failed" && job.error && (
        <div className="mx-4 mb-4 rounded border border-[#E7C9C2] bg-[#FBF1EF] p-3">
          <p className="text-sm text-alert">{job.error}</p>
        </div>
      )}

      {job.status === "done" && result && (
        <div className="border-t border-hairline">
          {result.output ? (
            <>
              <div className="grid grid-cols-2 gap-px bg-hairline sm:grid-cols-4">
                <Metric
                  label="Noise removed"
                  value={`${result.noise_reduction_db.toFixed(1)} dB`}
                  emphasis
                />
                <Metric
                  label="Noise floor"
                  value={`${result.noise_floor_after_db.toFixed(0)} dB`}
                  sub={`was ${result.noise_floor_before_db.toFixed(0)}`}
                />
                <Metric
                  label="Clicks repaired"
                  value={String(result.clicks_repaired)}
                />
                <Metric
                  label="Took"
                  value={`${result.elapsed_seconds.toFixed(0)}s`}
                  sub={
                    result.duration_seconds > 0
                      ? `${(
                          result.duration_seconds / result.elapsed_seconds
                        ).toFixed(1)}x speed`
                      : undefined
                  }
                />
              </div>

              {result.waveform_before.length > 0 && (
                <div className="px-4 pb-1 pt-4">
                  <Waveform
                    before={result.waveform_before}
                    after={result.waveform_after}
                  />
                </div>
              )}

              <div className="flex flex-wrap items-center gap-2 p-4">
                <a
                  href={api.downloadUrl(job.id)}
                  className="rounded bg-ink px-3.5 py-2 text-sm font-medium text-white transition-opacity hover:opacity-85"
                >
                  Download
                </a>
                <button
                  type="button"
                  onClick={() => setComparing(!comparing)}
                  className="rounded border border-hairline px-3.5 py-2 text-sm hover:border-muted"
                >
                  {comparing ? "Hide comparison" : "Listen to both"}
                </button>
                {result.video_untouched && (
                  <span className="ml-auto text-xs text-muted">
                    Video stream verified identical to the source
                  </span>
                )}
              </div>

              {comparing && (
                <div className="grid gap-4 border-t border-hairline p-4 sm:grid-cols-2">
                  <PlayerColumn
                    title="Original"
                    src={api.sourceUrl(job.id)}
                    note="What you started with."
                  />
                  <PlayerColumn
                    title="Cleaned"
                    src={api.previewUrl(job.id)}
                    note="Same picture, rebuilt audio."
                  />
                </div>
              )}
            </>
          ) : (
            <div className="grid grid-cols-2 gap-px bg-hairline">
              <Metric
                label="Noise floor"
                value={`${result.noise_floor_before_db.toFixed(1)} dB`}
              />
              <Metric
                label="Average level"
                value={`${result.level_before_db.toFixed(1)} dB`}
              />
            </div>
          )}

          {result.warnings.length > 0 && (
            <div className="border-t border-hairline px-4 py-3">
              {result.warnings.map((warning, i) => (
                <p key={i} className="text-xs text-muted">
                  {warning}
                </p>
              ))}
            </div>
          )}
        </div>
      )}
    </article>
  );
}

function Metric({
  label,
  value,
  sub,
  emphasis = false,
}: {
  label: string;
  value: string;
  sub?: string;
  emphasis?: boolean;
}) {
  return (
    <div className="bg-surface px-4 py-3">
      <p className="text-xs text-muted">{label}</p>
      <p
        className={`readout mt-0.5 ${
          emphasis ? "text-xl text-signal" : "text-lg"
        }`}
      >
        {value}
      </p>
      {sub && <p className="readout text-xs text-muted">{sub}</p>}
    </div>
  );
}

function PlayerColumn({
  title,
  src,
  note,
}: {
  title: string;
  src: string;
  note: string;
}) {
  return (
    <div>
      <h4 className="text-sm font-medium">{title}</h4>
      <p className="mb-2 text-xs text-muted">{note}</p>
      <video
        src={src}
        controls
        preload="metadata"
        className="w-full rounded bg-black"
        style={{ maxHeight: 240 }}
      />
    </div>
  );
}
