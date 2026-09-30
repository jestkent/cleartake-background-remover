import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import { DropZone, SettingsPanel } from "./components/Controls";
import { JobCard } from "./components/JobCard";
import type { Job, Settings, SystemInfo } from "./types";

const DEFAULTS: Settings = {
  preset: "screen_recording",
  engine: "auto",
  attenuation_db: 24,
  declick: true,
  declick_sensitivity: 1.2,
  high_pass_hz: 80,
  normalize: true,
  target_lufs: -16,
  keep_original_track: true,
};

export default function App() {
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [settings, setSettings] = useState<Settings>(DEFAULTS);
  const [uploading, setUploading] = useState<number | null>(null);
  const [uploadName, setUploadName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [booted, setBooted] = useState(false);

  const queue = useRef<File[]>([]);
  const busy = useRef(false);

  // -- load system info once -------------------------------------------
  useEffect(() => {
    api
      .system()
      .then(setSystem)
      .catch(() =>
        setError("Could not reach the ClearTake server. Is it still running?"),
      )
      .finally(() => setBooted(true));
  }, []);

  // -- poll while anything is in flight ---------------------------------
  useEffect(() => {
    let cancelled = false;

    const tick = async () => {
      try {
        const next = await api.jobs(40);
        if (!cancelled) setJobs(next);
      } catch {
        /* a dropped poll is not worth surfacing; the next one will tell */
      }
    };

    void tick();

    // Poll fast while something is running, slowly when idle. A fixed
    // one second interval would hammer the server all day for nothing.
    const active = jobs.some(
      (j) => j.status === "running" || j.status === "queued",
    );
    const interval = window.setInterval(tick, active ? 700 : 5000);

    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [jobs.some((j) => j.status === "running" || j.status === "queued")]);

  // -- upload queue ------------------------------------------------------
  const pump = useCallback(async () => {
    if (busy.current) return;
    const file = queue.current.shift();
    if (!file) return;

    busy.current = true;
    setUploadName(file.name);
    setUploading(0);
    setError(null);

    try {
      const job = await api.upload(file, settings, setUploading);
      setJobs((prev) => [job, ...prev]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed.");
      queue.current = [];
    } finally {
      setUploading(null);
      setUploadName("");
      busy.current = false;
      if (queue.current.length) void pump();
    }
  }, [settings]);

  const onFiles = useCallback(
    (files: File[]) => {
      queue.current.push(...files);
      void pump();
    },
    [pump],
  );

  const onCancel = useCallback(async (id: string) => {
    try {
      await api.cancel(id);
      setJobs(await api.jobs(40));
    } catch {
      /* the job finished between render and click; the poll will correct it */
    }
  }, []);

  const onClearHistory = useCallback(async () => {
    await api.clearHistory();
    setJobs(await api.jobs(40));
  }, []);

  const ready = Boolean(system?.ffmpeg);
  const finished = jobs.filter(
    (j) => j.status === "done" || j.status === "failed" || j.status === "cancelled",
  );
  const running = jobs.filter(
    (j) => j.status === "running" || j.status === "queued",
  );

  return (
    <div className="min-h-screen">
      <header className="border-b border-hairline bg-surface">
        <div className="mx-auto flex max-w-4xl items-center justify-between gap-4 px-5 py-4">
          <div>
            <h1 className="text-lg font-semibold tracking-tight">ClearTake</h1>
            <p className="text-sm text-muted">
              Clean the audio. Leave the video alone.
            </p>
          </div>
          {system && (
            <div className="text-right text-xs text-muted">
              <p>
                {system.active_engine === "none"
                  ? "No engine available"
                  : system.engines.find((e) => e.key === system.active_engine)
                      ?.name}
              </p>
              <p className="readout">v{system.version}</p>
            </div>
          )}
        </div>
      </header>

      <main className="mx-auto max-w-4xl space-y-5 px-5 py-6">
        {booted && system && !system.ffmpeg && (
          <Notice tone="alert" title="ffmpeg is missing">
            {system.ffmpeg_detail}
          </Notice>
        )}

        {booted && system?.active_engine === "spectral" && (
          <Notice tone="info" title="Running on the spectral gate">
            This handles steady noise like fans and hiss. For background
            chatter, traffic, or a noisy room, install DeepFilterNet 3 with{" "}
            <code className="readout rounded bg-paper px-1 py-0.5 text-[0.8em]">
              pip install -r requirements-models.txt
            </code>{" "}
            and restart. ClearTake will pick it up on its own.
          </Notice>
        )}

        {error && (
          <Notice tone="alert" title="Something went wrong">
            {error}
          </Notice>
        )}

        <DropZone
          onFiles={onFiles}
          disabled={!ready}
          uploading={uploading}
          uploadName={uploadName}
        />

        {system && (
          <SettingsPanel
            settings={settings}
            presets={system.presets}
            engines={system.engines}
            onChange={(next) => setSettings((prev) => ({ ...prev, ...next }))}
          />
        )}

        {running.length > 0 && (
          <section className="space-y-3">
            <h2 className="text-sm font-semibold">In progress</h2>
            {running.map((job) => (
              <JobCard key={job.id} job={job} onCancel={onCancel} />
            ))}
          </section>
        )}

        {finished.length > 0 && (
          <section className="space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold">Finished</h2>
              <button
                type="button"
                onClick={onClearHistory}
                className="text-xs text-muted underline-offset-2 hover:text-ink hover:underline"
              >
                Clear list
              </button>
            </div>
            {finished.map((job) => (
              <JobCard key={job.id} job={job} onCancel={onCancel} />
            ))}
          </section>
        )}

        {booted && jobs.length === 0 && ready && (
          <p className="py-6 text-center text-sm text-muted">
            Nothing cleaned yet. Drop a file above to start.
          </p>
        )}
      </main>

      <footer className="mx-auto max-w-4xl px-5 pb-10 pt-4">
        <p className="text-xs leading-relaxed text-muted">
          Everything runs on this machine. No uploads leave it, and nothing is
          sent to an external service.
        </p>
      </footer>
    </div>
  );
}

function Notice({
  tone,
  title,
  children,
}: {
  tone: "info" | "alert";
  title: string;
  children: React.ReactNode;
}) {
  const styles =
    tone === "alert"
      ? "border-[#E7C9C2] bg-[#FBF1EF]"
      : "border-hairline bg-surface";
  return (
    <div className={`rounded border p-4 ${styles}`}>
      <p
        className={`text-sm font-medium ${
          tone === "alert" ? "text-alert" : "text-ink"
        }`}
      >
        {title}
      </p>
      <p className="mt-1 text-sm leading-relaxed text-muted">{children}</p>
    </div>
  );
}
