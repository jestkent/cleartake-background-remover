import type { Job, Settings, SystemInfo } from "./types";

const BASE = "";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let message = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body?.detail) message = String(body.detail);
    } catch {
      /* response had no JSON body; the status message is all there is */
    }
    throw new Error(message);
  }
  return res.json() as Promise<T>;
}

export const api = {
  system: () => fetch(`${BASE}/api/system`).then(json<SystemInfo>),

  jobs: (limit = 50) =>
    fetch(`${BASE}/api/jobs?limit=${limit}`).then(json<Job[]>),

  job: (id: string) => fetch(`${BASE}/api/jobs/${id}`).then(json<Job>),

  cancel: (id: string) =>
    fetch(`${BASE}/api/jobs/${id}/cancel`, { method: "POST" }).then(
      json<{ ok: boolean }>,
    ),

  clearHistory: () =>
    fetch(`${BASE}/api/jobs`, { method: "DELETE" }).then(
      json<{ ok: boolean; removed: number }>,
    ),

  downloadUrl: (id: string) => `${BASE}/api/jobs/${id}/download`,
  previewUrl: (id: string) => `${BASE}/api/jobs/${id}/preview`,
  sourceUrl: (id: string) => `${BASE}/api/jobs/${id}/source`,

  /**
   * Upload one file.
   *
   * XMLHttpRequest rather than fetch, purely because fetch still has no
   * upload progress event. On a 3 GB lecture recording a progress bar is
   * not decoration, it is the difference between waiting and wondering
   * whether the thing has hung.
   */
  upload(
    file: File,
    settings: Settings,
    onProgress?: (fraction: number) => void,
  ): Promise<Job> {
    const form = new FormData();
    form.append("file", file);
    form.append("preset", settings.preset);
    form.append("engine", settings.engine);
    form.append("attenuation_db", String(settings.attenuation_db));
    form.append("declick", String(settings.declick));
    form.append("declick_sensitivity", String(settings.declick_sensitivity));
    form.append("high_pass_hz", String(settings.high_pass_hz));
    form.append("normalize", String(settings.normalize));
    form.append("target_lufs", String(settings.target_lufs));
    form.append("keep_original_track", String(settings.keep_original_track));

    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${BASE}/api/jobs`);

      xhr.upload.addEventListener("progress", (event) => {
        if (event.lengthComputable && onProgress) {
          onProgress(event.loaded / event.total);
        }
      });

      xhr.addEventListener("load", () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText));
          } catch {
            reject(new Error("The server sent back something unreadable."));
          }
        } else {
          let message = `Upload failed (${xhr.status})`;
          try {
            const body = JSON.parse(xhr.responseText);
            if (body?.detail) message = String(body.detail);
          } catch {
            /* keep the status-based message */
          }
          reject(new Error(message));
        }
      });

      xhr.addEventListener("error", () =>
        reject(new Error("Could not reach the server. Is it still running?")),
      );
      xhr.addEventListener("abort", () =>
        reject(new Error("Upload cancelled.")),
      );

      xhr.send(form);
    });
  },
};

export function formatBytes(n: number): string {
  if (!n) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let value = n;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${unit === 0 ? Math.round(value) : value.toFixed(1)} ${units[unit]}`;
}

export function formatDuration(seconds: number): string {
  if (!seconds || seconds < 0) return "0:00";
  const total = Math.round(seconds);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return h > 0
    ? `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`
    : `${m}:${String(s).padStart(2, "0")}`;
}
