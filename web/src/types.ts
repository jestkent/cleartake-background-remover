export type JobStatus = "queued" | "running" | "done" | "failed" | "cancelled";

export interface Engine {
  key: string;
  name: string;
  description: string;
  available: boolean;
  install_hint: string;
  requires_gpu: boolean;
  strength: string;
  detail: string;
}

export interface Preset {
  key: string;
  label: string;
  blurb: string;
  attenuation_db: number;
  declick: boolean;
  engine: string;
}

export interface SystemInfo {
  ffmpeg: boolean;
  ffmpeg_detail: string;
  engines: Engine[];
  presets: Preset[];
  active_engine: string;
  version: string;
  max_upload_mb: number;
}

export interface Stage {
  name: string;
  seconds: number;
  detail: string;
}

export interface JobResult {
  job_id: string;
  source: string;
  output: string;
  engine_used: string;
  duration_seconds: number;
  elapsed_seconds: number;
  video_untouched: boolean;
  clicks_repaired: number;
  noise_floor_before_db: number;
  noise_floor_after_db: number;
  noise_reduction_db: number;
  level_before_db: number;
  level_after_db: number;
  waveform_before: number[];
  waveform_after: number[];
  stages: Stage[];
  warnings: string[];
}

export interface Job {
  id: string;
  filename: string;
  status: JobStatus;
  progress: number;
  message: string;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  error: string | null;
  options: Record<string, unknown>;
  result: JobResult | null;
  source_size_bytes: number;
  output_size_bytes: number;
  duration_seconds: number;
}

export interface Settings {
  preset: string;
  engine: string;
  attenuation_db: number;
  declick: boolean;
  declick_sensitivity: number;
  high_pass_hz: number;
  normalize: boolean;
  target_lufs: number;
  keep_original_track: boolean;
}
