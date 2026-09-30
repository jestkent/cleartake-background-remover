import { useRef, useState } from "react";
import type { Engine, Preset, Settings } from "../types";

// ---------------------------------------------------------------------------
// drop zone
// ---------------------------------------------------------------------------

interface DropProps {
  onFiles: (files: File[]) => void;
  disabled: boolean;
  uploading: number | null;
  uploadName: string;
}

export function DropZone({ onFiles, disabled, uploading, uploadName }: DropProps) {
  const [hovering, setHovering] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const accept = (list: FileList | null) => {
    if (!list || disabled) return;
    const files = Array.from(list);
    if (files.length) onFiles(files);
  };

  if (uploading !== null) {
    return (
      <div className="rounded border border-hairline bg-surface p-8">
        <p className="text-sm text-muted">Uploading</p>
        <p className="mt-1 truncate font-medium">{uploadName}</p>
        <div className="mt-4 h-1 w-full overflow-hidden rounded-full bg-paper">
          <div
            className="h-full bg-signal transition-[width] duration-150"
            style={{ width: `${Math.round(uploading * 100)}%` }}
          />
        </div>
        <p className="readout mt-2 text-sm text-muted">
          {Math.round(uploading * 100)}%
        </p>
      </div>
    );
  }

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setHovering(true);
      }}
      onDragLeave={() => setHovering(false)}
      onDrop={(e) => {
        e.preventDefault();
        setHovering(false);
        accept(e.dataTransfer.files);
      }}
      className={[
        "rounded border-2 border-dashed p-10 text-center transition-colors",
        disabled
          ? "cursor-not-allowed border-hairline bg-paper opacity-60"
          : hovering
            ? "border-signal bg-signalsoft"
            : "border-hairline bg-surface hover:border-muted",
      ].join(" ")}
    >
      <p className="text-lg font-medium">Drop a video here</p>
      <p className="mx-auto mt-1 max-w-sm text-sm text-muted">
        The picture is copied across untouched. Only the audio is rebuilt.
      </p>
      <button
        type="button"
        disabled={disabled}
        onClick={() => inputRef.current?.click()}
        className="mt-5 rounded bg-ink px-4 py-2 text-sm font-medium text-white transition-opacity hover:opacity-85 disabled:cursor-not-allowed disabled:opacity-40"
      >
        Choose files
      </button>
      <input
        ref={inputRef}
        type="file"
        multiple
        accept="video/*,audio/*"
        className="hidden"
        onChange={(e) => {
          accept(e.target.files);
          e.target.value = "";
        }}
      />
      <p className="mt-4 text-xs text-muted">
        mp4, mov, mkv, webm, wav, mp3, m4a and more
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// settings
// ---------------------------------------------------------------------------

interface SettingsProps {
  settings: Settings;
  presets: Preset[];
  engines: Engine[];
  onChange: (next: Partial<Settings>) => void;
}

export function SettingsPanel({
  settings,
  presets,
  engines,
  onChange,
}: SettingsProps) {
  const [open, setOpen] = useState(false);
  const usable = engines.filter((e) => e.available);

  return (
    <section className="rounded border border-hairline bg-surface">
      <div className="p-5">
        <h2 className="text-sm font-semibold">What kind of recording is it?</h2>
        <div className="mt-3 grid gap-2 sm:grid-cols-2">
          {presets
            .filter((p) => p.key !== "analyse_only")
            .map((preset) => {
              const active = settings.preset === preset.key;
              return (
                <button
                  key={preset.key}
                  type="button"
                  onClick={() =>
                    onChange({
                      preset: preset.key,
                      attenuation_db: preset.attenuation_db,
                      declick: preset.declick,
                    })
                  }
                  className={[
                    "rounded border p-3 text-left transition-colors",
                    active
                      ? "border-signal bg-signalsoft"
                      : "border-hairline hover:border-muted",
                  ].join(" ")}
                >
                  <span className="block text-sm font-medium">{preset.label}</span>
                  <span className="mt-0.5 block text-xs leading-snug text-muted">
                    {preset.blurb}
                  </span>
                </button>
              );
            })}
        </div>
      </div>

      <div className="border-t border-hairline">
        <button
          type="button"
          onClick={() => setOpen(!open)}
          className="flex w-full items-center justify-between px-5 py-3 text-sm text-muted hover:text-ink"
          aria-expanded={open}
        >
          <span>Fine tuning</span>
          <span aria-hidden>{open ? "Hide" : "Show"}</span>
        </button>

        {open && (
          <div className="space-y-5 border-t border-hairline px-5 py-5">
            <Slider
              label="How much noise to remove"
              hint="Higher is quieter but less natural. Past about 35 dB you start to hear the processing."
              value={settings.attenuation_db}
              min={0}
              max={60}
              step={1}
              suffix=" dB"
              onChange={(v) => onChange({ attenuation_db: v })}
            />

            <div>
              <Toggle
                label="Repair mouse clicks and keyboard"
                checked={settings.declick}
                onChange={(v) => onChange({ declick: v })}
              />
              {settings.declick && (
                <div className="mt-3 pl-1">
                  <Slider
                    label="Click sensitivity"
                    hint="Above 2.0 it starts softening hard consonants."
                    value={settings.declick_sensitivity}
                    min={0.2}
                    max={2.5}
                    step={0.1}
                    onChange={(v) => onChange({ declick_sensitivity: v })}
                  />
                </div>
              )}
            </div>

            <Slider
              label="Cut rumble below"
              hint="Removes desk thumps and air handling. 0 leaves the low end alone."
              value={settings.high_pass_hz}
              min={0}
              max={200}
              step={5}
              suffix=" Hz"
              onChange={(v) => onChange({ high_pass_hz: v })}
            />

            <div>
              <Toggle
                label="Match loudness"
                checked={settings.normalize}
                onChange={(v) => onChange({ normalize: v })}
              />
              {settings.normalize && (
                <div className="mt-3 pl-1">
                  <Slider
                    label="Loudness target"
                    hint="-14 for YouTube, -16 for most other places."
                    value={settings.target_lufs}
                    min={-31}
                    max={-9}
                    step={1}
                    suffix=" LUFS"
                    onChange={(v) => onChange({ target_lufs: v })}
                  />
                </div>
              )}
            </div>

            <Toggle
              label="Keep the original audio as a second track"
              hint="Lets you A/B inside your editor. Costs a few MB."
              checked={settings.keep_original_track}
              onChange={(v) => onChange({ keep_original_track: v })}
            />

            <div>
              <label
                htmlFor="engine"
                className="block text-sm font-medium"
              >
                Engine
              </label>
              <select
                id="engine"
                value={settings.engine}
                onChange={(e) => onChange({ engine: e.target.value })}
                className="mt-1.5 w-full rounded border border-hairline bg-surface px-3 py-2 text-sm"
              >
                <option value="auto">
                  Pick automatically
                  {usable.length ? ` (${usable[0].name})` : ""}
                </option>
                {engines.map((engine) => (
                  <option
                    key={engine.key}
                    value={engine.key}
                    disabled={!engine.available}
                  >
                    {engine.name}
                    {engine.available ? "" : " — not installed"}
                  </option>
                ))}
              </select>
              <p className="mt-1.5 text-xs leading-snug text-muted">
                {engines.find((e) => e.key === settings.engine)?.description ??
                  "Uses the best engine you have installed."}
              </p>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------

function Slider({
  label,
  hint,
  value,
  min,
  max,
  step,
  suffix = "",
  onChange,
}: {
  label: string;
  hint?: string;
  value: number;
  min: number;
  max: number;
  step: number;
  suffix?: string;
  onChange: (value: number) => void;
}) {
  const id = label.replace(/\s+/g, "-").toLowerCase();
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <label htmlFor={id} className="text-sm font-medium">
          {label}
        </label>
        <span className="readout text-sm text-muted">
          {value}
          {suffix}
        </span>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="mt-2 w-full"
      />
      {hint && <p className="mt-1 text-xs leading-snug text-muted">{hint}</p>}
    </div>
  );
}

function Toggle({
  label,
  hint,
  checked,
  onChange,
}: {
  label: string;
  hint?: string;
  checked: boolean;
  onChange: (value: boolean) => void;
}) {
  const id = label.replace(/\s+/g, "-").toLowerCase();
  return (
    <div>
      <div className="flex items-center gap-2.5">
        <input
          id={id}
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          className="h-4 w-4 accent-signal"
        />
        <label htmlFor={id} className="text-sm font-medium">
          {label}
        </label>
      </div>
      {hint && <p className="mt-1 pl-7 text-xs leading-snug text-muted">{hint}</p>}
    </div>
  );
}
