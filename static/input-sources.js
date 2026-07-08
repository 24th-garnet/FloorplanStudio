/** Supported scan / model input sources (must match app.py INPUT_SOURCE_SPECS). */

export const INPUT_SOURCES = {
  roomplan_usdz: {
    id: "roomplan_usdz",
    label: "RoomPlan",
    description: "RoomPlan のファイルをインポートします。",
    extensions: [".usdz"],
    kind: "usdz",
  },
  polycam_dxf: {
    id: "polycam_dxf",
    label: "PolyCAM",
    description: "PolyCAM からエクスポートしたファイルをインポートします。",
    extensions: [".dxf", ".xdf"],
    kind: "dxf",
  },
  scaniverse_usdz: {
    id: "scaniverse_usdz",
    label: "Scaniverse",
    description: "Scaniverse からエクスポートしたファイルをインポートします。",
    extensions: [".usdz"],
    kind: "usdz",
  },
};

export const STORAGE_KEY = "prestage-input-source";

export function listSources({ kind } = {}) {
  const all = Object.values(INPUT_SOURCES);
  if (!kind) return all;
  return all.filter((s) => s.kind === kind);
}

export function getSource(id) {
  return INPUT_SOURCES[id] || null;
}

export function acceptAttribute(source) {
  return (source?.extensions || []).join(",");
}

export function sourceLabel(id) {
  return getSource(id)?.label || id;
}

export function primaryExtension(id) {
  const source = getSource(id);
  return source?.extensions?.[0] || "";
}

export function decorateInputSourcePicker(picker) {
  if (!picker) return;
  picker.querySelectorAll(".input-source-option").forEach((option) => {
    const input = option.querySelector('input[type="radio"]');
    const source = getSource(input?.value);
    if (!source) return;

    const ext = option.querySelector(".input-source-ext");
    if (ext) ext.textContent = primaryExtension(source.id);
  });
}

export function readStoredSource(fallback = "roomplan_usdz") {
  const stored = localStorage.getItem(STORAGE_KEY);
  return getSource(stored) ? stored : fallback;
}

export function storeSource(id) {
  if (getSource(id)) localStorage.setItem(STORAGE_KEY, id);
}

export function validateFileForSource(file, sourceId) {
  const source = getSource(sourceId);
  if (!source) return "入力ソースを選択してください";
  if (!file) return "ファイルを選択してください";
  const ext = file.name.includes(".") ? file.name.slice(file.name.lastIndexOf(".")).toLowerCase() : "";
  if (!source.extensions.includes(ext)) {
    return `${source.label}で作成したファイルを選択してください`;
  }
  return null;
}

export function syncInputSourceThumb(picker) {
  if (!picker) return;
  const track = picker.querySelector(".input-source-track") || picker;
  const thumb = track.querySelector(".input-source-thumb");
  const checked = picker.querySelector('input[type="radio"]:checked');
  const option = checked?.closest(".input-source-option");
  if (!thumb || !option) return;

  const trackRect = track.getBoundingClientRect();
  const optionRect = option.getBoundingClientRect();
  const x = optionRect.left - trackRect.left;
  const y = optionRect.top - trackRect.top;

  thumb.style.width = `${optionRect.width}px`;
  thumb.style.height = `${optionRect.height}px`;
  thumb.style.top = `${y}px`;
  thumb.style.transform = `translateX(${x}px)`;
}

export function bindInputSourceThumb(picker, onChange, options = {}) {
  if (!picker) return;
  const persistSelection = options.persistSelection !== false;

  const refresh = () => {
    syncInputSourceThumb(picker);
    if (typeof onChange === "function") {
      const checked = picker.querySelector('input[type="radio"]:checked');
      if (checked instanceof HTMLInputElement) onChange(checked.value);
    }
  };

  picker.querySelectorAll('input[type="radio"]').forEach((input) => {
    input.addEventListener("change", () => {
      if (!(input instanceof HTMLInputElement) || !input.checked) return;
      if (persistSelection) storeSource(input.value);
      refresh();
    });
  });

  window.addEventListener("resize", () => syncInputSourceThumb(picker));

  const panel = picker.closest(".sidebar-panel");
  panel?.addEventListener("toggle", () => {
    requestAnimationFrame(() => syncInputSourceThumb(picker));
  });

  requestAnimationFrame(refresh);
}
