import { useEffect, useRef, useState } from "react";

type CUEvent =
  | { type: "screenshot"; data: string; turn: number }
  | { type: "reasoning"; text: string; turn: number }
  | { type: "action"; name: string; args: Record<string, unknown>; result: string; turn: number }
  | { type: "done"; turns: number }
  | { type: "error"; message: string };

interface ClickMarker {
  x: number;
  y: number;
  turn: number;
  id: number;
}

const VIEWPORT_W = 1280;
const VIEWPORT_H = 800;

export function ComputerUsePanel({ onClose }: { onClose: () => void }) {
  const [running, setRunning] = useState(false);
  const [events, setEvents] = useState<CUEvent[]>([]);
  const [screenshot, setScreenshot] = useState<string | null>(null);
  const [markers, setMarkers] = useState<ClickMarker[]>([]);
  const [, setMarkerSeq] = useState(0);
  const logRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  const start = () => {
    if (running) return;
    setRunning(true);
    setEvents([]);
    setScreenshot(null);
    setMarkers([]);
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    (async () => {
      try {
        const res = await fetch("/api/computer-use", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({}),
          signal: ctrl.signal,
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const reader = res.body!.getReader();
        const decoder = new TextDecoder();
        let buf = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          const lines = buf.split("\n");
          buf = lines.pop()!;
          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            try {
              const ev: CUEvent = JSON.parse(line.slice(6));
              if (ev.type === "screenshot") {
                setScreenshot(ev.data);
              }
              if (ev.type === "action") {
                const coord =
                  (ev.args.coordinate as number[] | undefined) ||
                  ([ev.args.x ?? 0, ev.args.y ?? 0] as number[]);
                if (
                  ["click", "left_click", "single_click", "double_click", "right_click"].includes(
                    ev.name
                  ) &&
                  coord
                ) {
                  setMarkerSeq((s) => {
                    const id = s + 1;
                    setMarkers((ms) => [
                      ...ms.filter((m) => m.turn === ev.turn),
                      { x: Number(coord[0]), y: Number(coord[1]), turn: ev.turn, id },
                    ]);
                    return id;
                  });
                }
              }
              setEvents((prev) => [...prev, ev]);
            } catch {}
          }
        }
      } catch (err: unknown) {
        if ((err as { name?: string }).name !== "AbortError") {
          setEvents((prev) => [...prev, { type: "error", message: String(err) }]);
        }
      } finally {
        setRunning(false);
      }
    })();
  };

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: "smooth" });
  }, [events]);

  // Clear click markers when new screenshot arrives
  useEffect(() => {
    setMarkers([]);
  }, [screenshot]);

  const nonScreenshotEvents = events.filter((e) => e.type !== "screenshot");

  return (
    <div className="cu-overlay">
      <div className="cu-modal">
        {/* Header */}
        <div className="cu-header">
          <div className="cu-title">
            <span className="cu-icon">🖥</span>
            <span>Computer Use</span>
            <span className="cu-subtitle">Gemini 3.5 Flash · visual remediation</span>
          </div>
          <div className="cu-controls">
            {!running && (
              <button className="cu-btn cu-btn-run" onClick={start}>
                {events.length === 0 ? "▶ Run Demo" : "↺ Restart"}
              </button>
            )}
            {running && (
              <span className="cu-status-badge">
                <span className="cu-spin">⟳</span> Running…
              </span>
            )}
            <button className="cu-btn cu-btn-close" onClick={onClose}>
              ✕
            </button>
          </div>
        </div>

        {/* Body */}
        <div className="cu-body">
          {/* Screenshot panel */}
          <div className="cu-screenshot-panel">
            {screenshot ? (
              <div className="cu-screenshot-wrap">
                <img
                  className="cu-screenshot"
                  src={`data:image/png;base64,${screenshot}`}
                  alt="Dashboard screenshot"
                />
                {/* Click markers */}
                {markers.map((m) => (
                  <div
                    key={m.id}
                    className="cu-marker"
                    style={{
                      left: `${(m.x / VIEWPORT_W) * 100}%`,
                      top: `${(m.y / VIEWPORT_H) * 100}%`,
                    }}
                  />
                ))}
              </div>
            ) : (
              <div className="cu-screenshot-empty">
                {running ? (
                  <>
                    <span className="cu-spin-lg">⟳</span>
                    <span>Loading dashboard…</span>
                  </>
                ) : (
                  <span className="cu-hint">Click "Run Demo" to start</span>
                )}
              </div>
            )}
            <div className="cu-screenshot-label">
              Gemini 3.5 Flash sees this screen
            </div>
          </div>

          {/* Action log */}
          <div className="cu-log-panel" ref={logRef}>
            <div className="cu-log-title">Agent Actions</div>
            {nonScreenshotEvents.length === 0 && (
              <div className="cu-log-empty">
                {running ? "Waiting for model…" : "No events yet"}
              </div>
            )}
            {nonScreenshotEvents.map((ev, i) => (
              <LogEntry key={i} ev={ev} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function LogEntry({ ev }: { ev: CUEvent }) {
  if (ev.type === "reasoning") {
    return (
      <div className="cu-log-entry cu-reasoning">
        <span className="cu-log-turn">T{ev.turn}</span>
        <div className="cu-log-text">{ev.text}</div>
      </div>
    );
  }
  if (ev.type === "action") {
    const coord =
      (ev.args.coordinate as number[] | undefined) ||
      (ev.args.x !== undefined ? [ev.args.x, ev.args.y] : null);
    return (
      <div className="cu-log-entry cu-action">
        <span className="cu-log-turn">T{ev.turn}</span>
        <span className="cu-action-icon">⚡</span>
        <span className="cu-action-name">{ev.name}</span>
        {coord && (
          <span className="cu-action-coord">
            ({Math.round(Number(coord[0]))}, {Math.round(Number(coord[1]))})
          </span>
        )}
        {ev.args.text != null && (
          <span className="cu-action-text">"{String(ev.args.text)}"</span>
        )}
        <span className="cu-action-result">{ev.result}</span>
      </div>
    );
  }
  if (ev.type === "done") {
    return (
      <div className="cu-log-entry cu-done">
        ✓ Session complete · {ev.turns} turn{ev.turns !== 1 ? "s" : ""}
      </div>
    );
  }
  if (ev.type === "error") {
    return (
      <div className="cu-log-entry cu-error">
        ✕ {ev.message}
      </div>
    );
  }
  return null;
}
