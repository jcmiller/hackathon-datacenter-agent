import { useEffect, useRef, useState } from "react";
import type { AgentEvent, FeedbackOutcome, Incident, ModelCard } from "../types";

const DISP_META: Record<string, { label: string; cls: string }> = {
  PAGE_TECHNICIAN:  { label: "Page Technician",   cls: "disp-crit" },
  ESCALATE_TO_OPS:  { label: "Escalate to Ops",   cls: "disp-warn" },
  RESTART_AND_WATCH:{ label: "Restart & Watch",   cls: "disp-ok"   },
};

export function AgentTriage({
  incidentId,
  incidentData,
}: {
  incidentId: string | null;
  incidentData: Incident | null;
}) {
  const [shown, setShown] = useState<AgentEvent[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [feedback, setFeedback] = useState<FeedbackOutcome | null>(null);
  const [feedbackSent, setFeedbackSent] = useState(false);
  const [model, setModel] = useState<ModelCard | null>(null);
  const streamRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Fetch model card on mount
  useEffect(() => {
    fetch("/api/model").then(r => r.json()).then(d => {
      if (d.model) setModel(d.model);
    }).catch(() => {});
  }, []);

  // Elapsed timer while agent is running
  useEffect(() => {
    if (!loading) { setElapsed(0); return; }
    const id = setInterval(() => setElapsed((e) => e + 1), 1000);
    return () => clearInterval(id);
  }, [loading]);

  useEffect(() => {
    setShown([]);
    setError(null);
    setFeedback(null);
    setFeedbackSent(false);
    abortRef.current?.abort();
    if (!incidentId || !incidentData) return;

    setLoading(true);
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    (async () => {
      try {
        const res = await fetch("/api/triage", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(incidentData),
          signal: ctrl.signal,
        });
        if (!res.ok) throw new Error(`Triage backend error: ${res.statusText}`);

        const reader = res.body!.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop()!;
          for (const line of lines) {
            if (line.startsWith("data: ")) {
              try {
                const ev: AgentEvent = JSON.parse(line.slice(6));
                setShown((s) => [...s, ev]);
              } catch {}
            }
          }
        }
      } catch (err: any) {
        if (err.name !== "AbortError") {
          setError(err.message || "Failed to stream agent triage");
        }
      } finally {
        setLoading(false);
        // Refresh model card after triage completes (train_and_validate may have promoted)
        fetch("/api/model").then(r => r.json()).then(d => {
          if (d.model) setModel(d.model);
        }).catch(() => {});
      }
    })();

    return () => ctrl.abort();
  }, [incidentId, incidentData]);

  useEffect(() => {
    streamRef.current?.scrollTo({ top: streamRef.current.scrollHeight });
  }, [shown]);

  const sendFeedback = async (outcome: FeedbackOutcome) => {
    if (!incidentId || feedbackSent) return;
    setFeedback(outcome);
    try {
      await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ incident_id: incidentId, outcome }),
      });
      setFeedbackSent(true);
    } catch {}
  };

  const visible = shown.filter((e) => e.type !== "disposition");
  const disposition = shown.find((e) => e.type === "disposition");
  const lastVisible = visible[visible.length - 1];

  const awaitingTool = loading && lastVisible?.type === "tool_call";
  const thinking     = loading && (!lastVisible || lastVisible.type === "observation" || lastVisible.type === "file_update");
  const activeTool   = awaitingTool
    ? (lastVisible as Extract<AgentEvent, { type: "tool_call" }>).tool
    : null;

  const dispMeta = disposition?.type === "disposition"
    ? (DISP_META[disposition.disposition] ?? { label: disposition.disposition.replace(/_/g, " "), cls: "disp-ok" })
    : null;

  return (
    <section className="col panel">
      <div className="panel-title">
        <span>Agent triage · ReAct</span>
        <span className="faint">
          {loading ? `${elapsed}s` : incidentId ?? ""}
        </span>
      </div>

      {/* Model card — always visible when a model exists */}
      {model && (
        <div className="model-card">
          <span className="model-label">predictor</span>
          <span className="model-type">{model.model_type}</span>
          <span className="model-ver">v{model.version}</span>
          <span className="model-auc">AUC {model.val_auc.toFixed(3)}</span>
          <span className="model-n">{model.n_samples} samples</span>
        </div>
      )}

      {/* Phase status bar */}
      {loading && (
        <div className={`triage-status ${awaitingTool ? "status-tool" : "status-think"}`}>
          {awaitingTool ? (
            <>
              <span className="spin">⟳</span>
              <span>running <strong>{activeTool}</strong></span>
            </>
          ) : thinking && shown.length === 0 ? (
            <>
              <ThinkingDots />
              <span>connecting to Gemini 2.5 Flash</span>
            </>
          ) : (
            <>
              <ThinkingDots />
              <span>Gemini is reasoning</span>
            </>
          )}
        </div>
      )}

      <div className="triage-body">
        {!incidentId ? (
          <div className="empty">select an incident — the agent will triage it live</div>
        ) : (
          <>
            <div className="stream" ref={streamRef}>
              {error && (
                <div className="empty" style={{ color: "var(--crit)" }}>{error}</div>
              )}
              {visible.map((ev, i) => (
                <EventLine key={i} ev={ev} inFlight={i === visible.length - 1 && awaitingTool} />
              ))}
            </div>
            {disposition && dispMeta && disposition.type === "disposition" && (
              <div className={`disp fade-up ${dispMeta.cls}`}>
                <div className="disp-header">
                  <span className="tag">{dispMeta.label}</span>
                  {disposition.ticket && (
                    <span className="ticket">▸ {disposition.ticket}</span>
                  )}
                </div>
                <div className="action">{disposition.action}</div>

                {/* Outcome feedback */}
                <div className="feedback-row">
                  {feedbackSent ? (
                    <span className="feedback-sent">
                      {feedback === "confirmed" ? "✓ Confirmed" : feedback === "false_alarm" ? "✗ False alarm logged" : "? Noted as uncertain"}
                      {" "}— outcome saved to memory
                    </span>
                  ) : (
                    <>
                      <span className="feedback-label">Was this correct?</span>
                      <button className="fb-btn fb-ok"    onClick={() => sendFeedback("confirmed")}>✓ Confirmed</button>
                      <button className="fb-btn fb-bad"   onClick={() => sendFeedback("false_alarm")}>✗ False alarm</button>
                      <button className="fb-btn fb-unsure" onClick={() => sendFeedback("uncertain")}>? Uncertain</button>
                    </>
                  )}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </section>
  );
}

function ThinkingDots() {
  return (
    <span className="thinking-dots">
      <span>·</span><span>·</span><span>·</span>
    </span>
  );
}

function EventLine({ ev, inFlight }: { ev: AgentEvent; inFlight?: boolean }) {
  if (ev.type === "tool_call")
    return (
      <div className={`ev tool fade-up${inFlight ? " in-flight" : ""}`}>
        <div className="line">
          <span className="tool-arrow">{inFlight ? "⟳" : "✓"}</span>
          <span className="tool-name">{ev.tool}</span>
          <span className="faint tool-args">({ev.args})</span>
        </div>
      </div>
    );
  if (ev.type === "observation")
    return (
      <div className="ev obs fade-up">
        <div className="line">{ev.text}</div>
      </div>
    );
  if (ev.type === "file_update")
    return (
      <div className="ev file fade-up">
        <div className="line">
          <span className="file-icon">✎</span>
          <span className="file-path">{ev.path}</span>
          <span className="file-badge">sop written</span>
        </div>
        <pre className="file-entry">{JSON.stringify(ev.entry, null, 2)}</pre>
      </div>
    );
  if (ev.type === "user")
    return (
      <div className="ev user fade-up">
        <div className="role">incident</div>
        <div className="line">{ev.text}</div>
      </div>
    );
  return null;
}
