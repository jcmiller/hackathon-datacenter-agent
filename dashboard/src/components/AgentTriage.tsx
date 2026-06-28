import { useEffect, useRef, useState } from "react";
import type { AgentEvent, Incident } from "../types";

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
  const streamRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    setShown([]);
    setError(null);
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
      }
    })();

    return () => ctrl.abort();
  }, [incidentId, incidentData]);

  useEffect(() => {
    streamRef.current?.scrollTo({ top: streamRef.current.scrollHeight });
  }, [shown]);

  const disposition = shown.find((e) => e.type === "disposition");

  return (
    <section className="col panel">
      <div className="panel-title">
        <span>Agent triage · ReAct</span>
        {incidentId && <span className="faint">{incidentId}</span>}
      </div>
      <div className="triage-body">
        {!incidentId ? (
          <div className="empty">select an incident — the agent will triage it live</div>
        ) : (
          <>
            <div className="stream" ref={streamRef}>
              {loading && shown.length === 0 && (
                <div className="empty">Activating Google ADK RCA Agent...</div>
              )}
              {error && (
                <div className="empty" style={{ color: "var(--crit)" }}>{error}</div>
              )}
              {shown
                .filter((e) => e.type !== "disposition")
                .map((ev, i) => (
                  <EventLine key={i} ev={ev} />
                ))}
              {loading && <div className="dim cursor" />}
            </div>
            {disposition && disposition.type === "disposition" && (
              <div className="disp fade-up">
                <span className="tag">{disposition.disposition.replace(/_/g, " ")}</span>
                <div className="action">{disposition.action}</div>
                {disposition.ticket && (
                  <div className="ticket">▸ ticket {disposition.ticket} opened</div>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </section>
  );
}

function EventLine({ ev }: { ev: AgentEvent }) {
  if (ev.type === "tool_call")
    return (
      <div className="ev tool fade-up">
        <div className="line">
          ▶ {ev.tool}
          <span className="faint">({ev.args})</span>
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
          ✎ <span className="faint">{ev.path}</span>
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
