import { useEffect, useRef, useState } from "react";
import type { AgentEvent } from "../types";
import { replayAgentRun } from "../lib/replay";

export function AgentTriage({
  incidentId,
  events,
}: {
  incidentId: string | null;
  events: AgentEvent[] | null;
}) {
  const [shown, setShown] = useState<AgentEvent[]>([]);
  const [done, setDone] = useState(false);
  const streamRef = useRef<HTMLDivElement>(null);

  // restart the replay whenever the selected incident changes
  useEffect(() => {
    setShown([]);
    setDone(false);
    if (!events) return;
    const handle = replayAgentRun(
      events,
      (ev) => setShown((s) => [...s, ev]),
      () => setDone(true),
    );
    return () => handle.cancel();
  }, [incidentId, events]);

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
              {shown
                .filter((e) => e.type !== "disposition")
                .map((ev, i) => (
                  <EventLine key={i} ev={ev} />
                ))}
              {!done && <div className="dim cursor" />}
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
  if (ev.type === "user")
    return (
      <div className="ev user fade-up">
        <div className="role">incident</div>
        <div className="line">{ev.text}</div>
      </div>
    );
  return null;
}
