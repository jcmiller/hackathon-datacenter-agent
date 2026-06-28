// Fake SSE: replays a canned agent reasoning trace with realistic pacing so the
// live "agent is thinking" feel works before the real /api/triage SSE exists.
// Drop-in shape matches what an EventSource consumer would do.
import type { AgentEvent } from "../types";

// delay (ms) before each event type is revealed — mimics tool latency + thinking
const DELAY: Record<AgentEvent["type"], number> = {
  user: 250,
  tool_call: 700,
  observation: 900,
  disposition: 1100,
};

export interface ReplayHandle {
  cancel: () => void;
}

export function replayAgentRun(
  events: AgentEvent[],
  onEvent: (ev: AgentEvent, index: number) => void,
  onDone?: () => void,
): ReplayHandle {
  let cancelled = false;
  const timers: ReturnType<typeof setTimeout>[] = [];

  let acc = 0;
  events.forEach((ev, i) => {
    acc += DELAY[ev.type] ?? 600;
    const t = setTimeout(() => {
      if (cancelled) return;
      onEvent(ev, i);
      if (i === events.length - 1) onDone?.();
    }, acc);
    timers.push(t);
  });

  return {
    cancel: () => {
      cancelled = true;
      timers.forEach(clearTimeout);
    },
  };
}
