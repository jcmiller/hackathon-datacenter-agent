import { useEffect, useState } from "react";
import type {
  Fleet,
  Incident,
  Meta,
  SourceBadge,
  TelemetryWindow,
} from "./types";
import {
  loadFleet,
  loadMeta,
  loadTelemetry,
} from "./data";
import { TopBar } from "./components/TopBar";
import { IncidentFeed } from "./components/IncidentFeed";
import { FleetHeatmap } from "./components/FleetHeatmap";
import { TelemetryStrip } from "./components/TelemetryStrip";
import { AgentTriage } from "./components/AgentTriage";
import { ComputerUsePanel } from "./components/ComputerUse";
import { SelfImprovement } from "./components/SelfImprovement";

export function App() {
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [fleet, setFleet] = useState<Fleet | null>(null);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedGpuId, setSelectedGpuId] = useState<string | null>(null);
  const [tele, setTele] = useState<TelemetryWindow | null>(null);
  // Honesty badge for the dashboard substrate (meta/fleet/telemetry/incidents).
  // Sourced from the NAMED provenance SSE event (bead h7w contract), with the
  // /api/meta payload as a fallback when the stream hasn't reported yet.
  const [source, setSource] = useState<SourceBadge | null>(null);

  // Collapsible panels state
  const [feedCollapsed, setFeedCollapsed] = useState(false);
  const [triageCollapsed, setTriageCollapsed] = useState(false);
  const [cuOpen, setCuOpen] = useState(false);

  // initial load — load initial topology and agent run definitions
  useEffect(() => {
    Promise.all([loadFleet(), loadMeta()])
      .then(([fl, mt]) => {
        // Start with a clean healthy cluster where all cells are initialized as active/idle
        // to clearly demonstrate the live incoming incident stream turning cells red!
        const healthyCells = fl.cells.map(c => ({
          ...c,
          status: (c.status === "fault" ? "active" : c.status) as "fault" | "active" | "idle"
        }));

        setFleet({
          ...fl,
          cells: healthyCells,
          faulted: 0
        });
        setMeta(mt);
        // Fallback badge from the meta payload until the SSE provenance event lands.
        setSource((prev) =>
          prev ?? {
            dataSource: mt.dataSource ?? fl.dataSource ?? "unavailable",
            provenance: mt.provenance ?? fl.provenance ?? null,
          },
        );
      })
      .catch((e) => console.error("dashboard load failed", e));

    // Connect to live SSE stream for real-time incidents!
    const es = new EventSource('/api/incidents');

    // NAMED provenance event (bead h7w): the stream emits the data-source badge on
    // its OWN event channel so onmessage stays Incident-only and back-compatible.
    // We MUST read it via addEventListener — onmessage never sees a named event.
    es.addEventListener('provenance', (event) => {
      try {
        const payload = JSON.parse((event as MessageEvent).data) as SourceBadge;
        setSource(payload);
      } catch {
        /* malformed provenance frame — keep the meta-derived fallback badge */
      }
    });

    // Unnamed message frames are Incidents (every one dereferences incident.gpu).
    es.onmessage = (event) => {
      const incident = JSON.parse(event.data) as Incident;

      setIncidents((prev) => {
        if (prev.some((i) => i.id === incident.id)) return prev;
        const next = [incident, ...prev];
        // Auto-select the first incident that arrives so the agent thinking animation triggers
        if (prev.length === 0) {
          setSelectedId(incident.id);
          setSelectedGpuId(`${incident.gpu.node}-${incident.gpu.idx}`);
        }
        return next;
      });

      // Turn the matching GPU cell Red in real-time!
      setFleet((prevFleet) => {
        if (!prevFleet) return null;
        const cells = prevFleet.cells.map((cell) => {
          if (cell.node === incident.gpu.node && cell.idx === incident.gpu.idx) {
            return {
              ...cell,
              status: "fault" as const,
              xid: incident.xid,
            };
          }
          return cell;
        });
        return {
          ...prevFleet,
          cells,
          faulted: cells.filter(c => c.status === 'fault').length,
        };
      });
    };

    return () => {
      es.close();
    };
  }, []);

  // load telemetry for the selected incident
  useEffect(() => {
    if (!selectedId) return;
    setTele(null);
    loadTelemetry(selectedId).then(setTele).catch(() => setTele(null));
  }, [selectedId]);

  // clicking a heatmap cell selects its incident if one exists for that GPU.
  // For a healthy/idle GPU we highlight the cell but don't fabricate a telemetry
  // window — real per-GPU DCGM history comes from the backend; until there's an
  // incident for this GPU the telemetry strip stays in its empty state.
  const selectGpu = (gid: string) => {
    setSelectedGpuId(gid);
    const match = incidents.find((i) => `${i.gpu.node}-${i.gpu.idx}` === gid);
    if (match) {
      setSelectedId(match.id);
    } else {
      setSelectedId(null);
      setTele(null);
    }
  };

  // Calculate dynamic grid template columns based on collapsed states
  const gridStyle = {
    gridTemplateColumns: `${feedCollapsed ? "" : "300px "}1fr${triageCollapsed ? "" : " 420px"}`,
  };

  return (
    <div className="app">
      <TopBar
        meta={meta}
        incidents={incidents}
        fleet={fleet}
        source={source}
        feedCollapsed={feedCollapsed}
        setFeedCollapsed={setFeedCollapsed}
        triageCollapsed={triageCollapsed}
        setTriageCollapsed={setTriageCollapsed}
        onComputerUse={() => setCuOpen(true)}
      />
      {cuOpen && <ComputerUsePanel onClose={() => setCuOpen(false)} />}
      <div className="cols" style={gridStyle}>
        {!feedCollapsed && (
          <IncidentFeed
            incidents={incidents}
            selectedId={selectedId}
            onSelect={(id) => {
              setSelectedId(id);
              const match = incidents.find((i) => i.id === id);
              if (match) {
                setSelectedGpuId(`${match.gpu.node}-${match.gpu.idx}`);
              }
            }}
          />
        )}
        <div className="center">
          <FleetHeatmap
            fleet={fleet}
            selectedGpu={selectedGpuId}
            onSelectGpu={selectGpu}
          />
          <TelemetryStrip tele={tele} />
          <SelfImprovement />
        </div>
        {!triageCollapsed && (
          <AgentTriage
            incidentId={selectedId}
            incidentData={selectedId ? incidents.find(i => i.id === selectedId) ?? null : null}
          />
        )}
      </div>
    </div>
  );
}
