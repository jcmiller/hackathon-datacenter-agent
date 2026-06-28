import { useEffect, useMemo, useState } from "react";
import type {
  AgentRuns,
  Fleet,
  Incident,
  Meta,
  TelemetryWindow,
} from "./types";
import {
  loadAgentRuns,
  loadFleet,
  loadMeta,
  loadTelemetry,
} from "./data";
import { TopBar } from "./components/TopBar";
import { IncidentFeed } from "./components/IncidentFeed";
import { FleetHeatmap } from "./components/FleetHeatmap";
import { TelemetryStrip } from "./components/TelemetryStrip";
import { AgentTriage } from "./components/AgentTriage";

export function App() {
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [fleet, setFleet] = useState<Fleet | null>(null);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [runs, setRuns] = useState<AgentRuns>({});
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [tele, setTele] = useState<TelemetryWindow | null>(null);

  // Collapsible panels state
  const [feedCollapsed, setFeedCollapsed] = useState(false);
  const [triageCollapsed, setTriageCollapsed] = useState(false);

  // initial load — load initial topology and agent run definitions
  useEffect(() => {
    Promise.all([loadFleet(), loadMeta(), loadAgentRuns()])
      .then(([fl, mt, rn]) => {
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
        setRuns(rn);
      })
      .catch((e) => console.error("fixture load failed", e));

    // Connect to live SSE stream for real-time incidents!
    const source = new EventSource('/api/incidents');
    source.onmessage = (event) => {
      const incident = JSON.parse(event.data) as Incident;
      
      setIncidents((prev) => {
        if (prev.some((i) => i.id === incident.id)) return prev;
        const next = [incident, ...prev];
        // Auto-select the first incident that arrives so the agent thinking animation triggers
        if (prev.length === 0) {
          setSelectedId(incident.id);
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
      source.close();
    };
  }, []);

  // load telemetry for the selected incident
  useEffect(() => {
    if (!selectedId) return;
    setTele(null);
    loadTelemetry(selectedId).then(setTele).catch(() => setTele(null));
  }, [selectedId]);

  const selected = useMemo(
    () => incidents.find((i) => i.id === selectedId) ?? null,
    [incidents, selectedId],
  );
  const selectedGpu = selected ? `${selected.gpu.node}-${selected.gpu.idx}` : null;

  // clicking a heatmap cell selects its incident if one exists for that GPU
  const selectGpu = (gid: string) => {
    const match = incidents.find((i) => `${i.gpu.node}-${i.gpu.idx}` === gid);
    if (match) setSelectedId(match.id);
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
        feedCollapsed={feedCollapsed}
        setFeedCollapsed={setFeedCollapsed}
        triageCollapsed={triageCollapsed}
        setTriageCollapsed={setTriageCollapsed}
      />
      <div className="cols" style={gridStyle}>
        {!feedCollapsed && (
          <IncidentFeed
            incidents={incidents}
            selectedId={selectedId}
            onSelect={setSelectedId}
          />
        )}
        <div className="center">
          <FleetHeatmap
            fleet={fleet}
            selectedGpu={selectedGpu}
            onSelectGpu={selectGpu}
          />
          <TelemetryStrip tele={tele} />
        </div>
        {!triageCollapsed && (
          <AgentTriage
            incidentId={selectedId}
            events={selectedId ? runs[selectedId] ?? null : null}
          />
        )}
      </div>
    </div>
  );
}
