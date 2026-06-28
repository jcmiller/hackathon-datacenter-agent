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
  loadIncidents,
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

  // initial load — default-select the hero (cascade) incident
  useEffect(() => {
    Promise.all([loadIncidents(), loadFleet(), loadMeta(), loadAgentRuns()])
      .then(([inc, fl, mt, rn]) => {
        setIncidents(inc);
        setFleet(fl);
        setMeta(mt);
        setRuns(rn);
        const hero = inc.find((i) => i.hero) ?? inc[0];
        if (hero) setSelectedId(hero.id);
      })
      .catch((e) => console.error("fixture load failed", e));
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
