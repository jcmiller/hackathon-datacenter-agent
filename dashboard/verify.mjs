// Rigorous headless verification of the dashboard draft.
// Asserts render + interactions (incident selection, telemetry swap, heatmap
// cell -> incident, agent replay -> disposition) and fails loudly on any
// console error or broken assertion. Saves screenshots for eyeballing.
import { chromium } from "playwright";
import { readFileSync } from "node:fs";

const BASE = process.argv[2] || "http://localhost:4320/";
const incidents = JSON.parse(
  readFileSync(new URL("./public/fixtures/incidents.json", import.meta.url)),
);

let failures = 0;
const ok = (cond, msg) => {
  console.log(`${cond ? "  ✓" : "  ✗ FAIL:"} ${msg}`);
  if (!cond) failures++;
};

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });

const consoleErrors = [];
page.on("console", (m) => m.type() === "error" && consoleErrors.push(m.text()));
page.on("pageerror", (e) => consoleErrors.push(`pageerror: ${e.message}`));

await page.goto(BASE, { waitUntil: "networkidle" });
await page.waitForSelector(".gpu-cell", { timeout: 8000 });

// ---- 1. static render ----
console.log("\n[1] render");
const counts = await page.evaluate(() => ({
  inc: document.querySelectorAll(".inc").length,
  cells: document.querySelectorAll(".gpu-cell").length,
  faults: document.querySelectorAll(".gpu-cell.pulse").length,
  sparks: document.querySelectorAll(".spark").length,
  active: document.querySelectorAll(".gpu-cell").length,
}));
ok(counts.inc === incidents.length, `incident cards = ${counts.inc} (expect ${incidents.length})`);
ok(counts.cells === 2344, `gpu cells = ${counts.cells} (expect 2344)`);
ok(counts.faults === 882, `pulsing fault cells = ${counts.faults} (expect 882)`);
ok(counts.sparks === 3, `telemetry sparklines = ${counts.sparks} (expect 3)`);
// heatmap should NOT be all-red: healthy/idle (non-fault) cells must exist
ok(counts.cells - counts.faults > 1000, `non-fault cells = ${counts.cells - counts.faults} (>1000, not a wall of red)`);

// helper: select incident card n, wait for THIS incident's agent run to render
// (guard against reading the previous run's stale disposition), then read state
async function selectIncident(n) {
  const id = incidents[n].id;
  await page.locator(".inc").nth(n).click();
  // wait until the agent stream's user line reflects the newly-selected incident
  await page.waitForFunction(
    (wantId) => {
      const u = document.querySelector(".ev.user .line");
      return !!u && u.textContent.includes(wantId);
    },
    id,
    { timeout: 12000 },
  );
  await page.waitForSelector(".disp .tag", { timeout: 12000 });
  return page.evaluate(() => ({
    selCount: document.querySelectorAll(".inc.sel").length,
    selText: document.querySelector(".inc.sel")?.textContent ?? "",
    userLine: document.querySelector(".ev.user .line")?.textContent ?? "",
    sparkVals: [...document.querySelectorAll(".spark .val")].map((e) =>
      e.textContent?.trim(),
    ),
    dispTag: document.querySelector(".disp .tag")?.textContent ?? "",
    teleGpu:
      [...document.querySelectorAll(".panel-title .faint")]
        .map((e) => e.textContent || "")
        .find((t) => /\d+\.\d+.*-\d/.test(t)) ?? "",
  }));
}

// ---- 2. default selection = hero (cascade) ----
console.log("\n[2] default hero selection");
const hero = incidents.find((i) => i.hero);
const s0 = await selectIncident(0);
ok(s0.selCount === 1, `exactly one incident selected (got ${s0.selCount})`);
ok(s0.userLine.includes(incidents[0].id), `agent stream references ${incidents[0].id}`);
ok(s0.dispTag.length > 0, `disposition rendered: "${s0.dispTag}"`);
ok(/escalate/i.test(s0.dispTag), `cascade hero escalates (got "${s0.dispTag}")`);
ok(!!hero, `hero incident exists in fixtures (${hero?.id})`);

// ---- 3. selecting another incident swaps telemetry + agent run ----
console.log("\n[3] incident switch updates telemetry + agent");
const s2 = await selectIncident(2); // INC-003, Xid 31
ok(s2.userLine.includes(incidents[2].id), `agent stream switched to ${incidents[2].id}`);
ok(
  JSON.stringify(s0.sparkVals) !== JSON.stringify(s2.sparkVals),
  `telemetry values changed between incidents (${s0.sparkVals} -> ${s2.sparkVals})`,
);
ok(s2.teleGpu.includes(incidents[2].gpu.node), `telemetry panel shows ${incidents[2].gpu.node}`);

// ---- 4. heatmap fault cell -> selects its incident ----
console.log("\n[4] heatmap cell click selects incident");
const heroGid = `${incidents[0].gpu.node}-${incidents[0].gpu.idx}`;
await page.locator(`.gpu-cell[title^="${heroGid} "]`).first().click();
await page.waitForSelector(".disp .tag", { timeout: 12000 });
const afterCell = await page.evaluate(
  () => document.querySelector(".inc.sel")?.textContent ?? "",
);
ok(afterCell.includes(`Xid ${incidents[0].xid}`), `clicking ${heroGid} cell selected its incident`);

// ---- 5. no console errors the whole run ----
console.log("\n[5] console");
ok(consoleErrors.length === 0, `console errors = ${consoleErrors.length}`);
consoleErrors.slice(0, 10).forEach((e) => console.log("     " + e));

// ---- 6. narrow viewport sanity (no crash, still renders) ----
console.log("\n[6] narrow viewport (1280)");
await page.setViewportSize({ width: 1280, height: 800 });
await page.waitForTimeout(400);
const narrowCells = await page.evaluate(() => document.querySelectorAll(".gpu-cell").length);
ok(narrowCells === 2344, `still renders all cells at 1280px (${narrowCells})`);
await page.screenshot({ path: "/tmp/dash_narrow.png" });

await page.setViewportSize({ width: 1600, height: 900 });
await selectIncident(0);
await page.screenshot({ path: "/tmp/dash_verified.png" });

await browser.close();
console.log(`\n${failures === 0 ? "ALL CHECKS PASSED" : failures + " CHECK(S) FAILED"}`);
process.exit(failures === 0 ? 0 : 1);
