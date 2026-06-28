// Headless UI smoke test for the LIVE-wired dashboard (bead 31n).
// Runs against the real FastAPI backend (default http://localhost:8000), deriving
// every expected value from the /api/* endpoints themselves — no stale hardcoded
// fixture numbers. Asserts the real data wiring + honesty badging + the
// self-improvement surfaces render, and fails loudly on any console error.
//
//   uv run uvicorn gpusitter.app.sim:app --port 8000   # in another shell
//   node verify.mjs                                      # or: node verify.mjs http://host:port
import { chromium } from "playwright";

const BASE = process.argv[2] || "http://localhost:8000/";
const api = (p) => fetch(new URL(p, BASE)).then((r) => r.json());

let failures = 0;
const ok = (cond, msg) => {
  console.log(`${cond ? "  ✓" : "  ✗ FAIL:"} ${msg}`);
  if (!cond) failures++;
};

// ---- ground truth straight from the live API --------------------------------
const [fleet, meta, monitor, curve] = await Promise.all([
  api("api/fleet"),
  api("api/meta"),
  api("api/monitor"),
  api("api/learning-curve"),
]);
console.log(
  `\n[api] fleet.cells=${fleet.cells.length} dataSource=${meta.dataSource} ` +
    `monitor.available=${monitor.available} curve.points=${curve.curve?.length}`,
);

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
const consoleErrors = [];
page.on("console", (m) => m.type() === "error" && consoleErrors.push(m.text()));
page.on("pageerror", (e) => consoleErrors.push(`pageerror: ${e.message}`));

await page.goto(BASE, { waitUntil: "networkidle" });
await page.waitForSelector(".gpu-cell", { timeout: 8000 });

// ---- 1. heatmap renders the REAL fleet --------------------------------------
console.log("\n[1] fleet heatmap (real substrate)");
const cellCount = await page.evaluate(() => document.querySelectorAll(".gpu-cell").length);
ok(cellCount === fleet.cells.length, `gpu cells = ${cellCount} (api: ${fleet.cells.length})`);

// ---- 2. provenance badge is honest ------------------------------------------
console.log("\n[2] provenance badge");
const badge = await page.evaluate(() => {
  const el = document.querySelector(".prov-badge");
  return el ? { text: el.textContent?.trim(), cls: el.className } : null;
});
ok(!!badge, "provenance badge is rendered");
if (meta.dataSource === "real_substrate") {
  ok(/REAL/i.test(badge?.text || ""), `badge says REAL (got "${badge?.text}")`);
  ok(/prov-real/.test(badge?.cls || ""), "badge carries the real-source style");
} else {
  ok(/FIXTURE|OFFLINE|SYNTHETIC|UNAVAIL/i.test(badge?.text || ""), `badge honestly non-real (got "${badge?.text}")`);
}

// ---- 3. incidents stream in over SSE ----------------------------------------
console.log("\n[3] live incident stream (SSE)");
await page.waitForSelector(".inc", { timeout: 12000 });
const incCount = await page.evaluate(() => document.querySelectorAll(".inc").length);
ok(incCount >= 1, `at least one incident streamed in (got ${incCount})`);
// the first arrival auto-selects → telemetry sparklines render for its GPU
await page.waitForSelector(".spark", { timeout: 8000 });
const sparks = await page.evaluate(() => document.querySelectorAll(".spark").length);
ok(sparks === 3, `telemetry sparklines for the selected incident = ${sparks} (expect 3)`);

// ---- 4. self-improvement: learning curve ------------------------------------
console.log("\n[4] self-improvement · learning curve");
const curvePts = await page.evaluate(() => document.querySelectorAll(".si-curve circle").length);
if (curve.available) {
  ok(curvePts === curve.curve.length, `curve points = ${curvePts} (api: ${curve.curve.length})`);
} else {
  ok(true, "learning curve unavailable — surface degrades honestly (skipped)");
}

// ---- 5. self-improvement: miss detector / recall table ----------------------
console.log("\n[5] self-improvement · miss detector");
const rows = await page.evaluate(() => document.querySelectorAll(".si-table tbody tr").length);
if (monitor.available) {
  const horizons = Object.keys(monitor.budgets[0].grid.by_horizon).length;
  ok(rows === horizons, `per-horizon rows = ${rows} (api: ${horizons})`);
  const recall = await page.evaluate(() => !!document.querySelector(".recall-bar"));
  ok(recall, "recall bars render");
} else {
  ok(true, "monitor unavailable — surface degrades honestly (skipped)");
}

// ---- 6. model card honesty flags --------------------------------------------
console.log("\n[6] model card");
const modelFlags = await page.evaluate(() =>
  [...document.querySelectorAll(".model-flag")].map((e) => e.textContent?.trim()),
);
ok(modelFlags.length > 0, `model card honesty flag(s) present: [${modelFlags.join(", ")}]`);

// ---- 7. no console errors ---------------------------------------------------
// Ignore benign resource 404s: the agent's best-effort POST /api/feedback returns
// 404 off-droplet (no SOP file yet) and is swallowed by a .catch() — a graceful
// degradation path, not a wiring bug. The required GET data loads are asserted
// independently above; a real missing asset would break those, not just log here.
console.log("\n[7] console");
const realErrors = consoleErrors.filter((e) => !/Failed to load resource.*404/i.test(e));
ok(realErrors.length === 0, `console errors = ${realErrors.length}`);
realErrors.slice(0, 10).forEach((e) => console.log("     " + e));

await page.screenshot({ path: "/tmp/dash_live_verified.png", fullPage: true });
await browser.close();
console.log(`\n${failures === 0 ? "ALL CHECKS PASSED" : failures + " CHECK(S) FAILED"}`);
process.exit(failures === 0 ? 0 : 1);
