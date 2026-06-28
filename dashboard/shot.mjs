import { chromium } from "playwright";

const url = process.argv[2] || "http://localhost:4317/";
const out = process.argv[3] || "/tmp/dash.png";

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });

const errors = [];
page.on("console", (m) => {
  if (m.type() === "error" || m.type() === "warning")
    errors.push(`[${m.type()}] ${m.text()}`);
});
page.on("pageerror", (e) => errors.push(`[pageerror] ${e.message}`));

await page.goto(url, { waitUntil: "networkidle" });
await page.waitForSelector(".gpu-cell", { timeout: 8000 }).catch(() => {});
// let the agent reasoning replay finish + disposition render
await page.waitForTimeout(7000);

const counts = await page.evaluate(() => ({
  incidents: document.querySelectorAll(".inc").length,
  gpuCells: document.querySelectorAll(".gpu-cell").length,
  faultCells: document.querySelectorAll(".gpu-cell.pulse").length,
  sparklines: document.querySelectorAll(".spark").length,
  agentEvents: document.querySelectorAll(".ev").length,
  hasDisposition: !!document.querySelector(".disp"),
  dispositionTag:
    document.querySelector(".disp .tag")?.textContent ?? null,
}));

await page.screenshot({ path: out, fullPage: false });
await browser.close();

console.log("DOM counts:", JSON.stringify(counts, null, 0));
console.log("console errors/warnings:", errors.length);
errors.slice(0, 20).forEach((e) => console.log("  " + e));
console.log("screenshot ->", out);
