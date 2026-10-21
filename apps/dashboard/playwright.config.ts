import { defineConfig, devices } from "@playwright/test";

// End-to-end tests for the dashboard against the real gateway (no mocks): Playwright starts
// uvicorn and vite, or reuses them if they are already running (`pnpm dev` + the gateway).
//
//   pnpm --filter @metacore/dashboard test:e2e
//
// Specs are *.e2e.ts so vitest's default *.spec/*.test glob never picks them up.
const repoRoot = new URL("../../", import.meta.url).pathname;

export default defineConfig({
  testDir: "./e2e",
  testMatch: "*.e2e.ts",
  timeout: 60_000,
  expect: { timeout: 20_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: "http://127.0.0.1:5173",
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: [
    {
      command: "uv run --no-sync uvicorn gateway.main:app --app-dir services/gateway --port 8000",
      cwd: repoRoot,
      url: "http://127.0.0.1:8000/health",
      reuseExistingServer: true,
      timeout: 120_000,
    },
    {
      command: "pnpm exec vite --host 127.0.0.1 --port 5173",
      url: "http://127.0.0.1:5173",
      reuseExistingServer: true,
      timeout: 120_000,
    },
  ],
});
