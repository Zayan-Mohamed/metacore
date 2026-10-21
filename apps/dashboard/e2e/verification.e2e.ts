import { expect, test, type Page } from "@playwright/test";

// The Module 4 page against the real gateway and OpenDSS. Every verdict here is computed live.

async function waitForVerdict(page: Page) {
  // A verify call disables the controls while it runs.
  await expect(page.getByTestId("mode-certified")).toBeEnabled();
  await expect(page.getByTestId("verdict")).toBeVisible();
}

test.beforeEach(async ({ page }) => {
  await page.goto("/verification");
  await expect(page.getByRole("heading", { name: /Risk-Certified Verification/ })).toBeVisible();
  await waitForVerdict(page);
});

test("the fragile battery export passes the plain gate but not the certificate", async ({ page }) => {
  await expect(page.getByTestId("preset-fragile_export")).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByTestId("verdict")).toHaveText("REJECT");
  await expect(page.getByTestId("check-a")).toHaveText("APPROVE");
  await expect(page.getByTestId("k-of-n")).toContainText("/ 100");
  await expect(page.getByTestId("risk-interval")).toBeVisible();
  await expect(page.getByTestId("attribution")).toContainText("Eluvaitivu");
  await expect(page.getByTestId("causal-log")).toContainText("points come from Eluvaitivu load uncertainty");
  await expect(page.getByTestId("grounding")).toContainText("grounded in the topology");

  await page.getByTestId("mode-plain").click();
  await waitForVerdict(page);
  await expect(page.getByTestId("verdict")).toHaveText("APPROVE");
  await expect(page.getByTestId("risk-card")).toContainText("Plain gate (baseline B0)");
  await expect(page.getByTestId("upper-bound")).toHaveText("—");
});

test("a safe action is approved with a full 688-scenario certificate", async ({ page }) => {
  await page.getByTestId("preset-nominal_safe").click();
  await waitForVerdict(page);
  await expect(page.getByTestId("verdict")).toHaveText("APPROVE");
  await expect(page.getByTestId("k-of-n")).toHaveText("0 / 688");
  await expect(page.getByTestId("upper-bound")).toHaveText("1.00%");
  await expect(page.getByTestId("causal-log")).toContainText("0 of 688 likely next-6-hour conditions");
  await expect(page.getByTestId("attribution")).toHaveCount(0);
});

test("the Ditwah evening runs conformal PID in storm mode", async ({ page }) => {
  await page.getByTestId("preset-cyclone_survival").click();
  await waitForVerdict(page);
  await expect(page.getByTestId("hour-select")).toHaveValue("ditwah_evening");
  await expect(page.getByTestId("verdict")).toHaveText("APPROVE");
  await expect(page.getByTestId("pid-state")).toContainText("on (storm)");
  await expect(page.getByTestId("risk-card")).toContainText(/SEVERE|EXTREME/);
  await expect(page.getByTestId("risk-caveat")).toContainText("downscaled from CEB monthly totals");

  await page.getByTestId("pid-select").selectOption("off");
  await waitForVerdict(page);
  await expect(page.getByTestId("pid-state")).toHaveText("off");
});

test("a genset outage at the peak is rejected on adequacy", async ({ page }) => {
  await page.getByTestId("preset-adequacy_breach").click();
  await waitForVerdict(page);
  await expect(page.getByTestId("verdict")).toHaveText("REJECT");
  await expect(page.getByTestId("adequacy")).toContainText("INADEQUATE");
  await expect(page.getByTestId("causal-log")).toContainText("Generation inadequate at station NAI_PS");
});

test("the limit profile changes the bands the verdict is judged against", async ({ page }) => {
  await page.getByTestId("preset-nominal_safe").click();
  await waitForVerdict(page);
  await page.getByTestId("profile-select").selectOption("EN-50160");
  await waitForVerdict(page);
  await expect(page.getByText("Red outside 0.90–1.10 pu (EN-50160-v1)")).toBeVisible();
});

test("any recorded hour can be replayed through Module 1's snapshot", async ({ page }) => {
  await page.getByTestId("preset-nominal_safe").click();
  await waitForVerdict(page);
  await page.getByTestId("hour-input").fill("2025-03-14T18:00");
  await waitForVerdict(page);
  await expect(page.getByTestId("hour-select")).toHaveValue("custom");
  await expect(page.getByTestId("check-a")).toContainText(/APPROVE|REJECT|UNVERIFIED/);
  await expect(page.getByText("snapshot: module1")).toBeVisible();
});
