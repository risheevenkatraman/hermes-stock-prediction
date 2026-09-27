import { expect, test } from "@playwright/test";
import { demoResearch } from "../lib/demo";

test("preview, ticker selection, forecasts and accessible chart data", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Design preview", exact: true }).click();
  await expect(page.getByText("All prices and forecasts shown are synthetic examples.", { exact: false })).toBeVisible();
  await page.getByRole("textbox", { name: "Search ticker" }).fill("NVDA");
  await page.getByRole("button", { name: "Search", exact: true }).click();
  await expect(page.getByRole("heading", { name: "NVIDIA", exact: true }).first()).toBeVisible();
  await page.getByLabel("Forecast horizon").selectOption("5");
  await expect(page.getByText("+1.50% estimated")).toBeVisible();
  await page.getByRole("button", { name: "1W", exact: true }).click();
  await page.getByText("View chart data", { exact: true }).click();
  await expect(page.getByRole("table")).toBeVisible();
  await expect(page.getByRole("row")).toHaveCount(6);
  await page.getByRole("button", { name: "Long-term view" }).click();
  await expect(page.getByText("A longer horizon needs a broader picture.")).toBeVisible();
});

test("guest profile and watchlist persist without pretending to be an account", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Personalize your workspace" }).click();
  const dialog = page.getByRole("dialog");
  await dialog.getByRole("spinbutton").fill("2500");
  await dialog.getByRole("textbox", { name: "Your financial goal" }).fill("Build a long-term investment fund");
  await dialog.getByRole("button", { name: "Save preferences" }).click();
  await expect(dialog).not.toBeVisible();
  await expect(page.getByText("$2,500.00", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Remove MSFT from watchlist" }).click();
  await page.reload();
  await expect(page.getByText("$2,500.00", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Remove MSFT from watchlist" })).toHaveCount(0);
  await page.getByRole("button", { name: "Sign in", exact: true }).click();
  await expect(page.getByRole("dialog")).toContainText("Accounts are not connected");
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).not.toBeVisible();
});

test("live errors do not show synthetic prices and layout stays within viewport", async ({ page }) => {
  await page.route("**/v1/research/**", route => route.fulfill({ status: 404, json: { detail: "Research has not been published for this ticker yet." }, headers: { "access-control-allow-origin": "*" } }));
  await page.goto("/");
  await page.getByRole("button", { name: "Research", exact: true }).click();
  await expect(page.locator(".market-panel").getByRole("alert")).toContainText("Research has not been published");
  await expect(page.getByText("Illustrative forecast", { exact: true })).toHaveCount(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
});

test("late ticker response cannot replace the latest selection", async ({ page }) => {
  await page.route("**/v1/research/**", async route => {
    const ticker = route.request().url().split("/").at(-1);
    if (ticker === "AAPL") await new Promise(resolve => setTimeout(resolve, 600));
    await route.fulfill({ status: 404, json: { detail: `No snapshot for ${ticker}` }, headers: { "access-control-allow-origin": "*" } });
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Research", exact: true }).click();
  await page.getByRole("textbox", { name: "Search ticker" }).fill("NVDA");
  await page.getByRole("button", { name: "Search", exact: true }).click();
  await expect(page.locator(".market-panel").getByRole("alert")).toHaveText("No snapshot for NVDA");
  await page.waitForTimeout(750);
  await expect(page.locator(".market-panel").getByRole("alert")).toHaveText("No snapshot for NVDA");
});

test("published research marks stale snapshots and unavailable horizons", async ({ page }) => {
  const fixture = demoResearch("AAPL");
  fixture.market_data.source = "Test provider";
  fixture.market_data.price_type = "Adjusted daily close";
  fixture.forecasts = fixture.forecasts.filter(item => [1, 5].includes(item.horizon));
  fixture.stale = true;
  await page.route("**/v1/research/**", route => route.fulfill({ json: fixture, headers: { "access-control-allow-origin": "*" } }));
  await page.goto("/");
  await page.getByRole("button", { name: "Research", exact: true }).click();
  await expect(page.getByText("STALE SNAPSHOT", { exact: true })).toBeVisible();
  await expect(page.getByText("Test provider", { exact: true })).toBeVisible();
  await page.getByLabel("Forecast horizon").selectOption("3");
  await expect(page.getByText("This horizon is not published")).toBeVisible();
  await expect(page.getByText("Not available", { exact: true })).toBeVisible();
});
