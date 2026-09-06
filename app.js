const toast = document.querySelector("#toast");
const refreshButton = document.querySelector("#refreshButton");
const chartPrice = document.querySelector("#chartPrice");
const currentPrice = document.querySelector("#currentPrice");
const budget = document.querySelector(".budget-card .metric-value");
const API_BASE = "http://localhost:8000";
const performanceButton = document.querySelector("#loadPerformance");
const performanceContent = document.querySelector("#performanceContent");
const tickerForm = document.querySelector("#tickerForm");
const tickerInput = document.querySelector("#tickerInput");
const tickerPresets = document.querySelectorAll(".ticker-preset");
const rangeButtons = document.querySelectorAll(".range");
const trendChart = document.querySelector("#trendChart");
const chartArea = document.querySelector("#chartArea");
const chartLine = document.querySelector("#chartLine");
const chartPoint = document.querySelector("#chartPoint");
const todayLine = document.querySelector("#todayLine");
const todayLabel = document.querySelector("#todayLabel");
const chartStartLabel = document.querySelector("#chartStartLabel");
const chartEndLabel = document.querySelector("#chartEndLabel");
let selectedTicker = tickerInput.value;
let selectedRange = "1D";

function showToast(message) {
  toast.firstChild.textContent = message + " ";
  toast.classList.add("show");
  window.setTimeout(() => toast.classList.remove("show"), 2400);
}

async function refreshLivePrediction() {
  refreshButton.disabled = true;
  refreshButton.innerHTML = "<span>↻</span> Analyzing…";
  try {
    const response = await fetch(
      `${API_BASE}/predict/live/${encodeURIComponent(selectedTicker)}`,
    );
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(
        payload.detail || "The prediction service rejected the request.",
      );
    }
    const formatPrice = (value) =>
      Number(value).toLocaleString("en-US", {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      });
    chartPrice.textContent = formatPrice(payload.expected_price);
    currentPrice.textContent = formatPrice(payload.market_data.latest_price);
    document.querySelector(".price-label").textContent =
      `${payload.ticker} · forecast`;
    document.querySelector(".confidence").innerHTML =
      `Current price <strong id="currentPrice">${formatPrice(payload.market_data.latest_price)}</strong>`;
    document.querySelector(".forecast-callout strong").textContent =
      `Hybrid model forecast: ${payload.direction === "up" ? "continued upside" : payload.direction === "down" ? "downside pressure" : "range-bound movement"}`;
    document.querySelector(".forecast-callout p").innerHTML =
      `Forecast price <b>${formatPrice(payload.expected_price)}</b> (${(payload.predicted_return * 100).toFixed(2)}%). Current price ${formatPrice(payload.market_data.latest_price)} as of ${payload.market_data.as_of}.`;
    await loadTrendChart(selectedRange);
    showToast(`Live ${payload.ticker} prediction loaded`);
  } catch (error) {
    showToast(error.message || "Prediction service unavailable");
  } finally {
    refreshButton.disabled = false;
    refreshButton.innerHTML = "<span>↻</span> Refresh analysis";
  }
}

refreshButton.addEventListener("click", refreshLivePrediction);

function renderTrendChart(payload) {
  const points = payload.prices;
  if (!points.length) {
    throw new Error("No price history was returned for this view.");
  }
  const width = 760;
  const height = 250;
  const chartTop = 18;
  const chartBottom = 224;
  const values = points.map((point) => Number(point.close));
  const min = Math.min(...values);
  const max = Math.max(...values);
  const padding = Math.max((max - min) * 0.12, max * 0.002);
  const low = min - padding;
  const high = max + padding;
  const x = (index) =>
    points.length === 1 ? width : (index / (points.length - 1)) * width;
  const y = (value) =>
    chartBottom -
    ((value - low) / (high - low || 1)) * (chartBottom - chartTop);
  const coordinates = points.map((point, index) => [
    x(index),
    y(Number(point.close)),
  ]);
  const linePath = coordinates
    .map(
      ([pointX, pointY], index) =>
        `${index ? "L" : "M"}${pointX.toFixed(1)} ${pointY.toFixed(1)}`,
    )
    .join(" ");
  const areaPath = `${linePath} V${chartBottom} H0 Z`;
  chartLine.setAttribute("d", linePath);
  chartArea.setAttribute("d", areaPath);
  const [lastX, lastY] = coordinates.at(-1);
  chartPoint.setAttribute("cx", lastX);
  chartPoint.setAttribute("cy", lastY);
  todayLine.setAttribute("x1", lastX);
  todayLine.setAttribute("x2", lastX);
  todayLabel.setAttribute("x", Math.max(0, lastX - 22));
  trendChart.setAttribute(
    "aria-label",
    `${payload.ticker} ${payload.range} daily closing price trend`,
  );
  const formatDate = (value) =>
    new Date(`${value}T00:00:00`).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
    });
  chartStartLabel.textContent = formatDate(points[0].date);
  chartEndLabel.textContent = formatDate(points.at(-1).date);
}

async function loadTrendChart(range = selectedRange) {
  const response = await fetch(
    `${API_BASE}/prices/live/${encodeURIComponent(selectedTicker)}?range=${range}`,
  );
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(
      payload.detail || "The trend data service rejected the request.",
    );
  }
  renderTrendChart(payload);
}

function selectTicker(ticker) {
  selectedTicker = ticker.trim().toUpperCase();
  tickerInput.value = selectedTicker;
  tickerPresets.forEach((preset) => {
    preset.classList.toggle("active", preset.dataset.ticker === selectedTicker);
  });
}

tickerForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const ticker = tickerInput.value.trim().toUpperCase();
  if (!ticker || !/^[A-Z0-9.-]+$/.test(ticker)) {
    showToast("Enter a valid ticker symbol");
    return;
  }
  selectTicker(ticker);
  refreshLivePrediction();
});

tickerPresets.forEach((preset) => {
  preset.addEventListener("click", () => {
    selectTicker(preset.dataset.ticker);
    refreshLivePrediction();
  });
});

function formatPercent(value) {
  const percent = (Number(value) * 100).toFixed(2);
  return `${percent > 0 ? "+" : ""}${percent}%`;
}

function renderPerformance(payload) {
  const strategyLabels = {
    model: "Hermes model",
    previous_day: "Previous-day return",
    buy_and_hold: "Buy and hold",
    five_day_model: "Hermes five-day model",
    direction_classifier: "Direction classifier",
  };
  const cards = Object.entries(payload.summary)
    .map(([key, metrics]) => {
      const isModel = key === "model";
      return `
        <div class="performance-card ${isModel ? "featured" : ""}">
          <div class="performance-card-top">
            <strong>${strategyLabels[key]}</strong>
            ${isModel ? '<span class="view-pill bullish">Primary</span>' : ""}
          </div>
          <div class="performance-stat">
            <span>Directional accuracy</span>
            <strong>${(Number(metrics.average_directional_accuracy) * 100).toFixed(1)}%</strong>
          </div>
          <div class="performance-stat">
            <span>Average MAE</span>
            <strong>${Number(metrics.average_mae).toFixed(4)}</strong>
          </div>
          <div class="performance-stat">
            <span>Average return</span>
            <strong class="${metrics.average_cumulative_return >= 0 ? "positive" : "negative"}">${formatPercent(metrics.average_cumulative_return)}</strong>
          </div>
          <div class="performance-stat">
            <span>Average drawdown</span>
            <strong class="negative">${formatPercent(metrics.average_max_drawdown)}</strong>
          </div>
          <div class="performance-stat">
            <span>Annualized volatility</span>
            <strong>${formatPercent(metrics.average_annualized_volatility)}</strong>
          </div>
          ${
            key === "direction_classifier"
              ? `<div class="performance-stat"><span>Brier score</span><strong>${Number(metrics.average_brier_score).toFixed(4)}</strong></div>`
              : ""
          }
        </div>
      `;
    })
    .join("");

  const tickerRows = Object.entries(payload.results)
    .map(([ticker, result]) => {
      const model = result.strategies.model;
      return `
        <div class="performance-ticker">
          <strong>${ticker}</strong>
          <span>${model.observations} observations</span>
          <span>${(Number(model.directional_accuracy) * 100).toFixed(1)}% directional accuracy</span>
          <span class="${model.cumulative_return >= 0 ? "positive" : "negative"}">${formatPercent(model.cumulative_return)}</span>
        </div>
      `;
    })
    .join("");

  const errorCount = Object.keys(payload.errors || {}).length;
  performanceContent.innerHTML = `
    <div class="performance-cards">${cards}</div>
    <div class="performance-tickers">
      <div class="performance-tickers-head">
        <span>SYMBOL</span><span>OBSERVATIONS</span><span>MODEL ACCURACY</span><span>MODEL RETURN</span>
      </div>
      ${tickerRows}
    </div>
    <p class="performance-note">
      ${Object.keys(payload.results).length} tickers evaluated over ${payload.period}.
      ${errorCount ? `${errorCount} ticker${errorCount === 1 ? "" : "s"} could not be evaluated.` : "All requested tickers returned valid history."}
    </p>
  `;
}

async function loadPerformance() {
  performanceButton.disabled = true;
  performanceButton.textContent = "Running backtest…";
  performanceContent.innerHTML =
    '<div class="performance-empty"><span class="loading-spinner"></span><div><strong>Evaluating historical predictions…</strong><p>This may take a moment while each walk-forward window is trained.</p></div></div>';
  try {
    const basket = Array.from(
      new Set([
        selectedTicker,
        "SPY",
        "QQQ",
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "TSLA",
      ]),
    ).join(",");
    const response = await fetch(
      `${API_BASE}/backtest/live?tickers=${encodeURIComponent(basket)}&period=5y`,
    );
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(
        payload.detail?.message || payload.detail || "Backtest request failed.",
      );
    }
    renderPerformance(payload);
    showToast("Model performance loaded");
  } catch (error) {
    performanceContent.innerHTML = `<div class="performance-empty error-state"><span>!</span><div><strong>Performance data unavailable</strong><p>${error.message || "Start the Hermes API and try again."}</p></div></div>`;
  } finally {
    performanceButton.disabled = false;
    performanceButton.innerHTML = "Refresh performance <span>→</span>";
  }
}

performanceButton.addEventListener("click", loadPerformance);

rangeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    if (button.dataset.range === selectedRange) return;
    document.querySelector(".range.active")?.classList.remove("active");
    button.classList.add("active");
    selectedRange = button.dataset.range;
    loadTrendChart(selectedRange)
      .then(() => showToast(`${selectedTicker} ${selectedRange} trend loaded`))
      .catch((error) => showToast(error.message || "Trend data unavailable"));
  });
});

document.querySelector("#editBudget").addEventListener("click", () => {
  const current = budget.textContent.replace(/[$,]/g, "");
  const next = window.prompt("Set your available investment budget", current);
  if (next !== null && Number.isFinite(Number(next)) && Number(next) >= 0) {
    budget.textContent = `$${Number(next).toLocaleString("en-US")}`;
    showToast("Budget updated");
  }
});

document.querySelectorAll(".add-button").forEach((button) => {
  button.addEventListener("click", () => {
    button.textContent = "✓";
    button.style.color = "var(--green)";
    button.style.borderColor = "var(--green)";
    showToast(
      `${button.closest(".stock-row").querySelector(".asset strong").textContent} added to watchlist`,
    );
  });
});

refreshLivePrediction();
