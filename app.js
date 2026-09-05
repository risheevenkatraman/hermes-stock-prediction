const toast = document.querySelector("#toast");
const refreshButton = document.querySelector("#refreshButton");
const chartPrice = document.querySelector("#chartPrice");
const budget = document.querySelector(".budget-card .metric-value");
const API_BASE = "http://localhost:8000";

function showToast(message) {
  toast.firstChild.textContent = message + " ";
  toast.classList.add("show");
  window.setTimeout(() => toast.classList.remove("show"), 2400);
}

async function refreshLivePrediction() {
  refreshButton.disabled = true;
  refreshButton.innerHTML = "<span>↻</span> Analyzing…";
  try {
    const response = await fetch(`${API_BASE}/predict/live/SPY`);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(
        payload.detail || "The prediction service rejected the request.",
      );
    }
    chartPrice.textContent = Number(
      payload.market_data.latest_price,
    ).toLocaleString("en-US", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
    document.querySelector(".price-label").textContent =
      `${payload.ticker} · live`;
    document.querySelector(".confidence strong").textContent =
      `${Math.round(payload.confidence * 100)}%`;
    document.querySelector(".forecast-callout strong").textContent =
      `Model forecast: ${payload.direction === "up" ? "continued upside" : payload.direction === "down" ? "downside pressure" : "range-bound movement"}`;
    document.querySelector(".forecast-callout p").innerHTML =
      `Next trading day estimate <b>${(payload.predicted_return * 100).toFixed(2)}%</b>. Data as of ${payload.market_data.as_of}.`;
    showToast(`Live ${payload.ticker} prediction loaded`);
  } catch (error) {
    showToast(error.message || "Prediction service unavailable");
  } finally {
    refreshButton.disabled = false;
    refreshButton.innerHTML = "<span>↻</span> Refresh analysis";
  }
}

refreshButton.addEventListener("click", refreshLivePrediction);

document.querySelectorAll(".range").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelector(".range.active").classList.remove("active");
    button.classList.add("active");
    const prices = {
      "1D": "5,521.94",
      "1W": "5,521.94",
      "1M": "5,487.11",
      "3M": "5,319.42",
    };
    chartPrice.textContent = prices[button.dataset.range];
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
