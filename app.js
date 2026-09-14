"use strict";
const $ = (id) => document.getElementById(id);
const validTicker = (value) => /^[A-Z0-9.-]{1,10}$/.test(value);
const finite = (value) => typeof value === "number" && Number.isFinite(value);
const number = (value, digits = 2) =>
  finite(value)
    ? value.toLocaleString("en-US", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      })
    : "Not available";
const percent = (value, signed = false) =>
  finite(value)
    ? `${signed && value > 0 ? "+" : ""}${number(value * 100)}%`
    : "Not available";
const points = (value, signed = false) =>
  finite(value)
    ? `${signed && value > 0 ? "+" : ""}${number(value * 100)} pp`
    : "Not available";
function sessionDate(value) {
  if (!value) return "Unknown session";
  const date = new Date(`${String(value).slice(0, 10)}T12:00:00Z`);
  return Number.isNaN(date.getTime())
    ? "Unknown session"
    : date.toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
        timeZone: "UTC",
      });
}
function timestamp(value) {
  const date = new Date(value);
  return !value || Number.isNaN(date.getTime())
    ? "Unknown time"
    : date.toLocaleString();
}
function readStorage(key, fallback) {
  try {
    return JSON.parse(localStorage.getItem(key)) ?? fallback;
  } catch {
    return fallback;
  }
}
function writeStorage(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch {
    notify(
      "Browser storage is unavailable. Changes will last only for this visit.",
    );
    return false;
  }
}
function serverURL(value) {
  const url = new URL(value);
  if (
    !["http:", "https:"].includes(url.protocol) ||
    url.username ||
    url.password ||
    url.search ||
    url.hash
  )
    throw new Error(
      "Enter an HTTP or HTTPS server URL without credentials, a query, or a fragment.",
    );
  return url.href.replace(/\/$/, "");
}
const defaultAPI =
  window.HERMES_CONFIG?.apiBase ||
  (location.protocol === "file:" || ["3000", "5173"].includes(location.port)
    ? "http://localhost:8000"
    : location.origin);
let apiBase;
try {
  apiBase = serverURL(readStorage("hermes.apiBase.v1", defaultAPI));
} catch {
  apiBase = defaultAPI;
}
const storedTicker = readStorage("hermes.ticker.v1", "SPY");
let selectedTicker =
  typeof storedTicker === "string" && validTicker(storedTicker)
    ? storedTicker
    : "SPY";
let savedTickers = readStorage("hermes.watchlist.v1", []);
savedTickers = Array.isArray(savedTickers)
  ? [
      ...new Set(
        savedTickers.filter((s) => typeof s === "string" && validTicker(s)),
      ),
    ].slice(0, 30)
  : [];
let selectedRange = "1M",
  chartPoints = [],
  chartGeometry = null,
  lastBacktest = null,
  newsConfigured = false,
  toastTimer;
const requests = new Map();
function notify(message) {
  clearTimeout(toastTimer);
  $("toast").textContent = message;
  $("toast").classList.add("show");
  toastTimer = setTimeout(() => $("toast").classList.remove("show"), 4500);
}
function status(id, message, state = "idle") {
  $(id).textContent = message;
  $(id).dataset.state = state;
}
function color(node, value) {
  node.classList.toggle("positive", finite(value) && value > 0);
  node.classList.toggle("negative", finite(value) && value < 0);
}
function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
function errorText(payload) {
  const detail = payload?.detail ?? payload;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail))
    return detail.map((item) => item.msg || "Invalid request").join("; ");
  return typeof detail?.message === "string"
    ? detail.message
    : "The server could not complete this request.";
}
function cancel(key) {
  requests.get(key)?.controller.abort();
  requests.delete(key);
}
function cancelAll() {
  [...requests.keys()].forEach(cancel);
}
async function request(
  key,
  path,
  { method = "GET", body, timeout = 180000 } = {},
) {
  cancel(key);
  const controller = new AbortController(),
    entry = { controller };
  requests.set(key, entry);
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeout);
  try {
    const response = await fetch(`${apiBase}${path}`, {
      method,
      signal: controller.signal,
      headers:
        body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    let payload;
    try {
      payload = await response.json();
    } catch {
      throw new Error(
        "The server returned an unreadable response. Check the server URL.",
      );
    }
    if (requests.get(key) !== entry)
      throw new DOMException("Superseded", "AbortError");
    if (!response.ok) throw new Error(errorText(payload));
    return payload;
  } catch (error) {
    if (requests.get(key) !== entry)
      throw new DOMException("Superseded", "AbortError");
    if (timedOut)
      throw new Error(
        "The request timed out. Try a smaller backtest or check the server.",
      );
    if (error instanceof TypeError)
      throw new Error(
        "Cannot reach the API. Check Connection settings and that the Hermes server is running.",
      );
    throw error;
  } finally {
    clearTimeout(timer);
    if (requests.get(key) === entry) requests.delete(key);
  }
}
function report(id, error) {
  if (error.name !== "AbortError") status(id, error.message, "error");
}
async function checkHealth() {
  $("connectionText").textContent = "Checking API";
  $("checkConnection").dataset.state = "checking";
  try {
    const data = await request("health", "/health", { timeout: 8000 });
    if (data.status !== "ok" || data.service !== "hermes-prediction-api")
      throw new Error("This address is not the Hermes API.");
    $("connectionText").textContent = "API connected";
    $("checkConnection").dataset.state = "online";
    $("checkConnection").title =
      `Connected to ${apiBase}. Market and news availability are shown separately.`;
  } catch (error) {
    if (error.name === "AbortError") return;
    $("connectionText").textContent = "API unavailable";
    $("checkConnection").dataset.state = "offline";
    $("checkConnection").title = error.message;
  }
}
function renderWatchlist() {
  $("watchlist").replaceChildren();
  $("savedCount").textContent = savedTickers.length;
  if (!savedTickers.length)
    $("watchlist").append(
      element("p", "Save a ticker to return to it here.", "empty-saved"),
    );
  for (const ticker of savedTickers) {
    const row = element("div", undefined, "saved-row"),
      choose = element("button", ticker),
      remove = element("button", "\u00d7");
    choose.type = remove.type = "button";
    choose.setAttribute("aria-label", `Analyze saved ticker ${ticker}`);
    choose.addEventListener("click", () => selectTicker(ticker));
    remove.setAttribute("aria-label", `Remove ${ticker} from saved tickers`);
    remove.addEventListener("click", () => {
      savedTickers = savedTickers.filter((symbol) => symbol !== ticker);
      writeStorage("hermes.watchlist.v1", savedTickers);
      renderWatchlist();
    });
    row.append(choose, remove);
    $("watchlist").append(row);
  }
  const saved = savedTickers.includes(selectedTicker);
  $("saveTicker").textContent = saved ? "Saved ticker" : "+ Save ticker";
  $("saveTicker").setAttribute("aria-pressed", String(saved));
}
function resetTickerPanels() {
  ["forecast", "direction", "chart", "news", "scan", "import"].forEach(cancel);
  $("forecastContent").hidden =
    $("directionContent").hidden =
    $("chartContainer").hidden =
      true;
  $("closeValue").textContent = "\u2014";
  $("closeAsOf").textContent = "No market data loaded.";
  $("windowChange").textContent = "";
  $("priceRows").replaceChildren();
  $("newsArticles").replaceChildren();
  $("newsCoverage").textContent = "";
  $("priceDetails").open = false;
  chartPoints = [];
  newsConfigured = false;
  $("scanNews").disabled = true;
  $("newsImport").disabled = false;
  $("reloadDirection").disabled = false;
}
function selectTicker(ticker) {
  const symbol = ticker.trim().toUpperCase();
  if (!validTicker(symbol)) {
    notify("Enter a valid ticker, up to 10 letters, digits, dots, or hyphens.");
    return;
  }
  const previous = selectedTicker;
  selectedTicker = symbol;
  $("tickerInput").value = symbol;
  $("selectedSymbol").textContent = symbol;
  $("newsSubtitle").textContent = `Saved articles for ${symbol}.`;
  if ($("backtestTickers").value === previous)
    $("backtestTickers").value = symbol;
  writeStorage("hermes.ticker.v1", symbol);
  document
    .querySelectorAll("[data-ticker]")
    .forEach((button) =>
      button.setAttribute(
        "aria-pressed",
        String(button.dataset.ticker === symbol),
      ),
    );
  resetTickerPanels();
  renderWatchlist();
  void Promise.allSettled([
    loadChart(),
    loadForecast(),
    loadDirection(),
    loadNews(),
  ]);
}
async function loadForecast() {
  const symbol = selectedTicker;
  $("forecastContent").hidden = true;
  status(
    "forecastStatus",
    `Training the return models for ${symbol}...`,
    "loading",
  );
  try {
    const data = await request(
      "forecast",
      `/predict/live/${encodeURIComponent(symbol)}`,
    );
    if (symbol !== selectedTicker) return;
    if (!finite(data.expected_price) || !finite(data.predicted_return))
      throw new Error("The API returned an incomplete forecast.");
    $("forecastPrice").textContent = number(data.expected_price);
    $("forecastChange").textContent =
      `${percent(data.predicted_return, true)} estimated next-session return`;
    color($("forecastChange"), data.predicted_return);
    $("forecastAsOf").textContent =
      `Based on ${symbol} close ${number(data.market_data?.latest_price)} on ${sessionDate(data.market_data?.as_of)}.`;
    $("fiveDayReturn").textContent = percent(
      data.five_day?.predicted_return,
      true,
    );
    color($("fiveDayReturn"), data.five_day?.predicted_return);
    $("forecastMae").textContent = points(data.validation?.mae);
    $("forecastAccuracy").textContent = percent(
      data.validation?.directional_accuracy,
    );
    $("deepWeight").textContent = percent(data.validation?.deep_weight);
    $("trainingRows").textContent = number(data.training_rows, 0);
    $("forecastEvaluation").textContent =
      `These holdout metrics use ${number(data.validation?.evaluation_rows, 0)} observations, not the full walk-forward test below.`;
    $("forecastContent").hidden = false;
    status("forecastStatus", "");
  } catch (error) {
    report("forecastStatus", error);
  }
}
async function loadDirection() {
  const symbol = selectedTicker;
  $("directionContent").hidden = true;
  $("reloadDirection").disabled = true;
  status(
    "directionStatus",
    `Checking ${symbol} against always-up...`,
    "loading",
  );
  try {
    const data = await request(
      "direction",
      `/predict/direction/live/${encodeURIComponent(symbol)}`,
    );
    if (symbol !== selectedTicker) return;
    const evaluation = data.evaluation;
    if (
      !evaluation ||
      !finite(evaluation.accuracy) ||
      !finite(evaluation.always_up_accuracy)
    )
      throw new Error(
        "Direction validation is not available in this response.",
      );
    $("directionPolicy").textContent =
      data.policy === "always_up"
        ? "Always-up fallback"
        : data.direction === "up"
          ? "Up"
          : "Down or flat";
    $("directionProbability").textContent =
      `Up probability ${percent(data.up_probability)} (uncalibrated)`;
    $("directionAccuracy").textContent = percent(evaluation.accuracy);
    $("baselineAccuracy").textContent = percent(evaluation.always_up_accuracy);
    $("directionRows").textContent =
      `${number(evaluation.observations, 0)} evaluation observations`;
    const difference = evaluation.accuracy - evaluation.always_up_accuracy;
    $("accuracyDifference").textContent = points(difference, true);
    color($("accuracyDifference"), difference);
    $("directionConclusion").textContent =
      difference > 0
        ? "Higher accuracy than always-up on this holdout. A broader independent test is still needed before claiming an advantage."
        : difference < 0
          ? "The direction model underperformed always-up on this holdout. This result does not support a predictive advantage."
          : "The selected policy matched always-up on this holdout. No accuracy improvement was measured.";
    $("directionNews").textContent =
      data.news_status === "trained"
        ? `News contribution to this prediction: ${percent(data.news_weight)}. ${number(data.news_training_days, 0)} news-covered training days.`
        : `News is not contributing yet: ${number(data.news_training_days, 0)} of the required ${number(data.minimum_news_training_days, 0)} news-covered training days. The classifier uses price features.`;
    $("directionDates").textContent =
      `Evaluation: ${sessionDate(data.evaluation_start)} to ${sessionDate(data.evaluation_end)}. Forecast cutoff: ${timestamp(data.as_of)}. ${number(data.articles_after_cutoff, 0)} archived articles arrived after this cutoff and were excluded.`;
    $("directionContent").hidden = false;
    status("directionStatus", "");
  } catch (error) {
    report("directionStatus", error);
  } finally {
    if (symbol === selectedTicker && !requests.has("direction"))
      $("reloadDirection").disabled = false;
  }
}

async function loadChart() {
  const symbol = selectedTicker,
    range = selectedRange;
  $("chartContainer").hidden = true;
  $("priceRows").replaceChildren();
  $("closeValue").textContent = "\u2014";
  $("windowChange").textContent = "";
  $("closeAsOf").textContent = "Waiting for daily prices.";
  status("chartStatus", `Loading ${symbol} daily closes...`, "loading");
  try {
    const data = await request(
      "chart",
      `/prices/live/${encodeURIComponent(symbol)}?range=${range}`,
      { timeout: 60000 },
    );
    if (symbol !== selectedTicker || range !== selectedRange) return;
    if (
      !Array.isArray(data.prices) ||
      !data.prices.length ||
      data.prices.some((p) => !finite(p.close) || p.close <= 0 || !p.date)
    )
      throw new Error("No valid daily prices were returned.");
    chartPoints = data.prices;
    const first = chartPoints[0],
      last = chartPoints.at(-1),
      change = last.close / first.close - 1;
    $("closeValue").textContent = number(last.close);
    $("windowChange").textContent =
      `${percent(change, true)} over ${chartPoints.length} closes`;
    color($("windowChange"), change);
    $("closeAsOf").textContent =
      `${sessionDate(last.date)} | ${data.market_data?.source || "Market-data provider"} | Adjusted daily prices`;
    $("chartTitle").textContent =
      `${symbol} adjusted daily closes from ${sessionDate(first.date)} to ${sessionDate(last.date)}`;
    $("chartStart").textContent = sessionDate(first.date);
    $("chartEnd").textContent = sessionDate(last.date);
    for (const point of [...chartPoints].reverse()) {
      const row = element("tr");
      row.append(
        element("td", sessionDate(point.date)),
        element("td", number(point.close)),
      );
      $("priceRows").append(row);
    }
    drawChart();
    $("chartContainer").hidden = false;
    status("chartStatus", "");
  } catch (error) {
    report("chartStatus", error);
  }
}
function svg(tag, attrs, text) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attrs).forEach(([key, value]) =>
    node.setAttribute(key, String(value)),
  );
  if (text !== undefined) node.textContent = text;
  return node;
}
function drawChart() {
  const values = chartPoints.map((point) => point.close),
    min = Math.min(...values),
    max = Math.max(...values);
  const padding = Math.max((max - min) * 0.15, max * 0.002),
    low = min - padding,
    high = max + padding;
  const x = (index) =>
    chartPoints.length === 1
      ? 365
      : 8 + (index * 724) / (chartPoints.length - 1);
  const y = (value) => 225 - ((value - low) * 210) / (high - low);
  chartGeometry = { x, y };
  const coordinates = chartPoints.map(
    (point, i) => `${x(i).toFixed(2)},${y(point.close).toFixed(2)}`,
  );
  $("chartLine").setAttribute("d", `M${coordinates.join(" L")}`);
  $("chartArea").setAttribute(
    "d",
    `M${x(0)},225 L${coordinates.join(" L")} L${x(chartPoints.length - 1)},225 Z`,
  );
  $("chartGrid").replaceChildren();
  for (let i = 0; i < 4; i++) {
    const value = low + ((high - low) * i) / 3,
      height = y(value);
    $("chartGrid").append(
      svg("line", { x1: 0, x2: 733, y1: height, y2: height }),
      svg("text", { x: 747, y: height + 4 }, number(value)),
    );
  }
  $("chartScrubber").max = chartPoints.length - 1;
  $("chartScrubber").value = chartPoints.length - 1;
  inspectChart(chartPoints.length - 1);
}
function inspectChart(index) {
  if (!chartPoints[index] || !chartGeometry) return;
  const point = chartPoints[index],
    x = chartGeometry.x(index),
    y = chartGeometry.y(point.close);
  $("chartCursor").setAttribute("x1", x);
  $("chartCursor").setAttribute("x2", x);
  $("chartCursor").setAttribute("y1", 10);
  $("chartCursor").setAttribute("y2", 225);
  $("chartPoint").setAttribute("cx", x);
  $("chartPoint").setAttribute("cy", y);
  $("chartReadout").textContent =
    `${sessionDate(point.date)} | ${number(point.close)}`;
  $("chartScrubber").setAttribute(
    "aria-valuetext",
    $("chartReadout").textContent,
  );
}
function safeLink(url, text) {
  try {
    const parsed = new URL(url);
    if (!["http:", "https:"].includes(parsed.protocol))
      return element("span", text);
    const link = element("a", text);
    link.href = parsed.href;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    return link;
  } catch {
    return element("span", text);
  }
}
async function loadNews() {
  const symbol = selectedTicker;
  $("newsArticles").replaceChildren();
  status("newsStatus", `Reading the ${symbol} archive...`, "loading");
  try {
    const data = await request(
      "news",
      `/news/${encodeURIComponent(symbol)}?limit=100`,
      { timeout: 30000 },
    );
    if (symbol !== selectedTicker) return;
    newsConfigured = data.provider_configured === true;
    $("scanNews").disabled = !newsConfigured || requests.has("scan");
    $("scanNews").title = newsConfigured
      ? "Fetch company news from Alpha Vantage"
      : "Alpha Vantage is not configured on the server";
    const articles = Array.isArray(data.articles) ? data.articles : [];
    const setup = newsConfigured
      ? "Use Scan latest news to update the archive."
      : "Live scans are not configured. Set ALPHAVANTAGE_API_KEY on the server, or import an article archive.";
    status(
      "newsStatus",
      articles.length
        ? `Showing ${articles.length} of ${number(data.article_count, 0)} archived articles. ${newsConfigured ? "" : setup}`
        : `No saved news for ${symbol}. ${setup}`,
    );
    $("newsCoverage").textContent = data.first_available_at
      ? `Archive availability: ${timestamp(data.first_available_at)} to ${timestamp(data.last_available_at)}. This does not establish complete provider coverage.`
      : "No historical availability records have been saved for this ticker.";
    for (const article of articles) {
      const row = element("li");
      row.append(safeLink(article.url, article.title || "Untitled article"));
      if (article.summary) row.append(element("p", article.summary, "summary"));
      const meta = element("div", undefined, "news-meta"),
        tone = !finite(article.sentiment)
          ? "Tone unavailable"
          : article.sentiment > 0.15
            ? "Positive tone"
            : article.sentiment < -0.15
              ? "Negative tone"
              : "Neutral tone";
      const badge = element("span", tone, "tone");
      color(badge, article.sentiment);
      meta.append(
        element("span", article.source || "Unknown source"),
        element("span", timestamp(article.published_at)),
        badge,
      );
      row.append(meta);
      $("newsArticles").append(row);
    }
  } catch (error) {
    if (error.name !== "AbortError") {
      newsConfigured = false;
      $("scanNews").disabled = true;
    }
    report("newsStatus", error);
  }
}
async function scanNews() {
  const symbol = selectedTicker;
  $("scanNews").disabled = true;
  status("newsStatus", `Scanning ${symbol} news...`, "loading");
  try {
    const result = await request(
      "scan",
      `/news/refresh/${encodeURIComponent(symbol)}`,
      { method: "POST", timeout: 60000 },
    );
    if (symbol !== selectedTicker) return;
    notify(`${number(result.inserted, 0)} new articles saved for ${symbol}.`);
    await Promise.allSettled([loadNews(), loadDirection()]);
  } catch (error) {
    report("newsStatus", error);
  } finally {
    if (symbol === selectedTicker && !requests.has("scan"))
      $("scanNews").disabled = !newsConfigured;
  }
}
async function importNews(file) {
  if (!file) return;
  const symbol = selectedTicker;
  $("newsImport").disabled = true;
  status(
    "newsStatus",
    "Validating and importing the article archive...",
    "loading",
  );
  try {
    if (file.size > 10000000)
      throw new Error("Use a JSON archive smaller than 10 MB per import.");
    let payload;
    try {
      payload = JSON.parse(await file.text());
    } catch {
      throw new Error("This file is not valid JSON.");
    }
    if (
      !Array.isArray(payload.articles) ||
      !payload.articles.length ||
      payload.articles.length > 5000
    )
      throw new Error(
        "The JSON must contain an articles list with 1 to 5,000 records.",
      );
    if (symbol !== selectedTicker) return;
    const data = await request("import", "/news/import", {
      method: "POST",
      body: payload,
    });
    if (symbol !== selectedTicker) return;
    notify(
      `${number(data.inserted, 0)} articles imported; ${number(data.duplicates, 0)} duplicates skipped.`,
    );
    await Promise.allSettled([loadNews(), loadDirection()]);
  } catch (error) {
    report("newsStatus", error);
  } finally {
    if (symbol === selectedTicker) {
      $("newsImport").disabled = false;
      $("newsImport").value = "";
    }
  }
}
function metricTable(caption, headers, rows) {
  const wrapper = element("div", undefined, "table-scroll"),
    table = element("table"),
    head = element("thead"),
    tr = element("tr"),
    body = element("tbody");
  table.append(element("caption", caption));
  headers.forEach((label) => {
    const th = element("th", label);
    th.scope = "col";
    tr.append(th);
  });
  head.append(tr);
  for (const values of rows) {
    const row = element("tr");
    if (values[0].includes("baseline")) row.className = "baseline-row";
    values.forEach((value) => row.append(element("td", value)));
    body.append(row);
  }
  table.append(head, body);
  wrapper.append(table);
  return wrapper;
}
function renderBacktest(data) {
  $("backtestResults").replaceChildren();
  const labels = {
    hybrid_model: "Hybrid return model",
    model: "Statistical return model",
    deep_model: "Neural return model",
    previous_day: "Previous-day return baseline",
  };
  for (const [symbol, result] of Object.entries(data.results || {})) {
    const block = element("section");
    block.append(element("h4", symbol, "result-symbol"));
    block.append(
      element(
        "p",
        `History requested: ${data.period}. Latest source close: ${sessionDate(result.market_data?.as_of)}.`,
        "muted",
      ),
    );
    const strategies = result.strategies || {};
    const rows = Object.entries(labels)
      .filter(([key]) => strategies[key])
      .map(([key, label]) => {
        const item = strategies[key];
        return [
          label,
          percent(item.directional_accuracy),
          points(item.mae),
          number(item.observations, 0),
        ];
      });
    const baseline = strategies.buy_and_hold;
    if (baseline) {
      rows.push([
        "Always-up baseline",
        percent(baseline.directional_accuracy),
        "Not applicable",
        number(baseline.observations, 0),
      ]);
      rows.push([
        "Zero-return baseline",
        "Not applicable",
        points(baseline.mae),
        number(baseline.observations, 0),
      ]);
    }
    block.append(
      metricTable(
        "Next-session return models",
        ["Model", "Sign accuracy", "Return MAE", "Observations"],
        rows,
      ),
    );
    if (strategies.direction_classifier) {
      const item = strategies.direction_classifier;
      block.append(
        metricTable(
          "Original price direction classifier",
          ["Direction accuracy", "Brier score", "Observations"],
          [
            [
              percent(item.directional_accuracy),
              number(item.brier_score, 4),
              number(item.observations, 0),
            ],
          ],
        ),
      );
    }
    if (strategies.five_day_model) {
      const item = strategies.five_day_model;
      block.append(
        metricTable(
          "Five-session forecast (different horizon)",
          ["Sign accuracy", "Return MAE", "Observations"],
          [
            [
              percent(item.directional_accuracy),
              points(item.mae),
              number(item.observations, 0),
            ],
          ],
        ),
      );
    }
    $("backtestResults").append(block);
  }
  for (const [symbol, message] of Object.entries(data.errors || {}))
    $("backtestResults").append(
      element(
        "p",
        `${symbol}: ${typeof message === "string" ? message : "Could not evaluate this ticker."}`,
        "result-error",
      ),
    );
}

async function runBacktest(event) {
  event.preventDefault();
  const tickers = [
    ...new Set(
      $("backtestTickers")
        .value.split(",")
        .map((value) => value.trim().toUpperCase())
        .filter(Boolean),
    ),
  ];
  if (!tickers.length || tickers.length > 20 || !tickers.every(validTicker)) {
    status(
      "backtestStatus",
      "Enter 1 to 20 valid comma-separated tickers.",
      "error",
    );
    return;
  }
  const period = $("backtestPeriod").value;
  $("runBacktest").disabled = true;
  $("exportBacktest").disabled = true;
  lastBacktest = null;
  $("backtestResults").replaceChildren();
  const started = Date.now();
  const progress = () =>
    status(
      "backtestStatus",
      `Evaluating ${tickers.join(", ")} over ${period}. ${Math.floor((Date.now() - started) / 1000)}s elapsed. Each historical prediction trains on earlier observations; larger runs can take several minutes.`,
      "loading",
    );
  progress();
  const timer = setInterval(progress, 1000);
  try {
    const data = await request(
      "backtest",
      `/backtest/live?tickers=${encodeURIComponent(tickers.join(","))}&period=${period}`,
      { timeout: 1800000 },
    );
    clearInterval(timer);
    if (!data.results || !Object.keys(data.results).length)
      throw new Error("No backtest results were returned.");
    lastBacktest = data;
    renderBacktest(data);
    $("exportBacktest").disabled = false;
    const errors = Object.keys(data.errors || {}).length;
    status(
      "backtestStatus",
      `${Object.keys(data.results).length} ticker(s) evaluated; ${errors} failed. Completed ${timestamp(new Date().toISOString())}.`,
      errors ? "error" : "idle",
    );
  } catch (error) {
    report("backtestStatus", error);
  } finally {
    clearInterval(timer);
    $("runBacktest").disabled = false;
  }
}
async function loadTraders(refresh = false) {
  $("refreshTraders").disabled = true;
  $("traderResults").replaceChildren();
  status(
    "traderStatus",
    refresh
      ? "Refreshing configured disclosure sources..."
      : "Checking disclosure sources...",
    "loading",
  );
  try {
    let pipeline;
    if (refresh)
      pipeline = (
        await request("traders", "/trader-pipeline/refresh", { method: "POST" })
      ).status;
    else
      pipeline = await request("traders", "/trader-pipeline/status", {
        timeout: 30000,
      });
    const configured = pipeline.configured_sources || [];
    if (!configured.length) {
      status(
        "traderStatus",
        "No disclosure sources are configured. Quiver Quantitative, Stockcircle, or TradingView provider endpoints must be configured on the server before records can be shown.",
      );
      return;
    }
    const errors = Object.entries(pipeline.provider_errors || {})
      .map(([name, message]) => `${name}: ${message}`)
      .join("; ");
    if (!pipeline.records) {
      status(
        "traderStatus",
        `No records available from ${configured.join(", ")}. ${errors || "Use Refresh disclosures to request the configured feeds."}`,
        errors ? "error" : "idle",
      );
      return;
    }
    const data = await request(
      "traders",
      "/recommendations/live?refresh=false&include_price_model=false&limit=20",
    );
    status(
      "traderStatus",
      `Source refresh: ${timestamp(data.updated_at)}. ${errors}`,
      errors ? "error" : "idle",
    );
    for (const item of data.recommendations || []) {
      const card = element("article", undefined, "trader-card"),
        top = element("div", undefined, "trader-top"),
        button = element("button", item.ticker);
      button.type = "button";
      button.addEventListener("click", () => {
        selectTicker(item.ticker);
        $("overview").scrollIntoView();
      });
      top.append(
        button,
        element("span", `${number(item.trader_count, 0)} disclosed traders`),
      );
      card.append(top);
      card.append(
        element(
          "p",
          `Recency-weighted buy/sell consensus: ${number(item.trader_signal, 2)} on a -1 to +1 scale. Sources: ${(item.sources || []).join(", ")}.`,
          "muted",
        ),
      );
      const list = element("ul");
      for (const trade of item.recent_trades || [])
        list.append(
          element(
            "li",
            `${trade.trader || "Unknown trader"} | ${trade.action || "Unknown action"} | ${sessionDate(trade.trade_date)} | ${trade.source || "Unknown source"}`,
          ),
        );
      card.append(list);
      $("traderResults").append(card);
    }
  } catch (error) {
    report("traderStatus", error);
  } finally {
    $("refreshTraders").disabled = false;
  }
}
$("tickerForm").addEventListener("submit", (event) => {
  event.preventDefault();
  selectTicker($("tickerInput").value);
});
document
  .querySelectorAll("[data-ticker]")
  .forEach((button) =>
    button.addEventListener("click", () => selectTicker(button.dataset.ticker)),
  );
$("saveTicker").addEventListener("click", () => {
  if (savedTickers.includes(selectedTicker))
    savedTickers = savedTickers.filter((ticker) => ticker !== selectedTicker);
  else if (savedTickers.length >= 30) {
    notify("You can save up to 30 tickers in this browser.");
    return;
  } else savedTickers.push(selectedTicker);
  writeStorage("hermes.watchlist.v1", savedTickers);
  renderWatchlist();
});
document.querySelectorAll("[data-range]").forEach((button) =>
  button.addEventListener("click", () => {
    selectedRange = button.dataset.range;
    document
      .querySelectorAll("[data-range]")
      .forEach((item) =>
        item.setAttribute("aria-pressed", String(item === button)),
      );
    void loadChart();
  }),
);
$("chartScrubber").addEventListener("input", (event) =>
  inspectChart(Number(event.target.value)),
);
$("priceChart").addEventListener("pointermove", (event) => {
  if (!chartPoints.length) return;
  const bounds = $("priceChart").getBoundingClientRect(),
    position = ((event.clientX - bounds.left) * 800) / bounds.width;
  const index = Math.max(
    0,
    Math.min(
      chartPoints.length - 1,
      Math.round(((position - 8) / 724) * (chartPoints.length - 1)),
    ),
  );
  $("chartScrubber").value = index;
  inspectChart(index);
});
$("reloadDirection").addEventListener("click", loadDirection);
$("scanNews").addEventListener("click", scanNews);
$("newsImport").addEventListener("change", (event) =>
  importNews(event.target.files[0]),
);
$("backtestForm").addEventListener("submit", runBacktest);
$("refreshTraders").addEventListener("click", () => loadTraders(true));
$("exportBacktest").addEventListener("click", () => {
  if (!lastBacktest) return;
  const blob = new Blob([JSON.stringify(lastBacktest, null, 2)], {
      type: "application/json",
    }),
    url = URL.createObjectURL(blob),
    anchor = element("a");
  anchor.href = url;
  anchor.download = `hermes-backtest-${lastBacktest.period}.json`;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
$("checkConnection").addEventListener("click", checkHealth);
$("openSettings").addEventListener("click", () => {
  $("apiUrl").value = apiBase;
  $("settingsStatus").textContent = "";
  $("settingsDialog").showModal();
});
$("openAccount").addEventListener("click", () =>
  $("accountDialog").showModal(),
);
document
  .querySelectorAll("[data-close]")
  .forEach((button) =>
    button.addEventListener("click", () => $(button.dataset.close).close()),
  );
$("settingsForm").addEventListener("submit", (event) => {
  event.preventDefault();
  try {
    const url = serverURL($("apiUrl").value.trim());
    cancelAll();
    apiBase = url;
    writeStorage("hermes.apiBase.v1", url);
    $("settingsDialog").close();
    $("backtestResults").replaceChildren();
    lastBacktest = null;
    $("exportBacktest").disabled = true;
    status(
      "backtestStatus",
      "Connection changed. Run a new backtest for this server.",
    );
    selectTicker(selectedTicker);
    void checkHealth();
    void loadTraders();
  } catch (error) {
    $("settingsStatus").textContent = error.message;
  }
});
const navigation = [...document.querySelectorAll(".nav-link")];
const observer = new IntersectionObserver(
  (entries) => {
    for (const entry of entries)
      if (entry.isIntersecting)
        navigation.forEach((link) =>
          link.classList.toggle("active", link.hash === `#${entry.target.id}`),
        );
  },
  { rootMargin: "-10% 0px -65% 0px", threshold: 0 },
);
["overview", "news", "performance", "traders"].forEach((id) =>
  observer.observe($(id)),
);
$("today").textContent = new Date().toLocaleDateString("en-US", {
  weekday: "short",
  month: "short",
  day: "numeric",
  year: "numeric",
});
$("today").dateTime = new Date().toISOString();
selectTicker(selectedTicker);
void checkHealth();
void loadTraders();
