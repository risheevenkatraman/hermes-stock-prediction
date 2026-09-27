"use client";

import { useEffect, useState } from "react";
import { Hub } from "aws-amplify/utils";
import {
  ArrowDownRight,
  ArrowRight,
  ArrowUpRight,
  Bookmark,
  Check,
  ChevronDown,
  Compass,
  FlaskConical,
  Globe2,
  LayoutDashboard,
  LogOut,
  Plus,
  Search,
  Settings2,
  ShieldCheck,
  Sparkles,
  TrendingUp,
  X,
} from "lucide-react";
import { authConfigured, login, session, signOut } from "@/lib/auth";
import { getResearch, getWorkspace, saveWorkspace } from "@/lib/api";
import { demoResearch } from "@/lib/demo";
import {
  companies,
  defaultWorkspace,
  type Research,
  type Workspace,
} from "@/lib/types";
import PriceChart from "./chart";
import Modal from "./modal";
import ProfileForm from "./profile-form";

const money = (value: number) =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  }).format(value);
const guestKey = "hermes.guest-workspace.v3";
function readGuest(): Workspace {
  try {
    const value = JSON.parse(localStorage.getItem(guestKey) ?? "null");
    if (
      value &&
      Array.isArray(value.watchlist) &&
      value.watchlist.length <= 50 &&
      value.watchlist.every(
        (s: unknown) => typeof s === "string" && /^[A-Z0-9.-]{1,10}$/.test(s),
      ) &&
      value.profile &&
      typeof value.profile.goal === "string" &&
      value.profile.goal.length <= 200 &&
      Number.isFinite(value.profile.budget) &&
      value.profile.budget >= 0 &&
      ["short", "long"].includes(value.profile.horizon) &&
      ["conservative", "balanced", "growth"].includes(value.profile.risk)
    )
      return value;
  } catch {
    /* Restricted browser storage leaves an in-memory guest workspace. */
  }
  return structuredClone(defaultWorkspace);
}

export default function Dashboard() {
  const [workspace, setWorkspace] = useState<Workspace>(defaultWorkspace);
  const [account, setAccount] = useState<string | null>(null);
  const [accountReady, setAccountReady] = useState(false);
  const [workspaceReady, setWorkspaceReady] = useState(false);
  const [ticker, setTicker] = useState("AAPL");
  const [query, setQuery] = useState("");
  const [demo, setDemo] = useState(!process.env.NEXT_PUBLIC_API_URL);
  const [research, setResearch] = useState<Research | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [range, setRange] = useState("3M");
  const [horizon, setHorizon] = useState(1);
  const [showForecast, setShowForecast] = useState(true);
  const [view, setView] = useState<"short" | "long">("short");
  const [modal, setModal] = useState<"account" | "profile" | "method" | null>(
    null,
  );
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let active = true;
    let generation = 0;
    async function sync() {
      const version = ++generation;
      setWorkspaceReady(false);
      try {
        const current = await session();
        if (!active || version !== generation) return;
        setAccount(current?.name ?? null);
        const value = current ? await getWorkspace() : readGuest();
        if (active && version === generation) {
          setWorkspace(value);
          setWorkspaceReady(true);
        }
      } catch (error) {
        if (active && version === generation)
          setNotice(
            error instanceof Error
              ? error.message
              : "Could not load your account.",
          );
      } finally {
        if (active && version === generation) setAccountReady(true);
      }
    }
    void sync();
    const cancel = Hub.listen("auth", ({ payload }) => {
      if (
        ["signedIn", "signedOut", "signInWithRedirect"].includes(payload.event)
      )
        void sync();
      if (payload.event === "signInWithRedirect_failure")
        setNotice("Sign-in was not completed. Please try again.");
    });
    return () => {
      active = false;
      cancel();
    };
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError("");
    setResearch(null);
    if (demo) {
      setResearch(demoResearch(ticker));
      setLoading(false);
    } else
      getResearch(ticker, controller.signal)
        .then((value) => {
          if (!controller.signal.aborted) setResearch(value);
        })
        .catch((error) => {
          if (!controller.signal.aborted) setError(error.message);
        })
        .finally(() => {
          if (!controller.signal.aborted) setLoading(false);
        });
    return () => controller.abort();
  }, [ticker, demo]);

  async function update(value: Workspace) {
    if (!workspaceReady || saving) return;
    setSaving(true);
    try {
      if (account) await saveWorkspace(value);
      else {
        try {
          localStorage.setItem(guestKey, JSON.stringify(value));
        } catch {
          throw new Error(
            "Browser storage is unavailable. Your changes could not be saved.",
          );
        }
      }
      setWorkspace(value);
    } finally {
      setSaving(false);
    }
  }
  async function toggleTicker(symbol: string) {
    const existing = workspace.watchlist.includes(symbol);
    if (!existing && workspace.watchlist.length >= 50) {
      setNotice("Your watchlist can contain up to 50 stocks.");
      return;
    }
    try {
      await update({
        ...workspace,
        watchlist: existing
          ? workspace.watchlist.filter((item) => item !== symbol)
          : [...workspace.watchlist, symbol],
      });
    } catch (error) {
      setNotice(
        error instanceof Error
          ? error.message
          : "Could not save your watchlist.",
      );
    }
  }
  async function beginLogin(google = false) {
    try {
      await login(google);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Could not sign in.");
    }
  }
  const company = companies[ticker] ?? {
    name: ticker,
    sector: "US equity research",
    color: "stone",
  };
  const forecast = research?.forecasts.find((item) => item.horizon === horizon);
  const rows = research?.prices ?? [];
  const change = rows.length > 1 ? rows.at(-1)!.close - rows.at(-2)!.close : 0;
  const changePercent =
    rows.length > 1 ? (change / rows.at(-2)!.close) * 100 : 0;
  const saved = workspace.watchlist.includes(ticker);

  return (
    <div className="app-shell">
      <a href="#main" className="skip-link">
        Skip to dashboard
      </a>
      <aside className="sidebar">
        <a href="/" className="brand" aria-label="Hermes home">
          <span className="brand-mark">
            h<span>↗</span>
          </span>
          <span>
            hermes<span className="brand-period">.</span>
          </span>
        </a>
        <div className="workspace-label">YOUR INVESTING COMPANION</div>
        <nav aria-label="Main navigation">
          <a className="nav-item active" href="#main">
            <LayoutDashboard size={18} /> Overview <span className="nav-dot" />
          </a>
          <a className="nav-item" href="#watchlist">
            <Bookmark size={18} /> Watchlist{" "}
            <span className="nav-count">{workspace.watchlist.length}</span>
          </a>
          <a className="nav-item" href="#discover">
            <Compass size={18} /> Discover
          </a>
          <button className="nav-item" onClick={() => setModal("method")}>
            <FlaskConical size={18} /> Research & methodology
          </button>
        </nav>
        <div className="sidebar-note">
          <span className="orbit-mark">✳</span>
          <p>
            A little foresight.
            <br />
            <em>A broader perspective.</em>
          </p>
          <span>Built for the way you invest.</span>
        </div>
        <div className="sidebar-bottom">
          <button
            className="nav-item"
            onClick={() => setModal("profile")}
            disabled={!workspaceReady}
          >
            <Settings2 size={18} /> Investment preferences
          </button>
          <button className="account-card" onClick={() => setModal("account")}>
            <span className="avatar">
              {account ? account[0].toUpperCase() : "H"}
            </span>
            <span>
              <strong>{account ? "Your workspace" : "Guest workspace"}</strong>
              <small>
                {account ? "Account connected" : "Make room for your future"}
              </small>
            </span>
            <ChevronDown size={15} />
          </button>
        </div>
      </aside>

      <main id="main">
        <header className="topbar">
          <span className="breadcrumb">
            Workspace <span>/</span> <strong>Overview</strong>
          </span>
          <div className="topbar-actions">
            <span className="market-label">
              <Globe2 size={14} /> US markets
            </span>
            <button
              className="button small"
              onClick={() => setModal("account")}
            >
              {account ? "My account" : "Sign in"}
              <ArrowUpRight size={14} />
            </button>
          </div>
        </header>
        <div className="page-content">
          <section className="page-heading">
            <div>
              <div className="eyebrow">
                <span /> A MORE CONSIDERED WAY TO INVEST
              </div>
              <h1>
                Invest with <em>perspective.</em>
              </h1>
              <p>
                Your markets. Your ambitions. A clearer view of what comes next.
              </p>
            </div>
            <div className="mode-control" aria-label="Data source">
              <button aria-pressed={!demo} onClick={() => setDemo(false)}>
                Research
              </button>
              <button aria-pressed={demo} onClick={() => setDemo(true)}>
                Design preview
              </button>
            </div>
          </section>
          {demo && (
            <div className="preview-banner">
              <FlaskConical size={16} />
              <span>
                <strong>Design preview</strong> · All prices and forecasts shown
                are synthetic examples.
              </span>
              <button onClick={() => setDemo(false)}>
                Use research data <ArrowRight size={14} />
              </button>
            </div>
          )}
          {notice && (
            <div className="notice" role="status">
              <span>{notice}</span>
              <button
                aria-label="Dismiss notification"
                className="icon-button"
                onClick={() => setNotice("")}
              >
                <X size={16} />
              </button>
            </div>
          )}

          <section className="watchlist-section" id="watchlist">
            <div className="section-heading">
              <h2>
                Your watchlist{" "}
                <span>
                  {workspace.watchlist.length.toString().padStart(2, "0")}
                </span>
              </h2>
              <span className="muted tiny">
                {account ? "Synced to your account" : "Saved in this browser"}
              </span>
            </div>
            <div className="watchlist-strip">
              {workspace.watchlist.map((symbol) => {
                const info = companies[symbol];
                return (
                  <div
                    className={`watch-card ${ticker === symbol ? "selected" : ""}`}
                    key={symbol}
                  >
                    <button
                      className="watch-select"
                      onClick={() => setTicker(symbol)}
                      aria-pressed={ticker === symbol}
                    >
                      <span
                        className={`company-icon ${info?.color ?? "stone"}`}
                      >
                        {symbol.slice(0, 1)}
                      </span>
                      <span>
                        <strong>{symbol}</strong>
                        <small>{info?.name ?? "US stock"}</small>
                      </span>
                      <ArrowUpRight size={17} />
                    </button>
                    <button
                      className="watch-remove"
                      aria-label={`Remove ${symbol} from watchlist`}
                      disabled={saving || !workspaceReady}
                      onClick={() => void toggleTicker(symbol)}
                    >
                      <X size={12} />
                    </button>
                  </div>
                );
              })}
              <button
                className="watch-add"
                onClick={() =>
                  document.getElementById("ticker-search")?.focus()
                }
              >
                <Plus size={20} />
                <span>Find a stock</span>
              </button>
            </div>
          </section>

          <div className="research-grid">
            <section className="panel market-panel" aria-label="Stock research">
              <div className="market-toolbar">
                <div className="segmented">
                  <button
                    aria-pressed={view === "short"}
                    onClick={() => setView("short")}
                  >
                    Short-term view
                  </button>
                  <button
                    aria-pressed={view === "long"}
                    onClick={() => setView("long")}
                  >
                    Long-term view
                  </button>
                </div>
                <form
                  className="search-box"
                  onSubmit={(event) => {
                    event.preventDefault();
                    const value = query.trim().toUpperCase();
                    if (!/^[A-Z0-9.-]{1,10}$/.test(value)) {
                      setNotice(
                        "Enter a valid US ticker symbol, such as AAPL.",
                      );
                      return;
                    }
                    setTicker(value);
                    setQuery("");
                  }}
                >
                  <Search size={16} />
                  <input
                    id="ticker-search"
                    aria-label="Search ticker"
                    placeholder="Search ticker…"
                    maxLength={10}
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                  />
                  <button type="submit" aria-label="Search">
                    <ArrowRight size={15} />
                  </button>
                </form>
              </div>
              <div className="stock-heading">
                <div className={`company-icon large ${company.color}`}>
                  {ticker.slice(0, 1)}
                </div>
                <div>
                  <h2>{company.name}</h2>
                  <p>
                    {ticker} <span>·</span> {company.sector}
                  </p>
                </div>
                <button
                  className={`button small save-button ${saved ? "is-saved" : ""}`}
                  disabled={saving || !workspaceReady}
                  onClick={() => void toggleTicker(ticker)}
                >
                  {saved ? <Check size={15} /> : <Plus size={15} />}{" "}
                  {saved ? "Watching" : "Watch"}
                </button>
              </div>
              {loading ? (
                <div className="chart-loading" role="status">
                  <div className="skeleton price-skeleton" />
                  <div className="skeleton chart-skeleton" />
                  <span>Loading research for {ticker}…</span>
                </div>
              ) : error ? (
                <div className="empty-state">
                  <Compass size={32} />
                  <h3>Research is on its way.</h3>
                  <p role="alert">{error}</p>
                  <button className="button" onClick={() => setDemo(true)}>
                    Explore the design preview <ArrowRight size={15} />
                  </button>
                </div>
              ) : (
                research && (
                  <>
                    <div className="price-row">
                      <div>
                        <div className="price-value">
                          {money(research.market_data.latest_price)}
                          <span>USD</span>
                        </div>
                        <span
                          className={`price-change ${change < 0 ? "negative" : ""}`}
                        >
                          {change < 0 ? (
                            <ArrowDownRight size={15} />
                          ) : (
                            <ArrowUpRight size={15} />
                          )}{" "}
                          {change >= 0 ? "+" : ""}
                          {money(change)} ({changePercent.toFixed(2)}%){" "}
                          <span>last session</span>
                        </span>
                      </div>
                      <div className="quote-meta">
                        <span
                          className={research.stale ? "stale-tag" : "quiet-tag"}
                        >
                          {demo
                            ? "ILLUSTRATIVE"
                            : research.stale
                              ? "STALE SNAPSHOT"
                              : "DAILY SNAPSHOT"}
                        </span>
                        <small>
                          {research.market_data.price_type}
                          <br />
                          {research.market_data.as_of}
                        </small>
                      </div>
                    </div>
                    <div className="chart-toolbar">
                      <div
                        className="range-control"
                        aria-label="Chart time frame"
                      >
                        {["1W", "1M", "3M", "1Y", "ALL"].map((item) => (
                          <button
                            key={item}
                            aria-pressed={range === item}
                            onClick={() => setRange(item)}
                          >
                            {item}
                          </button>
                        ))}
                      </div>
                      {view === "short" && (
                        <button
                          className="forecast-toggle"
                          role="switch"
                          aria-checked={showForecast}
                          onClick={() => setShowForecast(!showForecast)}
                        >
                          <span
                            className={`switch ${showForecast ? "on" : ""}`}
                          />
                          Show forecast
                        </button>
                      )}
                    </div>
                    <PriceChart
                      research={research}
                      range={range}
                      horizon={horizon}
                      showForecast={showForecast && view === "short"}
                    />
                    {view === "short" ? (
                      <div className="forecast-bar">
                        <div className="forecast-label">
                          <Sparkles size={18} />
                          <span>
                            {demo
                              ? "Illustrative forecast"
                              : "Experimental forecast"}
                            <small>
                              {research.stale
                                ? "Snapshot is stale; awaiting refresh"
                                : "An estimate, with room for uncertainty"}
                            </small>
                          </span>
                        </div>
                        <label className="horizon-select">
                          <span className="sr-only">Forecast horizon</span>
                          <select
                            value={horizon}
                            onChange={(event) =>
                              setHorizon(Number(event.target.value))
                            }
                          >
                            {[1, 2, 3, 4, 5].map((day) => (
                              <option key={day} value={day}>
                                {day === 1
                                  ? "Next trading day"
                                  : `${day} trading days`}
                              </option>
                            ))}
                          </select>
                        </label>
                        <div className="forecast-number">
                          <strong>
                            {forecast
                              ? money(forecast.expected_price)
                              : "Not available"}
                          </strong>
                          <small>
                            {forecast
                              ? `${forecast.predicted_return >= 0 ? "+" : ""}${(forecast.predicted_return * 100).toFixed(2)}% estimated`
                              : "This horizon is not published"}
                          </small>
                        </div>
                      </div>
                    ) : (
                      <div className="long-term-note">
                        <ShieldCheck size={22} />
                        <div>
                          <strong>
                            A longer horizon needs a broader picture.
                          </strong>
                          <p>
                            Explore price history here. Fundamental analysis and
                            long-term rankings are still in development.
                          </p>
                        </div>
                      </div>
                    )}
                    <div className="chart-footnote">
                      <span>{research.market_data.source}</span>
                      <button onClick={() => setModal("method")}>
                        How to read this <ArrowUpRight size={12} />
                      </button>
                    </div>
                  </>
                )
              )}
            </section>

            <aside className="right-column">
              <section className="goal-card">
                <div className="goal-top">
                  <span className="eyebrow">YOUR INVESTMENT COMPASS</span>
                  <Compass size={21} />
                </div>
                <h2>
                  Big ambitions.
                  <br />
                  <em>Thoughtful steps.</em>
                </h2>
                <p>A starting point shaped around you.</p>
                <div className="budget-display">
                  <span>Available to invest</span>
                  <strong>
                    {workspace.profile.budget
                      ? money(workspace.profile.budget)
                      : "Set your budget"}
                  </strong>
                </div>
                <div className="goal-divider" />
                <div className="goal-details">
                  <div>
                    <span>Time horizon</span>
                    <strong>
                      {workspace.profile.horizon === "long"
                        ? "Long-term"
                        : "Short-term"}
                    </strong>
                  </div>
                  <div>
                    <span>Risk preference</span>
                    <strong className="capitalize">
                      {workspace.profile.risk}
                    </strong>
                  </div>
                </div>
                {workspace.profile.goal && (
                  <p className="goal-quote">“{workspace.profile.goal}”</p>
                )}
                <button
                  className="button peach full-width"
                  disabled={!workspaceReady}
                  onClick={() => setModal("profile")}
                >
                  {workspace.profile.budget
                    ? "Edit your preferences"
                    : "Personalize your workspace"}
                  <ArrowUpRight size={16} />
                </button>
              </section>
              <section className="panel signals-card">
                <div className="section-heading">
                  <h2>Behind the signal</h2>
                  <span className="tiny muted">RESEARCH</span>
                </div>
                <div className="signal-item">
                  <span className="signal-icon">
                    <TrendingUp size={17} />
                  </span>
                  <div>
                    <strong>Market history</strong>
                    <small>
                      {demo
                        ? "Illustrative price patterns"
                        : "Price & volume patterns"}
                    </small>
                  </div>
                  <span className="signal-status">
                    {demo ? "Preview" : research ? "Available" : "Pending"}
                  </span>
                </div>
                <div className="signal-item">
                  <span className="signal-icon">
                    <Globe2 size={17} />
                  </span>
                  <div>
                    <strong>News sentiment</strong>
                    <small>Company events & coverage</small>
                  </div>
                  <span className="signal-status neutral">
                    {research?.news.length ? "Archived" : "Pending"}
                  </span>
                </div>
                <div className="signal-item">
                  <span className="signal-icon">
                    <Compass size={17} />
                  </span>
                  <div>
                    <strong>Investor activity</strong>
                    <small>Disclosures & their timing</small>
                  </div>
                  <span className="signal-status neutral">Exploring</span>
                </div>
                <button
                  className="text-button"
                  onClick={() => setModal("method")}
                >
                  Explore our methodology <ArrowRight size={14} />
                </button>
              </section>
            </aside>
          </div>

          <section id="discover" className="discover-section">
            <div className="section-heading">
              <div>
                <div className="eyebrow">LOOK BEYOND THE OBVIOUS</div>
                <h2>Build your next perspective.</h2>
              </div>
              <span className="quiet-tag">RESEARCH STARTING POINTS</span>
            </div>
            <p className="section-copy">
              Explore companies for your watchlist. Personalized purchase
              recommendations are still being validated.
            </p>
            <div className="discovery-grid">
              {["MSFT", "NVDA", "AMZN"].map((symbol, index) => (
                <button
                  className="discovery-card"
                  key={symbol}
                  onClick={() => {
                    setTicker(symbol);
                    document
                      .getElementById("main")
                      ?.scrollIntoView({ behavior: "smooth" });
                  }}
                >
                  <div className="discovery-top">
                    <span className={`company-icon ${companies[symbol].color}`}>
                      {symbol[0]}
                    </span>
                    <span className="tiny muted">0{index + 1}</span>
                  </div>
                  <div className="discovery-title">
                    <div>
                      <h3>{companies[symbol].name}</h3>
                      <span>
                        {symbol} · {companies[symbol].sector}
                      </span>
                    </div>
                    <ArrowUpRight size={20} />
                  </div>
                  <div className="discovery-footer">
                    <span>
                      {
                        [
                          "Cloud & enterprise",
                          "Computing & infrastructure",
                          "Commerce & cloud",
                        ][index]
                      }
                    </span>
                    <span>Explore</span>
                  </div>
                </button>
              ))}
            </div>
          </section>

          {research && research.news.length > 0 && (
            <section className="news-section">
              <div className="section-heading">
                <h2>Company news</h2>
                <span className="muted tiny">
                  Archived coverage for {ticker}
                </span>
              </div>
              <div className="news-grid">
                {research.news.map((article, index) => (
                  <article
                    className="panel news-card"
                    key={`${article.url}-${index}`}
                  >
                    <span className="eyebrow">{article.source}</span>
                    <h3>{article.title}</h3>
                    <p>{article.published_at.slice(0, 10)}</p>
                    {/^https?:\/\//i.test(article.url) && (
                      <a
                        href={article.url}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        Read article <ArrowUpRight size={14} />
                      </a>
                    )}
                  </article>
                ))}
              </div>
            </section>
          )}
          <footer className="page-footer">
            <span className="footer-brand">hermes.</span>
            <p>
              Perspective, before possibility. Experimental research—not a
              guarantee of future performance.
            </p>
            <span>US equities · USD</span>
          </footer>
        </div>
      </main>

      {modal === "profile" && (
        <Modal title="Your investment preferences" close={() => setModal(null)}>
          <ProfileForm
            profile={workspace.profile}
            saving={saving}
            signedIn={Boolean(account)}
            save={async (profile) => {
              await update({ ...workspace, profile });
              setModal(null);
              setNotice(
                account
                  ? "Your preferences have been saved to your account."
                  : "Your preferences have been saved in this browser.",
              );
            }}
          />
        </Modal>
      )}
      {modal === "account" && (
        <Modal
          title={account ? "Your Hermes account" : "A workspace of your own."}
          close={() => setModal(null)}
        >
          <p className="muted">
            {account ??
              "Keep your watchlist and investment preferences with you, wherever you research."}
          </p>
          {account ? (
            <button
              className="button full-width"
              onClick={async () => {
                try {
                  await signOut();
                  setAccount(null);
                  setWorkspace(readGuest());
                  setModal(null);
                } catch {
                  setNotice("Could not sign out. Please try again.");
                }
              }}
            >
              <LogOut size={16} /> Sign out
            </button>
          ) : (
            <div className="auth-buttons">
              <button
                className="button primary full-width"
                disabled={!authConfigured || !accountReady}
                onClick={() => void beginLogin(true)}
              >
                Continue with Google <ArrowUpRight size={16} />
              </button>
              <button
                className="button full-width"
                disabled={!authConfigured || !accountReady}
                onClick={() => void beginLogin()}
              >
                Sign in or create an account
              </button>
              <p className="tiny muted">
                {authConfigured
                  ? "Email sign-in, account creation, and password recovery are handled securely by our account provider."
                  : "Accounts are not connected in this environment yet. You can explore and save guest preferences in this browser."}
              </p>
            </div>
          )}
        </Modal>
      )}
      {modal === "method" && (
        <Modal title="Evidence before confidence." close={() => setModal(null)}>
          <div className="method-content">
            <p>
              Hermes is building a research system around market history,
              company news, and investor disclosures. Each signal must
              demonstrate value on unseen data before it contributes to customer
              recommendations.
            </p>
            <h3>Where the research stands</h3>
            <p>
              The saved historical benchmark measured 50.32% direction accuracy
              for the price hybrid, compared with 52.10% for an always-up
              baseline. It does not establish a predictive advantage.
            </p>
            <h3>What the chart means</h3>
            <p>
              The solid line shows daily closing prices. The dashed line
              connects the last close to a forecast endpoint; it is not a
              prediction of the intervening path. Calibrated uncertainty
              intervals are not yet available.
            </p>
            <h3>Know the limits</h3>
            <p>
              Research snapshots currently publish one-day and five-day
              price-only estimates. News is separate context; investor activity
              and personalized purchase rankings are not validated. Design
              preview values are entirely synthetic.
            </p>
            {research && !demo && (
              <p className="tiny muted">
                Model version: {research.model_version}
                <br />
                Published: {research.published_at}
              </p>
            )}
          </div>
        </Modal>
      )}
    </div>
  );
}
