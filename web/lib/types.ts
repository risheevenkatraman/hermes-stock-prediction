export type Profile = { budget: number; currency: "USD"; goal: string; horizon: "short" | "long"; risk: "conservative" | "balanced" | "growth" };
export type Workspace = { profile: Profile; watchlist: string[] };
export type Research = {
  ticker: string;
  published_at: string;
  expires_at: string;
  stale: boolean;
  model_version: string;
  market_data: { as_of: string; latest_price: number; source: string; price_type: string };
  prices: { date: string; close: number }[];
  forecasts: { horizon: number; expected_price: number; predicted_return: number }[];
  news: { title: string; source: string; url: string; published_at: string; sentiment: number }[];
  limitations: string;
};
export const defaultWorkspace: Workspace = {
  profile: { budget: 0, currency: "USD", goal: "", horizon: "long", risk: "balanced" },
  watchlist: ["AAPL", "MSFT", "NVDA", "AMZN"],
};
export const companies: Record<string, { name: string; sector: string; color: string }> = {
  AAPL: { name: "Apple Inc.", sector: "Technology", color: "stone" },
  MSFT: { name: "Microsoft", sector: "Technology", color: "blue" },
  NVDA: { name: "NVIDIA", sector: "Semiconductors", color: "green" },
  AMZN: { name: "Amazon", sector: "Consumer discretionary", color: "orange" },
  GOOGL: { name: "Alphabet", sector: "Communication services", color: "blue" },
  TSLA: { name: "Tesla", sector: "Consumer discretionary", color: "orange" },
};
