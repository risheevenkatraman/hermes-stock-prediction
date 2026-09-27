import type { Research } from "./types";

// Deliberately synthetic design fixtures. Never used as live prices or model evidence.
export function demoResearch(ticker: string): Research {
  const seed = [...ticker].reduce(
    (value, char) => value + char.charCodeAt(0),
    0,
  );
  const base = 100 + (seed % 160);
  const prices: Research["prices"] = [];
  for (let day = 0; day < 180; day++) {
    const date = new Date(Date.UTC(2026, 0, 1 + day));
    if (date.getUTCDay() === 0 || date.getUTCDay() === 6) continue;
    prices.push({
      date: date.toISOString().slice(0, 10),
      close:
        Math.round(
          (base +
            day * 0.16 +
            Math.sin(day * 0.16) * 5 +
            Math.sin(day * 0.71 + seed) * 2) *
            100,
        ) / 100,
    });
  }
  const latest = prices.at(-1)!;
  return {
    ticker,
    published_at: "2026-06-29T20:30:00Z",
    expires_at: "2026-06-30T20:30:00Z",
    stale: false,
    model_version: "illustration",
    market_data: {
      as_of: latest.date,
      latest_price: latest.close,
      source: "Synthetic preview",
      price_type: "Illustrative daily close",
    },
    prices,
    forecasts: [1, 2, 3, 4, 5].map((horizon) => ({
      horizon,
      expected_price: latest.close * (1 + horizon * 0.003),
      predicted_return: horizon * 0.003,
    })),
    news: [],
    limitations:
      "Illustrative data only. These are not market prices or actual forecasts.",
  };
}
