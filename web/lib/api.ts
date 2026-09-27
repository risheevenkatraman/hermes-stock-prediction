import { session } from "./auth";
import type { Research, Workspace } from "./types";

const base = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");
export async function request<T>(path: string, init: RequestInit = {}, authenticated = false): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body) headers.set("Content-Type", "application/json");
  if (authenticated) {
    const account = await session();
    if (!account) throw new Error("Please sign in to sync your workspace.");
    headers.set("Authorization", `Bearer ${account.token}`);
  }
  let response: Response;
  try { response = await fetch(`${base}${path}`, { ...init, headers, cache: "no-store" }); }
  catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new Error("Research is temporarily unavailable. Please try again shortly.");
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : "We could not complete that request.");
  }
  return response.json();
}
export const getResearch = (ticker: string, signal: AbortSignal) => request<Research>(`/v1/research/${encodeURIComponent(ticker)}`, { signal });
export const getWorkspace = () => request<Workspace>("/v1/workspace", {}, true);
export const saveWorkspace = (value: Workspace) => request<Workspace>("/v1/workspace", { method: "PUT", body: JSON.stringify(value) }, true);
