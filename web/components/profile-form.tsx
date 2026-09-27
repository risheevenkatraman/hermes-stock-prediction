"use client";
import { useState } from "react";
import type { Profile } from "@/lib/types";

export default function ProfileForm({ profile, save, saving, signedIn }: { profile: Profile; save: (value: Profile) => Promise<void>; saving: boolean; signedIn: boolean }) {
  const [draft, setDraft] = useState(profile);
  const [error, setError] = useState("");
  return <form className="profile-form" onSubmit={async event => { event.preventDefault(); setError(""); try { await save(draft); } catch (error) { setError(error instanceof Error ? error.message : "Could not save your profile."); } }}>
    <p className="muted">Give your research a little direction. {signedIn ? "Your preferences are saved to your account." : "Guest preferences stay in this browser. Sign in to save across devices."}</p>
    <label>Available investment budget <span>USD</span><input name="budget" type="number" min="0" max="1000000000" step="0.01" value={draft.budget || ""} placeholder="e.g. 2,500" onChange={event => setDraft({ ...draft, budget: Number(event.target.value) })}/></label>
    <label>Your financial goal<textarea name="goal" maxLength={200} rows={3} placeholder="What are you investing toward?" value={draft.goal} onChange={event => setDraft({ ...draft, goal: event.target.value })}/><small>{draft.goal.length}/200 characters</small></label>
    <label>Investment horizon<select value={draft.horizon} onChange={event => setDraft({ ...draft, horizon: event.target.value as Profile["horizon"] })}><option value="short">Short-term · days to months</option><option value="long">Long-term · years</option></select></label>
    <label>Risk preference<select value={draft.risk} onChange={event => setDraft({ ...draft, risk: event.target.value as Profile["risk"] })}><option value="conservative">Conservative · prioritize stability</option><option value="balanced">Balanced · growth with measured risk</option><option value="growth">Growth · comfortable with larger swings</option></select></label>
    {error && <p role="alert" className="error-text">{error}</p>}
    <button className="button primary full-width" disabled={saving}>{saving ? "Saving…" : "Save preferences"}</button>
  </form>;
}
