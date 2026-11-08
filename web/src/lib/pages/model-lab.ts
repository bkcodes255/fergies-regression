// Types and wording for the Model Lab page. The data comes from scripts/build_model_lab.py, run
// offline; src/data/model-lab.json is committed, not exported in CI.

export type ModelType = "baseline" | "linear_regression" | "random_forest" | "xgboost";
export type FittedModel = Exclude<ModelType, "baseline">;

export interface ModelRow {
  r2: number | null; mae: number | null; rmse: number | null;
  train_r2: number | null; overfit_gap: number | null;
  r2_ci: [number | null, number | null];
  fit: "ok" | "possible_overfit" | "possible_underfit" | null;
  delta_r2?: number | null; p_value?: number | null;
}
export interface Experiment {
  group: string; removed: string[]; n_kept: number;
  models: Record<ModelType, ModelRow>;
}
export interface ModelLab {
  generated_at: string; test_season: string; train_seasons: string[];
  n_train: number; n_test: number; sample_frac: number | null; n_features: number;
  bootstrap_draws: number; overfit_gap_threshold: number;
  groups: { name: string; features: string[] }[];
  reference: Record<ModelType, ModelRow>;
  importances: Record<"random_forest" | "xgboost", { feature: string; importance: number }[]>;
  experiments: Experiment[];
}

export const FITTED: FittedModel[] = ["linear_regression", "random_forest", "xgboost"];
export const MODEL_LABEL: Record<ModelType, string> = {
  baseline: "Naive baseline", linear_regression: "Linear regression", random_forest: "Random forest", xgboost: "XGBoost",
};
export const ALPHA = 0.05;

export type Verdict = "worse" | "better" | "none";

/** Only a p-value under 0.05 counts as a difference. Anything else is "no clear difference". */
export function verdict(row: ModelRow): Verdict {
  if (row.p_value == null || row.delta_r2 == null || row.p_value >= ALPHA) return "none";
  return row.delta_r2 < 0 ? "worse" : "better";
}

export const VERDICT_TEXT: Record<Verdict, string> = {
  worse: "Clearly worse without it",
  better: "Clearly better without it",
  none: "No clear difference",
};

export const FIT_TEXT: Record<string, string> = {
  ok: "OK", possible_overfit: "Possible overfit", possible_underfit: "Possible underfit",
};

export function num(x: number | null | undefined, d = 3): string {
  return x == null ? "n/a" : x.toFixed(d);
}
export function signed(x: number | null | undefined, d = 3): string {
  if (x == null) return "n/a";
  const s = x.toFixed(d);
  return x > 0 ? `+${s}` : s.replace("-", "−");
}
export function pText(p: number | null | undefined): string {
  if (p == null) return "n/a";
  return p < 0.001 ? "<0.001" : p.toFixed(3);
}
