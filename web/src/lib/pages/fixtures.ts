// The Fixtures page's data (src/web_data/fixtures_page.py). Types only: safe to import anywhere.
export interface FixtureBit { opp: string; home: boolean; grade: number | null }
export interface FixtureCell { grade: number | null; fx: FixtureBit[] }
export interface FixtureTeam { team: string; avg: number; run: "Easy" | "Mixed" | "Brutal"; cells: (FixtureCell | null)[] }
export interface FixturesPage { start_gw: number; horizon: number; gws: number[]; teams: FixtureTeam[] }

/** The difficulty scale, easiest to hardest: the Streamlit tab's FIXTURE_SCALE. */
export const GRADE_NAMES = ["easiest", "easy", "neutral", "hard", "hardest"];
