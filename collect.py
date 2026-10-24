"""Collect NBA coach's challenge outcomes from NBA.com play-by-play (via nba_api).

Usage:
    python collect.py                       # default seasons
    python collect.py --seasons 2024-25     # one season

Raw play-by-play is cached per game in data/raw/, so reruns resume where they
stopped. Output: data/challenges.csv, one row per coach's challenge.
"""
import argparse
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder, playbyplayv3

DATA = Path(__file__).parent / "data"
RAW = DATA / "raw"
DEFAULT_SEASONS = ["2023-24", "2024-25", "2025-26"]
TIMEOUTS_PER_GAME = 7
_cooldown_until = [0.0]  # shared across threads: back off together when NBA.com throttles
_lock = threading.Lock()

# Rows that are bookkeeping, not the call being challenged.
NON_CALL_ACTIONS = {"Substitution", "Timeout", "Instant Replay", "period", ""}


def fetch_games(season):
    df = leaguegamefinder.LeagueGameFinder(
        season_nullable=season, league_id_nullable="00",
        season_type_nullable="Regular Season", timeout=60,
    ).get_data_frames()[0]
    df["IS_HOME"] = df.MATCHUP.str.contains(" vs. ")
    df["SEASON"] = season
    return df[["SEASON", "GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "IS_HOME"]]


def fetch_pbp(game_id, retries=6):
    path = RAW / f"{game_id}.csv"
    if path.exists():
        return pd.read_csv(path, dtype={"gameId": str}, keep_default_na=False)
    for attempt in range(retries):
        time.sleep(max(0.0, _cooldown_until[0] - time.time()))
        try:
            df = playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=60).get_data_frames()[0]
            df.to_csv(path, index=False)
            time.sleep(0.4)
            return pd.read_csv(path, dtype={"gameId": str}, keep_default_na=False)
        except Exception as e:  # NBA.com throttles intermittently
            if attempt == retries - 1:
                break
            wait = 20 * 2 ** attempt
            with _lock:
                _cooldown_until[0] = max(_cooldown_until[0], time.time() + wait)
            print(f"  {game_id}: {type(e).__name__}, all workers pausing {wait}s", flush=True)
    print(f"  {game_id}: giving up")
    return None


def clock_seconds(clock):
    m = re.match(r"PT(\d+)M([\d.]+)S", clock)
    return int(m.group(1)) * 60 + float(m.group(2)) if m else None


def call_category(row):
    """Bucket the logged call immediately preceding the challenge."""
    a, s = row.actionType, str(row.subType).lower()
    if a == "Foul":
        if "shooting" in s:
            return "Shooting foul"
        if "offensive" in s or "charge" in s:
            return "Offensive foul / charge"
        if "loose ball" in s:
            return "Loose-ball foul"
        return "Personal / other foul"
    if a == "Turnover":
        if "goaltend" in s or "basket interference" in s:
            return "Goaltending / BI"
        if "out of bounds" in s:
            return "Out of bounds"
        if "offensive foul" in s:
            return "Offensive foul / charge"
        return "Other turnover"
    if a == "Violation":
        if "goaltend" in s or "basket" in s:
            return "Goaltending / BI"
        return "Other violation"
    if a == "Rebound" and not str(row.playerName).strip():
        return "Out of bounds"  # team rebound = ball ruled out of bounds
    if a in ("Rebound", "Missed Shot", "Made Shot", "Jump Ball"):
        return "Other possession call"
    if a == "Free Throw":
        return "Foul (call not logged)"
    return "Other"


def parse_challenges(pbp, game_rows):
    """Return one dict per coach's challenge in a single game's play-by-play."""
    out = []
    pbp = pbp.reset_index(drop=True)
    # Forward-fill score so every row knows the score at that moment.
    for col in ("scoreHome", "scoreAway"):
        pbp[col] = pd.to_numeric(pbp[col].replace("", None), errors="coerce").ffill().fillna(0)

    home = game_rows[game_rows.IS_HOME].iloc[0]
    away = game_rows[~game_rows.IS_HOME].iloc[0]
    # Timeout rows name the team by nickname ("76ERS", "Trail Blazers").
    nick = {}
    for t in (home, away):
        nick[t.TEAM_NAME.split()[-1].lower()] = t
        nick[" ".join(t.TEAM_NAME.split()[-2:]).lower()] = t

    def timeout_team(desc):
        name = desc.split(" Timeout")[0].strip().lower()
        return nick.get(name) if name in nick else nick.get(name.split()[-1]) if name else None

    def timeouts_used(desc):
        m = re.search(r"\((?:Full|Reg\.)\s*(\d+)", desc)
        return int(m.group(1)) if m else None

    replay_idx = pbp.index[(pbp.actionType == "Instant Replay") & pbp.subType.str.startswith("Coach Challenge")]
    for i in replay_idx:
        r = pbp.loc[i]
        # Challenging team: the timeout row just before the replay, same period and clock.
        window = pbp.loc[max(0, i - 6):i - 1]
        to = window[(window.actionType == "Timeout") & (window.period == r.period) & (window.clock == r.clock)]
        if to.empty:
            continue
        to_row = to.iloc[-1]
        team = timeout_team(to_row.description)
        if team is None:
            continue
        is_home = bool(team.IS_HOME)

        # Timeouts the team had already used, from its previous timeout row's official tally.
        prior = pbp.loc[:to_row.name - 1]
        prior = prior[prior.actionType == "Timeout"]
        prior = prior[prior.description.map(lambda d: timeout_team(d) is not None and timeout_team(d).TEAM_ID == team.TEAM_ID)]
        used_before = timeouts_used(prior.iloc[-1].description) if len(prior) else 0
        used_before = used_before or 0
        prior_challenges = len(_team_replays(pbp, i, team, timeout_team))

        # The call being challenged: last real event before the timeout row.
        before = pbp.loc[:to_row.name - 1]
        before = before[~before.actionType.isin(NON_CALL_ACTIONS)]
        call = before.iloc[-1] if len(before) else None
        same_clock = call is not None and call.period == r.period and call.clock == r.clock

        sec_left = clock_seconds(r.clock)
        period = int(r.period)
        margin = (r.scoreHome - r.scoreAway) * (1 if is_home else -1)
        out.append({
            "game_id": r.gameId,
            "season": home.SEASON,
            "game_date": home.GAME_DATE,
            "period": period,
            "clock": r.clock,
            "sec_left_period": sec_left,
            "sec_left_game": (max(0, 4 - period) * 720 + sec_left) if period <= 4 else sec_left,
            "team": team.TEAM_ABBREVIATION,
            "opponent": (away if is_home else home).TEAM_ABBREVIATION,
            "is_home": is_home,
            "score_margin": margin,  # challenging team's lead (+) or deficit (-)
            "timeouts_left": TIMEOUTS_PER_GAME - used_before,
            "prior_challenges_by_team": prior_challenges,
            "call_action": call.actionType if call is not None else "",
            "call_subtype": call.subType if call is not None else "",
            "call_description": call.description if call is not None else "",
            "call_logged_same_clock": same_clock,
            "call_type": call_category(call) if same_clock else "Call not in log",
            "ruling": r.subType.replace("Coach Challenge ", ""),
            "success": int("Overturn" in r.subType),
        })
    return out


def _team_replays(pbp, upto, team, timeout_team):
    """Indices of earlier coach's-challenge replays initiated by this team."""
    idx = []
    reps = pbp.index[(pbp.index < upto) & (pbp.actionType == "Instant Replay") & pbp.subType.str.startswith("Coach Challenge")]
    for j in reps:
        w = pbp.loc[max(0, j - 6):j - 1]
        w = w[(w.actionType == "Timeout") & (w.clock == pbp.loc[j, "clock"])]
        if len(w):
            t = timeout_team(w.iloc[-1].description)
            if t is not None and t.TEAM_ID == team.TEAM_ID:
                idx.append(j)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="max games per season (for testing)")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)

    games = pd.concat([fetch_games(s) for s in args.seasons])
    games = games.groupby("GAME_ID").filter(lambda g: len(g) == 2 and g.IS_HOME.sum() == 1)
    ids = sorted(games.GAME_ID.unique())
    if args.limit:
        ids = [g for s in args.seasons for g in sorted(games[games.SEASON == s].GAME_ID.unique())[:args.limit]]
    print(f"{len(ids)} games across {args.seasons}")

    rows, done = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_pbp, gid): gid for gid in ids}
        for f in as_completed(futs):
            gid, pbp = futs[f], f.result()
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(ids)} games, {len(rows)} challenges so far", flush=True)
            if pbp is None or pbp.empty:
                continue
            try:
                rows += parse_challenges(pbp, games[games.GAME_ID == gid])
            except Exception as e:
                print(f"  {gid}: parse error {e!r}")

    df = pd.DataFrame(rows).sort_values(["game_date", "game_id", "period", "sec_left_period"],
                                        ascending=[True, True, True, False])
    df.to_csv(DATA / "challenges.csv", index=False)
    print(f"Wrote {len(df)} challenges to {DATA / 'challenges.csv'}  (success rate {df.success.mean():.1%})")


if __name__ == "__main__":
    main()
