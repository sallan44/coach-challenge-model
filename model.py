"""Fit a logistic regression of coach's-challenge success and write a one-page summary.

Usage:
    python model.py            # reads data/challenges.csv, writes report/summary.html

Data caveat handled here: when a challenge overturns a foul, NBA.com usually deletes
the original foul from the play-by-play. Those challenges show up as "Call not in log"
and are almost always successes, so treating them as their own call type would leak the
outcome. We assume they were fouls and model call type at the family level
(foul / out of bounds & possession / goaltending / other), where that assumption puts every
challenge in the right bucket.
"""
import html
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "challenges.csv"
OUT = ROOT / "report" / "summary.html"

# Success probability above which a challenge is recommended. A failed challenge
# costs a timeout; a successful one flips a call and refunds the timeout. 0.5 treats
# a timeout and an overturned call as roughly equal value; raise it late in close games
# if you value timeouts more.
BREAK_EVEN = 0.50

FAMILY = {
    "Shooting foul": "Foul", "Personal / other foul": "Foul", "Loose-ball foul": "Foul",
    "Offensive foul / charge": "Foul", "Foul (call not logged)": "Foul", "Call not in log": "Foul",
    # An overturned out-of-bounds call is logged as the corrected possession (a turnover,
    # rebound or made shot), which only appears when the challenge wins, so pool them.
    "Out of bounds": "Out of bounds / possession", "Other possession call": "Out of bounds / possession",
    "Other turnover": "Out of bounds / possession", "Goaltending / BI": "Goaltending / BI",
}
FAMILY_ORDER = ["Foul", "Out of bounds / possession", "Goaltending / BI", "Other"]
PHASE_ORDER = ["Q1–Q3", "Q4, before last 2:00", "Q4, last 2:00"]
MARGIN_ORDER = ["0–5", "6–12", "13+"]


def prepare(df):
    df = df[df.period <= 4].copy()  # OT has its own timeout rules; too few to model
    df["call_family"] = df.call_type.map(FAMILY).fillna("Other")
    df["phase"] = np.select(
        [df.period < 4, df.sec_left_period > 120], PHASE_ORDER[:2], PHASE_ORDER[2])
    m = df.score_margin.abs()
    df["margin"] = np.select([m <= 5, m <= 12], MARGIN_ORDER[:2], MARGIN_ORDER[2])
    df["trailing"] = (df.score_margin < 0).astype(int)
    df["timeouts_left"] = df.timeouts_left.clip(0, 7)
    df["second_challenge"] = (df.prior_challenges_by_team > 0).astype(int)
    for col, order in [("call_family", FAMILY_ORDER), ("phase", PHASE_ORDER), ("margin", MARGIN_ORDER)]:
        df[col] = pd.Categorical(df[col], categories=[c for c in order if c in set(df[col])])
    return df


FORMULA = ("success ~ C(call_family) + C(phase) + C(margin) + trailing"
           " + timeouts_left + second_challenge + C(season)")


def fit(df):
    model = smf.logit(FORMULA, data=df).fit(disp=False)
    # Out-of-sample check with the same design matrix.
    X = model.model.exog
    y = df.success.values
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    p = cross_val_predict(LogisticRegression(C=1e6, max_iter=2000, fit_intercept=False), X, y,
                          cv=cv, method="predict_proba")[:, 1]
    metrics = {"auc": roc_auc_score(y, p), "brier": brier_score_loss(y, p),
               "brier_base": brier_score_loss(y, np.full_like(p, y.mean()))}
    return model, metrics


LABELS = {
    "C(call_family)[T.Out of bounds / possession]": "Out-of-bounds / possession (vs. foul)",
    "C(call_family)[T.Goaltending / BI]": "Goaltending / basket int. (vs. foul)",
    "C(call_family)[T.Other]": "Other call (vs. foul)",
    "C(phase)[T.Q4, before last 2:00]": "4th qtr before last 2:00 (vs. Q1–Q3)",
    "C(phase)[T.Q4, last 2:00]": "Last 2:00 of 4th (vs. Q1–Q3)",
    "C(margin)[T.6–12]": "Margin 6–12 (vs. 0–5)",
    "C(margin)[T.13+]": "Margin 13+ (vs. 0–5)",
    "trailing": "Challenging team is trailing",
    "timeouts_left": "Each extra timeout left",
    "second_challenge": "Second challenge of the game",
}


def effects_table(model):
    ci = model.conf_int()
    rows = []
    for k, label in LABELS.items():
        if k in model.params:
            rows.append({"term": label, "or_": np.exp(model.params[k]),
                         "lo": np.exp(ci.loc[k, 0]), "hi": np.exp(ci.loc[k, 1]),
                         "p": model.pvalues[k]})
    return pd.DataFrame(rows)


def scenario_grid(model, df):
    season = df.season.max()
    grid = []
    for fam in df.call_family.cat.categories:
        for phase in PHASE_ORDER:
            for margin in MARGIN_ORDER:
                grid.append({"call_family": fam, "phase": phase, "margin": margin, "trailing": 1,
                             "timeouts_left": 3 if phase != "Q1–Q3" else 5,
                             "second_challenge": 0, "season": season})
    g = pd.DataFrame(grid)
    g["p"] = model.predict(g)
    return g


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# ---------- SVG charts (single series, so one hue; native <title> tooltips) ----------

def bar_chart(rows, xmax=1.0, ref=None, fmt=lambda v: f"{v:.0%}", width=560):
    """Horizontal bars with CI whiskers. rows: list of (label, value, lo, hi, note)."""
    lab_w, row_h, pad = 170, 30, 110
    plot_w = width - lab_w - pad
    h = row_h * len(rows) + 28
    x = lambda v: lab_w + plot_w * min(max(v, 0), xmax) / xmax
    out = [f'<svg viewBox="0 0 {width} {h}" role="img" class="chart">']
    for t in np.linspace(0, xmax, 5):
        out.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="0" y2="{h-22}" class="grid"/>'
                   f'<text x="{x(t):.1f}" y="{h-6}" class="tick" text-anchor="middle">{fmt(t)}</text>')
    if ref is not None:
        out.append(f'<line x1="{x(ref):.1f}" x2="{x(ref):.1f}" y1="0" y2="{h-22}" class="ref"/>')
    for i, (label, v, lo, hi, note) in enumerate(rows):
        y = i * row_h + 6
        tip = html.escape(f"{label}: {fmt(v)} (95% CI {fmt(lo)}–{fmt(hi)}) · {note}")
        out.append(f'<g><title>{tip}</title>'
                   f'<rect x="{lab_w}" y="{y-3}" width="{plot_w}" height="{row_h-4}" class="hit"/>'
                   f'<text x="{lab_w-8}" y="{y+13}" class="lab" text-anchor="end">{html.escape(label)}</text>'
                   f'<path d="M{lab_w},{y+4} H{x(v)-4:.1f} a4,4 0 0 1 4,4 v6 a4,4 0 0 1 -4,4 H{lab_w} Z" class="bar"/>'
                   f'<line x1="{x(lo):.1f}" x2="{x(hi):.1f}" y1="{y+11}" y2="{y+11}" class="whisk"/>'
                   f'<text x="{x(max(v, hi))+6:.1f}" y="{y+15}" class="val">{fmt(v)}'
                   f'<tspan class="muted"> · {html.escape(note)}</tspan></text></g>')
    out.append("</svg>")
    return "".join(out)


def forest_chart(eff, width=560):
    """Odds ratios on a log axis, 1.0 marked."""
    lab_w, row_h, pad = 250, 28, 20
    plot_w = width - lab_w - pad
    lo_lim, hi_lim = 0.04, 16.0
    h = row_h * len(eff) + 28
    x = lambda v: lab_w + plot_w * (np.log(np.clip(v, lo_lim, hi_lim)) - np.log(lo_lim)) / (np.log(hi_lim) - np.log(lo_lim))
    out = [f'<svg viewBox="0 0 {width} {h}" role="img" class="chart">']
    for t in [0.05, 0.25, 1, 4, 16]:
        out.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="0" y2="{h-22}" class="{"ref" if t == 1 else "grid"}"/>'
                   f'<text x="{x(t):.1f}" y="{h-6}" class="tick" text-anchor="middle">{t:g}×</text>')
    for i, r in enumerate(eff.itertuples()):
        y = i * row_h + 14
        tip = html.escape(f"{r.term}: odds ×{r.or_:.2f} (95% CI {r.lo:.2f}–{r.hi:.2f}), p={r.p:.3f}")
        cls = "dot" if r.p < 0.05 else "dot faint"
        out.append(f'<g><title>{tip}</title>'
                   f'<rect x="0" y="{y-12}" width="{width}" height="{row_h-2}" class="hit"/>'
                   f'<text x="{lab_w-10}" y="{y+4}" class="lab" text-anchor="end">{html.escape(r.term)}</text>'
                   f'<line x1="{x(r.lo):.1f}" x2="{x(r.hi):.1f}" y1="{y}" y2="{y}" class="whisk"/>'
                   f'<circle cx="{x(r.or_):.1f}" cy="{y}" r="5" class="{cls}"/></g>')
    out.append("</svg>")
    return "".join(out)


def verdict(p):
    if p >= BREAK_EVEN + 0.10:
        return "go", "Challenge"
    if p >= BREAK_EVEN:
        return "maybe", "Borderline"
    return "no", "Hold"


def render(df, model, metrics, eff, grid):
    n, rate = len(df), df.success.mean()
    seasons = sorted(df.season.unique())
    unlogged = (df.call_type == "Call not in log")
    fam_rate = df.groupby("call_family", observed=True).success.mean().to_dict()
    phase_drop = 1 - np.exp(model.params["C(phase)[T.Q4, last 2:00]"])

    # Chart 1: observed success by call family.
    fam_rows = []
    for fam in df.call_family.cat.categories:
        s = df[df.call_family == fam]
        lo, hi = wilson(s.success.sum(), len(s))
        fam_rows.append((fam, s.success.mean(), lo, hi, f"n={len(s):,}"))
    # Chart 2: by game phase.
    phase_rows = []
    for ph in PHASE_ORDER:
        s = df[df.phase == ph]
        lo, hi = wilson(s.success.sum(), len(s))
        phase_rows.append((ph, s.success.mean(), lo, hi, f"n={len(s):,}"))

    # Scenario table: call family × phase, close game, trailing.
    tbl = ['<table class="grid-t"><thead><tr><th>Call challenged</th>']
    tbl += [f"<th>{p}</th>" for p in PHASE_ORDER] + ["</tr></thead><tbody>"]
    for fam in df.call_family.cat.categories:
        tbl.append(f"<tr><th scope='row'>{fam}</th>")
        for ph in PHASE_ORDER:
            p = grid[(grid.call_family == fam) & (grid.phase == ph) & (grid.margin == "0–5")].p.iloc[0]
            cls, word = verdict(p)
            tbl.append(f'<td class="{cls}"><b>{p:.0%}</b> <span>{word}</span></td>')
        tbl.append("</tr>")
    tbl.append("</tbody></table>")

    # Margin row for context (fouls, Q4 before last 2:00).
    marg = grid[(grid.call_family == "Foul") & (grid.phase == "Q4, before last 2:00")]
    marg_txt = ", ".join(f"{r.margin} pts: {r.p:.0%}" for r in marg.itertuples())

    sig = eff[eff.p < 0.05].sort_values("or_", key=lambda s: -abs(np.log(s)))
    findings = "".join(
        f"<li><b>{html.escape(r.term)}</b>: odds of success ×{r.or_:.2f} "
        f"(95% CI {r.lo:.2f}–{r.hi:.2f}).</li>" for r in sig.itertuples()) or \
        "<li>No situational factor reached p &lt; 0.05.</li>"

    css = (ROOT / "report_style.css").read_text()
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Coach's Challenge Model</title><style>{css}</style></head>
<body><main>
<header>
  <p class="eyebrow">NBA regular season · {seasons[0]} to {seasons[-1]} · NBA.com play-by-play via nba_api</p>
  <h1>When is a coach's challenge worth it?</h1>
  <p class="lede">Teams won <b>{rate:.0%}</b> of {n:,} regulation challenges. The biggest factor is what's being challenged:
  out-of-bounds and possession calls were overturned <b>{fam_rate.get("Out of bounds / possession", 0):.0%}</b> of the time,
  fouls <b>{fam_rate.get("Foul", 0):.0%}</b>, and goaltending only <b>{fam_rate.get("Goaltending / BI", 0):.0%}</b>.
  The odds of success fall {phase_drop:.0%} in the last two minutes of the 4th. Score margin made no measurable difference.</p>
</header>

<section class="tiles">
  <div><span class="big">{n:,}</span><span>challenges</span></div>
  <div><span class="big">{rate:.0%}</span><span>overturned</span></div>
  <div><span class="big">{metrics['auc']:.2f}</span><span>cross-validated AUC</span></div>
  <div><span class="big">{BREAK_EVEN:.0%}</span><span>break-even assumed</span></div>
</section>

<section class="two">
  <div><h2>Observed success by call</h2>{bar_chart(fam_rows, ref=rate)}
  <p class="note">The dashed line is the overall rate. Whiskers show 95% Wilson intervals.</p></div>
  <div><h2>Observed success by game phase</h2>{bar_chart(phase_rows, ref=rate)}</div>
</section>

<section>
  <h2>Predicted success: close game (0–5 pts), challenging team trailing</h2>
  {''.join(tbl)}
  <p class="note">Assumes 5 timeouts left in Q1–Q3 and 3 in the 4th, and that this is the team's first challenge.
  “Challenge” means ≥{BREAK_EVEN+0.10:.0%} predicted, “Borderline” {BREAK_EVEN:.0%}–{BREAK_EVEN+0.10:.0%},
  and “Hold” below {BREAK_EVEN:.0%}. Margin effect for a foul in the 4th, before the last 2:00: {marg_txt}.</p>
</section>

<section class="two">
  <div><h2>Logistic regression effects</h2>{forest_chart(eff)}
  <p class="note">Odds ratios on a log scale. Solid dots: p &lt; 0.05. Hollow dots: not distinguishable from no effect.
  Season fixed effects are included but not shown.</p></div>
  <div><h2>What the model says</h2><ul>{findings}</ul>
  <h2>Method &amp; caveats</h2><ul class="small">
  <li>Each challenge is the “Instant Replay – Coach Challenge” event in NBA.com play-by-play. The outcome is
  Overturn (success) vs. Support / Stands. The challenging team comes from the timeout called at that moment.</li>
  <li>Timeouts left come from the league's running timeout tally before the challenge. Score margin is from the challenging team's side.</li>
  <li><b>Call-type leak:</b> when a foul is overturned, NBA.com usually deletes it from the log.
  {unlogged.sum():,} challenges ({unlogged.mean():.0%}) had no call logged at that moment, and {df[unlogged].success.mean():.0%} of them succeeded.
  These are treated as fouls. That keeps the foul family unbiased, but it also means foul subtypes
  (shooting vs. loose-ball, etc.) can't be separated reliably. Overturned out-of-bounds calls show up as the corrected
  possession, so they are pooled into one “out of bounds / possession” family.</li>
  <li>This is selection-biased: it only covers calls coaches chose to challenge. It doesn't tell you how often
  an unchallenged call would have been overturned.</li>
  <li>Fit: AUC {metrics['auc']:.2f}, Brier {metrics['brier']:.3f} vs. {metrics['brier_base']:.3f} for always guessing the base rate (5-fold CV).
  The situation explains only part of the outcome. Most of it is what the coach saw on the replay.</li>
  </ul></div>
</section>
<footer>Generated by model.py · overtime excluded · {pd.Timestamp.today():%Y-%m-%d}</footer>
</main></body></html>"""


def main():
    df = prepare(pd.read_csv(DATA, dtype={"game_id": str}))
    model, metrics = fit(df)
    eff = effects_table(model)
    grid = scenario_grid(model, df)
    print(model.summary())
    print(metrics)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(render(df, model, metrics, eff, grid))
    eff.to_csv(OUT.parent / "effects.csv", index=False)
    grid.to_csv(OUT.parent / "scenarios.csv", index=False)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
