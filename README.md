# Coach's Challenge Success Model

Pulls every NBA coach's challenge from NBA.com play-by-play (through the free `nba_api` package)
and fits a logistic regression of whether the call gets overturned. The inputs are call type,
game phase, score margin, timeouts left, and whether it's the team's second challenge.

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python collect.py                 # ~35 min the first time; cached in data/raw/
.venv/bin/python model.py                   # writes report/summary.html
```

- `collect.py --seasons 2024-25 2025-26` changes the seasons. Reruns reuse the cached games.
- `data/challenges.csv`: one row per challenge, with the raw call text that preceded it.
- `report/summary.html`: the one-page summary. `effects.csv` and `scenarios.csv` hold the numbers behind it.
- `BREAK_EVEN` in `model.py` sets the success probability at which a challenge is "worth it."

**Known data quirk:** when a foul is overturned, NBA.com usually deletes it from the play-by-play.
Those challenges show up as `call_type = "Call not in log"` and are treated as fouls. The model
therefore works at the call-family level (foul / out of bounds / goaltending / other).
