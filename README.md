# Sleeper Fantasy Scoring and League Competitiveness Analysis

A Sleeper-only analysis of positional value, roster composition, weekly
matchups, season outcomes, and scoring volatility. It compares current league
scoring with QB/TE full PPR and RB/WR half-PPR while preserving the existing
six-position ranking outputs and charts.

The analysis is descriptive and does not generate conclusions or scoring
recommendations. The notebook leaves every personal-analysis section for the
league manager to complete.

## Sleeper-only data workflow

The pipeline uses these Sleeper API endpoints:

- League details and `scoring_settings`:
  `https://api.sleeper.app/v1/league/{league_id}`
- Current NFL season and week:
  `https://api.sleeper.app/v1/state/nfl`
- Full NFL player directory:
  `https://api.sleeper.app/v1/players/nfl`
- Season-to-date regular-season statistics:
  `https://api.sleeper.app/v1/stats/nfl/regular/{season}`
- Weekly raw regular-season statistics:
  `https://api.sleeper.app/v1/stats/nfl/regular/{season}/{week}`
- Current roster ownership and team labels:
  `/v1/league/{league_id}/rosters` and `/v1/league/{league_id}/users`
- Historical weekly matchups, actual starters, and roster snapshots:
  `/v1/league/{league_id}/matchups/{week}`

The statistics endpoint is undocumented. Its response is validated as a
non-empty object keyed by Sleeper player ID, containing stat-keyed objects
with finite numeric values. A changed or invalid response stops the run rather
than silently creating incomplete rankings.

Sleeper's player directory and season/weekly stats responses are cached under
`data/raw/`; non-empty matchup snapshots are cached there as well. Existing
caches are reused. `--refresh-players`, `--refresh-stats`, and
`--refresh-weekly` explicitly refresh those resources. Empty weekly stats
responses are not cached so an in-progress week can be checked again later.
The season payload includes player IDs with empty stat objects; those are
matched players with no production, not failed joins.

Player records are joined only by Sleeper player ID. Sleeper's player
directory uses team abbreviations such as `ARI` as both the stable ID and team
field for its 32 `DEF` entries. The season stats response has matching raw
defense records under those same IDs. The separate `TEAM_ARI` records are not
the DEF directory IDs and are not double-counted. The match report identifies
those additional stats IDs as unmatched. `DEF`, `DST`, and `D/ST` position
labels are normalized to `DEF/DST`; duplicate defense entries for a team are
rejected. The universe contains active eligible entries and any eligible
entry with current-season stats, preserving the existing eligibility rule.

## Scoring and rankings

- **Current:** the exact scoring settings returned for the configured Sleeper
  league.
- **Comparison:** 1.0 point per reception for QB and TE, and 0.5 points per
  reception for RB and WR. K and DEF/DST receive no reception points in the
  comparison format.
- All non-reception settings are inherited from the current league settings.

Both scores are calculated from Sleeper's raw stat categories. The scorer does
not use the endpoint's precomputed `pts_std`, `pts_half_ppr`, `pts_ppr`, or
rank fields. The scoring audit identifies league settings without an
applicable raw stat mapping for the eligible pool. Review
`outputs/scoring_audit.csv` before interpreting totals as exact for every
setting.

Every eligible player is ranked together once for current scoring and once
for comparison scoring. Rankings are deterministic: points descending,
receptions descending, case-insensitive player name, then Sleeper player ID.
Each tier is selected from the full player universe, independently by
scoring format. The tiers are Top 10, 20, 50, 75, and 100. When at least N
eligible entries exist, Top N contains exactly N unique entries. Ties are
broken by points descending, receptions descending, case-insensitive name,
then Sleeper ID.

For each tier and format, distribution tables count QB/RB/WR/TE/K/DEF/DST and
total entries. Percentages use the actual tier size as denominator. Count
change is `comparison_count - current_count`; percentage-point change is
`comparison_percentage - current_percentage` and is exported separately.
Tier membership uses Sleeper IDs and lists entering, exiting, and retained
entries with overall ranks in both formats.

## Outputs

### Overall positional distributions

- `outputs/positional_distribution_current.csv` — current format counts for
  Top 10/20/50/75/100, with all six positions and total.
- `outputs/positional_distribution_half_ppr_te.csv` — same counts for
  QB/TE PPR + RB/WR half-PPR.
- `outputs/positional_distribution_changes.csv` — current, comparison, and
  comparison-minus-current counts, percentages, and percentage-point
  differences for each position and tier.
- `outputs/positional_percentage_point_changes.csv` — comparison-minus-current
  positional percentage-point changes for every tier.
- `outputs/positional_distribution_percentages.csv` — count and percentage
  for every position, tier, and scoring format.
- `outputs/tier_membership_changes.csv` — overall-tier entrants, exits, and
  retained entries, with positions, both overall ranks, point totals, and rank
  movement.
- `outputs/eligible_position_counts.csv` and
  `outputs/player_directory_position_audit.csv` — included counts and raw
  Sleeper directory position normalization/exclusion audit.

### Charts

Charts are generated from the same distribution tables exported to CSV and
saved in `outputs/figures/`:

- `positional_composition_current.png` — current scoring stacked counts.
- `positional_composition_half_ppr_te.png` — comparison scoring stacked
  counts.
- `positional_composition_side_by_side.png` — both scoring formats side by
  side.
- `positional_count_changes.png` — positional count changes with a zero
  baseline; change means comparison minus current.
- `positional_percentage_distribution.png` — percentage composition for each
  tier and format.

Matplotlib defaults are used for chart colors.

## Competitiveness analyses

The pipeline uses the current `/rosters` response as the current team ownership
snapshot. `players`, taxi, and reserve IDs are combined and deduplicated.
Ownership conflicts and IDs absent from the full Sleeper player directory are
reported separately. Player rankings use the full active/stat-supported
universe; team Top 100 tables re-rank the same eligible players owned by the
current rosters. Both formats use identical rostered Sleeper IDs.

Additional CSV reports include:

- `roster_data_quality.csv` and `roster_ownership_audit.csv` — current roster
  snapshot coverage and one-team ownership validation.
- `rostered_player_rankings.csv` — player rankings within the rostered pool.
- `top100_team_distribution.csv`, `top100_team_changes.csv`,
  `top100_team_concentration.csv`, and `top100_player_membership_changes.csv`
  — team counts, shares, positional composition, HHI, and player entries,
  exits, and retained rank changes.
- `team_roster_strength.csv` and `team_positional_construction.csv` —
  season-to-date points represented on the current roster snapshot, top-player
  totals, position counts, production shares, and starter/replacement counts.
- `lineup_scenario_settings.json` and `lineup_slot_comparison.csv` — actual
  Sleeper starter slots and the proposed one-WR-to-one-standard-FLEX change.
- `lineup_player_scores.csv`, `team_roster_depth.csv`,
  `team_lineup_scenarios.csv`, `team_lineup_impact.csv`,
  `team_lineup_selections.csv`, `team_viable_options.csv`, and
  `team_flex_candidates.csv` — games-played coverage, position-specific depth,
  legal lineup selections under scenarios A–D, signed team effects, tolerance
  sensitivity, and RB/WR/TE FLEX candidate depth.
- `positional_scarcity.csv` — direct-slot starter and replacement thresholds,
  points-above-replacement gaps, and positional point gaps.
- `weekly_data_coverage.csv`, `weekly_matchup_rescoring.csv`,
  `weekly_team_scores.csv`, `weekly_team_season_outcomes.csv`,
  `weekly_competitiveness_summary.csv`, `weekly_by_week_summary.csv`, and
  `weekly_position_volatility.csv`, and `weekly_missing_starter_stats.csv` —
  fixed-lineup historical matchup
  rescoring, close-game rates, outcome changes, records, all-play comparisons,
  and variation across actual starter player-weeks.

Historical weekly scores are calculated from Sleeper raw stats using the
project scorer. The actual `starters` supplied for each week are held fixed
under both formats. Before a week is included, each team's recalculated
current-format starter points must reconcile with both Sleeper's
`starters_points` and matchup `points`; starter IDs must also belong to that
week's roster snapshot. All-play records compare each team's weekly score
against every other team in that week. A head-to-head minus all-play win-rate
difference is reported descriptively as a schedule comparison, not as proof
of schedule luck. HHI is the sum of squared team shares of the rostered-player
Top 100 and ranges from 0 to 1.

The replacement benchmark uses each position's direct starting slots times
the number of teams; one additional player per team defines the replacement
rank. Shared FLEX/SUPER_FLEX slots are not allocated to positions. Current
roster season-point totals are a snapshot of the players owned when the
pipeline runs; they are not historical manager totals or dynasty projections.

Figures saved under `outputs/figures/` include positional composition and
scarcity, top-two rostered points-per-game by position, rostered player counts,
best legal lineup output under all four scenarios, team-level effect
components, viable-option sensitivity, starter-to-bench gaps, FLEX candidates,
and fixed-starter weekly matchup margins. The notebook moves secondary team
distribution, rank-curve, close-game, schedule, and volatility charts to its
technical appendix.

The lineup optimizer uses Sleeper player IDs and current roster ownership. It
respects every configured starter slot, assigns each player at most once,
keeps SUPER_FLEX eligibility separate from standard FLEX, and returns empty
slots rather than inventing players. Scenario A is current scoring/current
slots; B is comparison scoring/current slots; C is current scoring/proposed
slots; D is comparison scoring/proposed slots. The only lineup change is one
fewer required WR and one additional standard FLEX. The four scenarios use
identical roster IDs and scored player records.

Lineup production is optimized on observed season-to-date points per game,
calculated using Sleeper `gms_active` and weekly active-game evidence only
when the season record lacks it. These are retrospective best-lineup
comparisons, not projections or actual submitted lineups. Viable options are
additional rostered players with a usable games-played sample whose points
per game are within a position-eligible starter's cutoff; the notebook shows
both 3- and 5-point tolerance sensitivity. The thresholds are descriptive,
not objective start recommendations.

### Player-level outputs and audits

- `outputs/overall_rankings_current.csv` and
  `outputs/overall_rankings_half_ppr_te.csv` — complete overall rankings for
  each scoring format.
- `outputs/top250_current_scoring.csv` and
  `outputs/top250_half_ppr_te.csv` — optional top-250 previews.
- `outputs/biggest_risers.csv` and `outputs/biggest_fallers.csv` — largest
  positional rank movements from the existing supplemental comparison.
- `outputs/positional_top100_comparison.csv` — supplemental positional-specific
  top-100 comparison; it is not the primary representation analysis.
- `outputs/player_match_report.csv` — eligible player stats status and
  unmatched stats IDs.
- `outputs/scoring_audit.csv` — league scoring settings, stat mappings, and
  applied, unsupported, or zero-value statuses.
- `outputs/current_sleeper_scoring_settings.csv` — current league setting
  values.
- `outputs/run_metadata.json` — saved league name (when supplied by Sleeper),
  season, NFL state week, player/stat record counts, and eligible counts by
  position for the notebook.
- `data/processed/all_ranked_players.csv` — complete eligible player pool,
  raw stats, both point totals, and ranks.

## Setup (Windows PowerShell and VS Code)

1. Open this folder in VS Code.
2. If `.env` does not exist, create it from `.env.example`:

   ```powershell
   Copy-Item .env.example .env
   ```

3. Set `SLEEPER_LEAGUE_ID` in `.env`. The league ID is the numeric segment
   after `/leagues/` in your Sleeper league URL. `SEASON` is optional; if
   omitted, Sleeper's NFL state supplies the season.
4. Create and activate a virtual environment and install dependencies:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

   If PowerShell blocks activation, use `.\.venv\Scripts\python.exe`.
5. Run:

   ```powershell
   python main.py
   ```

`.env` is ignored by Git and should not be committed.

## Options

- `py main.py --season 2026` — override the season for one run.
- `py main.py --refresh-players` — refresh the player-directory cache.
- `py main.py --refresh-stats` — refresh the selected season's stats cache.
- `py main.py --refresh-weekly` — refresh the weekly stats and matchup caches.
- `py main.py --no-charts` — skip generating PNG visualizations.
- Combine options as needed:

  ```powershell
  py main.py --refresh-stats --season 2026
  ```

## Jupyter analysis notebook

After running the pipeline, open
`notebooks/fantasy_scoring_analysis.ipynb` in VS Code, select the project's
Python environment, and choose **Run All**. The notebook reads the saved CSV
reports and `outputs/run_metadata.json` using paths relative to the project
root. It does not call Sleeper or duplicate scoring and lineup optimization;
those calculations live in the source pipeline. It checks the four scenario
coverage, starter-slot transformation, player uniqueness, and report schemas
before displaying the results.

The notebook tells a visual story from reception scoring and positional value
through team depth, best legal lineups under the four scenarios, viable
options, FLEX candidates, signed team-level changes, and scoring-only
historical matchup context. It distinguishes current roster ownership with
season-to-date results from actual historical starters. Secondary charts,
quality checks, and detailed CSV previews are in the technical appendix.
Figures are rendered inline and saved under `outputs/figures/`. The `My
Takeaway` prompts remain blank for the league manager; the notebook does not
write conclusions or recommendations.

The five tiers are Top 10, 20, 50, 75, and 100; eligible positions are QB, RB,
WR, TE, K, and DEF/DST. The two formats are the exact current league settings
and comparison reception settings described above.

The declared `ipykernel` and `nbformat` dependencies support selecting and
validating the notebook kernel. Open the notebook from either the project
folder or its `notebooks` folder; it searches parent folders for the project
root.

## Limitations

The Sleeper stats endpoint is undocumented and may change its shape or raw
stat coverage. Validation catches malformed responses, but the scoring audit
is still necessary: nonzero scoring categories without a usable raw
Sleeper-stat mapping cannot be included. Team-defense categories do not map
to player rankings unless the individual league settings and Sleeper raw
categories support them. The 2026 endpoint response has team defense raw
stats keyed by the directory's team-abbreviation IDs, as well as separate
`TEAM_*` records that are not those defense entries. The scoring audit
identifies any league settings absent from the raw response. No external NFL
stats source or Sleeper precomputed fantasy-point totals are used.

Historical matchup analysis includes only completed regular-season weeks with
valid weekly stats, historical starters/roster snapshots, and reconciled
points. The current NFL state week is excluded; see `weekly_data_coverage.csv`
for included and unavailable weeks. Weekly missing-stat starter IDs score
zero and are reported; validation requires the total to reconcile to Sleeper.
The reports reconstruct head-to-head and all-play records and points-for ranks;
they do not reproduce an official standings order or league-specific tie-breaks.
The current roster reports combine season-to-date production with today's
ownership, so players' earlier points can remain attributed to their current
team after a trade. The analysis does not infer injuries or manager intent and
does not optimize historical lineups because that would use realized scores
that were unknown at lineup lock. The current and proposed lineup comparison
is a retrospective optimization on observed season-to-date points per game,
not actual lineups or a projection; low game counts remain visible and no
minimum-games cutoff is imposed. Players with no usable games-played sample
remain in roster totals but are excluded from per-game lineup and viable-option
calculations. Historical matchup results hold actual historical starters
fixed and therefore measure scoring changes only, not the combined lineup
proposal. Flexible slots are not assigned to a specific position for the
league-wide starter/replacement thresholds. No multi-season or cross-league
comparison is currently included.

## Tests

Run the local unit and mocked end-to-end tests with:

```powershell
python -m pytest
```

Tests use fixtures and do not require network access or a Sleeper league.
