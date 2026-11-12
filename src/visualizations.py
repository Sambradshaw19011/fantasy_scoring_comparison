from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.representation import POSITIONS

POSITION_LABELS = {"DEF/DST": "DEF/DST"}


def create_positional_charts(
    current: pd.DataFrame,
    comparison: pd.DataFrame,
    changes: pd.DataFrame,
    percentages: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """Create five positional charts directly from the exported summary tables."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tiers = current["Tier"].tolist()
    if comparison["Tier"].tolist() != tiers:
        raise ValueError("Current and comparison distribution tiers do not match.")
    if len(tiers) != 5:
        raise ValueError("Positional charts require all five overall ranking tiers.")

    required = {"Tier", *POSITIONS}
    for label, frame in (("current", current), ("comparison", comparison)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"The {label} positional distribution is missing: {sorted(missing)}"
            )
    x = np.arange(len(tiers), dtype=float)
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {
        position: default_colors[index % len(default_colors)]
        for index, position in enumerate(POSITIONS)
    }

    def stacked_distribution(
        axis: plt.Axes, frame: pd.DataFrame, title: str
    ) -> None:
        bottom = np.zeros(len(tiers))
        for position in POSITIONS:
            values = frame[position].to_numpy(dtype=float)
            axis.bar(
                x,
                values,
                bottom=bottom,
                color=colors[position],
                label=POSITION_LABELS.get(position, position),
            )
            bottom += values
        axis.set_xticks(x, tiers)
        axis.set_xlabel("Overall ranking tier")
        axis.set_ylabel("Entries")
        axis.set_title(title)
        axis.legend(title="Position")

    figures: list[Path] = []
    for frame, title, filename in (
        (
            current,
            "Current league scoring: positional composition",
            "positional_composition_current.png",
        ),
        (
            comparison,
            "Proposed scoring: positional composition",
            "positional_composition_half_ppr_te.png",
        ),
    ):
        fig, axis = plt.subplots(figsize=(11, 6))
        stacked_distribution(axis, frame, title)
        fig.tight_layout()
        path = out / filename
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        figures.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    stacked_distribution(axes[0], current, "Current league scoring")
    stacked_distribution(axes[1], comparison, "Proposed scoring")
    fig.suptitle("Side-by-side positional composition")
    fig.tight_layout()
    path = out / "positional_composition_side_by_side.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    figures.append(path)

    fig, axis = plt.subplots(figsize=(12, 6))
    width = 0.8 / len(POSITIONS)
    for index, position in enumerate(POSITIONS):
        column = f"{position} count change"
        if column not in changes:
            raise ValueError(f"The positional changes table is missing {column!r}.")
        values = changes[column].to_numpy(dtype=float)
        positions_x = x - 0.4 + width * (index + 0.5)
        bars = axis.bar(
            positions_x,
            values,
            width,
            color=colors[position],
            label=POSITION_LABELS.get(position, position),
        )
        for bar, value in zip(bars, values):
            if value:
                axis.annotate(
                    f"{value:+.0f}",
                    (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 3 if value > 0 else -12),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if value > 0 else "top",
                )
    axis.axhline(0, linewidth=0.8)
    axis.set_xticks(x, tiers)
    axis.set_xlabel("Overall ranking tier")
    axis.set_ylabel("Entry count change (comparison - current)")
    axis.set_title("Positional count changes")
    axis.legend(title="Position")
    fig.tight_layout()
    path = out / "positional_count_changes.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    figures.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for axis, scoring_format in zip(
        axes, ("Current scoring", "Proposed scoring")
    ):
        selected = percentages.loc[
            percentages["Scoring format"].eq(scoring_format)
        ]
        bottom = np.zeros(len(tiers))
        for position in POSITIONS:
            values = (
                selected.loc[selected["Position"].eq(position)]
                .set_index("Tier")
                .reindex(tiers)["Percentage"]
                .to_numpy(dtype=float)
            )
            axis.bar(
                x,
                values,
                bottom=bottom,
                color=colors[position],
                label=POSITION_LABELS.get(position, position),
            )
            bottom += values
        axis.set_xticks(x, tiers)
        axis.set_xlabel("Overall ranking tier")
        axis.set_ylabel("Share of tier (%)")
        axis.set_ylim(0, 100)
        axis.set_title(
            "Current scoring"
            if scoring_format == "Current scoring"
            else "Proposed scoring"
        )
        axis.legend(title="Position")
    fig.suptitle("Positional percentage distribution")
    fig.tight_layout()
    path = out / "positional_percentage_distribution.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    figures.append(path)
    return figures


def create_competitiveness_charts(
    roster_tables: dict[str, pd.DataFrame],
    weekly_tables: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> list[Path]:
    """Create team balance and fixed-lineup matchup figures from report tables."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    distribution = roster_tables["top100_team_distribution"]
    changes = roster_tables["top100_team_changes"]
    construction = roster_tables["team_positional_construction"]
    distribution = distribution.copy()
    changes = changes.copy()
    construction = construction.copy()
    for frame in (distribution, changes, construction):
        frame["Roster ID"] = frame["Roster ID"].astype(str)
    roster_order = distribution.loc[
        distribution["Scoring format"].eq("Current scoring")
    ]
    expected_top100_size = min(100, len(roster_tables["rostered_player_rankings"]))
    for scoring_format in ("Current scoring", "QB/TE PPR + RB/WR half-PPR"):
        actual_size = distribution.loc[
            distribution["Scoring format"].eq(scoring_format),
            "Top 100 player count",
        ].sum()
        if actual_size != expected_top100_size:
            raise ValueError(
                f"{scoring_format} team Top 100 counts sum to {actual_size}; "
                f"expected {expected_top100_size}."
            )
    labels = roster_order["Team"].tolist()
    roster_ids = roster_order["Roster ID"].astype(str).tolist()
    colors = {
        position: plt.rcParams["axes.prop_cycle"].by_key()["color"][
            index % len(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        ]
        for index, position in enumerate(POSITIONS)
    }
    figures: list[Path] = []

    for scoring_format, filename in (
        ("Current scoring", "top100_players_by_team_current.png"),
        ("QB/TE PPR + RB/WR half-PPR", "top100_players_by_team_comparison.png"),
    ):
        selected = distribution.loc[
            distribution["Scoring format"].eq(scoring_format)
        ].set_index("Roster ID").reindex(roster_ids)
        fig, axis = plt.subplots(figsize=(12, 6))
        axis.bar(labels, selected["Top 100 player count"].fillna(0), color="#4C78A8")
        display_format = (
            "Current scoring"
            if scoring_format == "Current scoring"
            else "Proposed scoring"
        )
        axis.set_title(f"Rostered-player Top 100 by team — {display_format}")
        axis.set_xlabel("Team")
        axis.set_ylabel("Players in rostered Top 100")
        axis.tick_params(axis="x", labelrotation=35)
        fig.tight_layout()
        path = out / filename
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        figures.append(path)

    ordered_changes = changes.set_index("Roster ID").reindex(roster_ids)
    fig, axis = plt.subplots(figsize=(12, 6))
    values = ordered_changes["Top 100 count change"].fillna(0)
    bars = axis.bar(labels, values, color="#59A14F")
    for bar, value in zip(bars, values):
        if value:
            axis.annotate(
                f"{value:+.0f}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 3 if value > 0 else -12),
                textcoords="offset points",
                ha="center",
                va="bottom" if value > 0 else "top",
            )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_title("Change in rostered-player Top 100 count by team")
    axis.set_xlabel("Team")
    axis.set_ylabel("Comparison minus current players")
    axis.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    path = out / "top100_team_count_changes.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    figures.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    for axis, scoring_format in zip(
        axes, ("Current scoring", "QB/TE PPR + RB/WR half-PPR")
    ):
        selected = distribution.loc[
            distribution["Scoring format"].eq(scoring_format)
        ].set_index("Roster ID").reindex(roster_ids)
        bottom = np.zeros(len(roster_ids))
        x = np.arange(len(roster_ids))
        for position in POSITIONS:
            values = selected[f"{position} Top 100 count"].fillna(0).to_numpy(dtype=float)
            axis.bar(
                x, values, bottom=bottom, color=colors[position], label=position
            )
            bottom += values
        axis.set_xticks(x, labels, rotation=35, ha="right")
        axis.set_title(
            "Current scoring"
            if scoring_format == "Current scoring"
            else "Proposed scoring"
        )
        axis.set_xlabel("Team")
        axis.set_ylabel("Rostered Top 100 players")
        axis.legend(title="Position")
    fig.suptitle("Position composition of each team's rostered Top 100 players")
    fig.tight_layout()
    path = out / "top100_team_position_composition.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    figures.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    for axis, scoring_format in zip(
        axes, ("Current scoring", "QB/TE PPR + RB/WR half-PPR")
    ):
        selected = construction.loc[
            construction["Scoring format"].eq(scoring_format)
        ]
        pivot = selected.pivot(
            index="Roster ID",
            columns="Position",
            values="Season-to-date points on current roster",
        ).reindex(index=roster_ids, columns=POSITIONS).fillna(0)
        bottom = np.zeros(len(roster_ids))
        x = np.arange(len(roster_ids))
        for position in POSITIONS:
            values = pivot[position].to_numpy(dtype=float)
            axis.bar(x, values, bottom=bottom, color=colors[position], label=position)
            bottom += values
        axis.set_xticks(x, labels, rotation=35, ha="right")
        axis.set_title(
            "Current scoring"
            if scoring_format == "Current scoring"
            else "Proposed scoring"
        )
        axis.set_xlabel("Team")
        axis.set_ylabel("Season points on current roster")
        axis.legend(title="Position")
    fig.suptitle("Position production represented on current rosters")
    fig.tight_layout()
    path = out / "team_positional_production.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    figures.append(path)

    matchups = weekly_tables["weekly_matchup_rescoring"]
    if not matchups.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
        for axis, column, title, color in (
            (
                axes[0],
                "Current absolute margin",
                "Current scoring",
                "#4C78A8",
            ),
            (
                axes[1],
                "Comparison absolute margin",
                "Proposed scoring",
                "#F28E2B",
            ),
        ):
            axis.hist(
                pd.to_numeric(matchups[column], errors="coerce").dropna(),
                bins=12,
                color=color,
                alpha=0.85,
                edgecolor="white",
            )
            axis.set_title(title)
            axis.set_xlabel("Absolute matchup margin (points)")
            axis.set_ylabel("Matchups")
        fig.suptitle("Historical fixed-lineup matchup margin distributions")
        fig.tight_layout()
        path = out / "weekly_matchup_margin_distributions.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        figures.append(path)

        by_week = weekly_tables["weekly_by_week_summary"]
        fig, axis = plt.subplots(figsize=(12, 6))
        for scoring_format, color, marker in (
            ("Current scoring", "#4C78A8", "o"),
            ("QB/TE PPR + RB/WR half-PPR", "#F28E2B", "s"),
        ):
            selected = by_week.loc[
                by_week["Scoring format"].eq(scoring_format)
            ].sort_values("Week")
            axis.plot(
                selected["Week"],
                selected["Average absolute matchup margin"],
                color=color,
                marker=marker,
                label=(
                    "Current scoring"
                    if scoring_format == "Current scoring"
                    else "Proposed scoring"
                ),
            )
        axis.set_title("Average absolute matchup margin by completed week")
        axis.set_xlabel("Regular-season week")
        axis.set_ylabel("Average absolute margin (points)")
        axis.legend()
        fig.tight_layout()
        path = out / "weekly_matchup_margins_by_week.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        figures.append(path)
    return figures


def create_lineup_analysis_charts(
    tables: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> list[Path]:
    """Create the roster-depth and four-scenario lineup story charts."""
    required_tables = {
        "team_roster_depth",
        "team_lineup_scenarios",
        "team_lineup_impact",
        "team_lineup_selections",
        "team_viable_options",
        "team_flex_candidates",
    }
    missing = required_tables - set(tables)
    if missing:
        raise ValueError(f"Lineup analysis is missing tables: {sorted(missing)}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    scenarios = tables["team_lineup_scenarios"].copy()
    scenarios["Roster ID"] = scenarios["Roster ID"].astype(str)
    scenarios = scenarios.sort_values(
        "Roster ID", key=lambda column: pd.to_numeric(column, errors="coerce")
    )
    team_order_frame = scenarios.loc[
        scenarios["Scenario"].eq("A"), ["Roster ID", "Team"]
    ].drop_duplicates()
    if len(team_order_frame) != scenarios["Roster ID"].nunique():
        raise ValueError("Current-baseline lineup scenarios must include every team.")
    roster_ids = team_order_frame["Roster ID"].tolist()
    labels = team_order_frame["Team"].tolist()
    positions = ["QB", "RB", "WR", "TE", "K", "DEF/DST"]
    formats = ("Current scoring", "QB/TE PPR + RB/WR half-PPR")
    figures: list[Path] = []

    depth = tables["team_roster_depth"].copy()
    depth["Roster ID"] = depth["Roster ID"].astype(str)
    matrices = []
    for scoring_format in formats:
        selected = depth.loc[depth["Scoring format"].eq(scoring_format)]
        pivot = selected.pivot(
            index="Roster ID",
            columns="Position",
            values="Top 2 mean points per game",
        ).reindex(index=roster_ids, columns=positions)
        matrices.append(pivot.to_numpy(dtype=float))
    finite_values = np.concatenate(
        [matrix[np.isfinite(matrix)] for matrix in matrices]
    )
    maximum = float(finite_values.max()) if len(finite_values) else 1.0
    fig, axes = plt.subplots(
        1, 2, figsize=(16, 8), sharey=True, constrained_layout=True
    )
    for axis, matrix, scoring_format in zip(axes, matrices, formats):
        image = axis.imshow(
            np.ma.masked_invalid(matrix),
            aspect="auto",
            cmap="YlGnBu",
            vmin=0,
            vmax=maximum,
        )
        axis.set_xticks(np.arange(len(positions)), positions)
        axis.set_yticks(np.arange(len(labels)), labels)
        axis.set_title(
            "Current scoring"
            if scoring_format == "Current scoring"
            else "Proposed scoring"
        )
        axis.set_xlabel("Position")
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                if np.isfinite(value):
                    axis.text(
                        column_index,
                        row_index,
                        f"{value:.1f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="black",
                    )
    axes[0].set_ylabel("Team")
    fig.colorbar(
        image,
        ax=axes,
        label="Mean points per game among up to two rostered players",
    )
    fig.suptitle("Position-specific roster production, season to date")
    path = out / "team_roster_depth_top2_ppg.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    figures.append(path)

    count_pivot = (
        depth.loc[depth["Scoring format"].eq("Current scoring")]
        .pivot(index="Roster ID", columns="Position", values="Players with games played")
        .reindex(index=roster_ids, columns=positions)
        .fillna(0)
    )
    count_matrix = count_pivot.to_numpy(dtype=float)
    fig, axis = plt.subplots(figsize=(11, 8))
    image = axis.imshow(count_matrix, aspect="auto", cmap="Blues", vmin=0)
    axis.set_xticks(np.arange(len(positions)), positions)
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.set_xlabel("Position")
    axis.set_ylabel("Team")
    axis.set_title("Rostered players with a recorded game by position")
    for row_index in range(count_matrix.shape[0]):
        for column_index in range(count_matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                f"{int(count_matrix[row_index, column_index])}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
    fig.colorbar(image, ax=axis, label="Players with games played")
    fig.tight_layout()
    path = out / "team_roster_depth_player_counts.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    figures.append(path)

    scenario_order = ["A", "B", "C", "D"]
    scenario_labels = [
        "A Current", "B Scoring", "C Lineup", "D Combined"
    ]
    scenario_values = scenarios.pivot(
        index="Roster ID",
        columns="Scenario",
        values="Best lineup points per game",
    ).reindex(index=roster_ids, columns=scenario_order)
    x = np.arange(len(roster_ids), dtype=float)
    width = 0.19
    scenario_colors = ["#4C78A8", "#F28E2B", "#59A14F", "#B279A2"]
    fig, axis = plt.subplots(figsize=(16, 7))
    for index, (scenario, label, color) in enumerate(
        zip(scenario_order, scenario_labels, scenario_colors)
    ):
        axis.bar(
            x + (index - 1.5) * width,
            scenario_values[scenario].to_numpy(dtype=float),
            width,
            label=label,
            color=color,
        )
    axis.set_xticks(x, labels, rotation=35, ha="right")
    axis.set_ylabel("Best legal lineup points per game")
    axis.set_title("Retrospective optimal lineup production by team and scenario")
    axis.legend(ncol=4, frameon=False)
    fig.tight_layout()
    path = out / "team_best_lineup_four_scenarios.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    figures.append(path)

    impact = tables["team_lineup_impact"].copy()
    impact["Roster ID"] = impact["Roster ID"].astype(str)
    impact = impact.set_index("Roster ID").reindex(roster_ids)
    impact_columns = [
        "Scoring-only lineup change (B minus A, points per game)",
        "Lineup-only change (C minus A, points per game)",
        "Combined proposal change (D minus A, points per game)",
        "Interaction effect (D - B - C + A, points per game)",
    ]
    impact_labels = [
        "Scoring only",
        "Lineup only",
        "Combined",
        "Combined beyond separate changes",
    ]
    impact_values = impact[impact_columns].to_numpy(dtype=float)
    bound = max(float(np.nanmax(np.abs(impact_values))), 1.0)
    fig, axis = plt.subplots(figsize=(11, 8))
    image = axis.imshow(
        np.ma.masked_invalid(impact_values),
        aspect="auto",
        cmap="RdBu",
        vmin=-bound,
        vmax=bound,
    )
    axis.set_xticks(np.arange(len(impact_labels)), impact_labels)
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.set_title("Lineup points-per-game changes isolate scoring and slot effects")
    for row_index in range(impact_values.shape[0]):
        for column_index in range(impact_values.shape[1]):
            value = impact_values[row_index, column_index]
            if np.isfinite(value):
                axis.text(
                    column_index,
                    row_index,
                    f"{value:+.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )
    fig.colorbar(image, ax=axis, label="Change in best lineup points per game")
    fig.tight_layout()
    path = out / "team_lineup_change_components.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    figures.append(path)

    viable = tables["team_viable_options"].copy()
    viable["Roster ID"] = viable["Roster ID"].astype(str)
    viable = viable.loc[viable["Scenario"].isin(("A", "D"))]
    tolerances = sorted(viable["Tolerance (points per game)"].unique())
    fig, axes = plt.subplots(
        1, len(tolerances), figsize=(8 * len(tolerances), 8), sharey=True
    )
    if len(tolerances) == 1:
        axes = [axes]
    y = np.arange(len(roster_ids), dtype=float)
    for axis, tolerance in zip(axes, tolerances):
        subset = viable.loc[
            viable["Tolerance (points per game)"].eq(tolerance)
        ]
        values = subset.pivot(
            index="Roster ID",
            columns="Scenario",
            values="Additional viable options",
        ).reindex(index=roster_ids, columns=["A", "D"])
        axis.barh(y - 0.18, values["A"], height=0.34, label="A Current", color="#4C78A8")
        axis.barh(y + 0.18, values["D"], height=0.34, label="D Combined", color="#F28E2B")
        axis.set_title(f"Within {tolerance:g} points per game")
        axis.set_xlabel("Additional eligible rostered players")
        axis.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Team")
    axes[-1].legend(frameon=False)
    fig.suptitle("Roster options near a legal starter's points-per-game cutoff")
    fig.tight_layout()
    path = out / "team_viable_options_sensitivity.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    figures.append(path)

    gap_values = scenarios.pivot(
        index="Roster ID",
        columns="Scenario",
        values="Best starter-to-bench gap (points per game)",
    ).reindex(index=roster_ids, columns=["A", "D"])
    fig, axis = plt.subplots(figsize=(11, 8))
    for row_index, roster_id in enumerate(roster_ids):
        current_gap, combined_gap = gap_values.loc[roster_id, ["A", "D"]]
        if pd.notna(current_gap) and pd.notna(combined_gap):
            axis.plot(
                [current_gap, combined_gap],
                [row_index, row_index],
                color="#B0B0B0",
                linewidth=1,
                zorder=1,
            )
        if pd.notna(current_gap):
            axis.scatter(current_gap, row_index, color="#4C78A8", label="A Current" if row_index == 0 else None)
        if pd.notna(combined_gap):
            axis.scatter(combined_gap, row_index, color="#F28E2B", label="D Combined" if row_index == 0 else None)
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.invert_yaxis()
    axis.set_xlabel("Smallest legal starter-to-bench gap (points per game)")
    axis.set_title("Closest position-compatible bench alternative")
    axis.grid(axis="x", alpha=0.2)
    handles, legend_labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, legend_labels, frameon=False)
    fig.tight_layout()
    path = out / "team_starter_bench_cutoff_gap.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    figures.append(path)

    flex = tables["team_flex_candidates"].copy()
    flex["Roster ID"] = flex["Roster ID"].astype(str)
    current_flex = flex.loc[flex["Scoring format"].eq("Current scoring")].set_index(
        "Roster ID"
    ).reindex(roster_ids)
    flex_selections = tables["team_lineup_selections"]
    flex_selections = flex_selections.loc[
        flex_selections["Slot type"].eq("FLEX") & flex_selections["Filled"]
    ]
    selected_counts = (
        flex_selections.groupby(["Roster ID", "Scenario", "Position"])
        .size()
        .rename("Players")
        .reset_index()
    )
    flex_positions = ("RB", "WR", "TE")
    flex_scenario_columns = [
        (scenario, position)
        for scenario in scenario_order
        for position in flex_positions
    ]
    selected_matrix = (
        selected_counts.pivot_table(
            index="Roster ID",
            columns=["Scenario", "Position"],
            values="Players",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(index=roster_ids, columns=pd.MultiIndex.from_tuples(flex_scenario_columns))
        .fillna(0)
        .to_numpy(dtype=float)
    )
    fig, axes = plt.subplots(1, 2, figsize=(19, 8))
    bottom = np.zeros(len(roster_ids), dtype=float)
    flex_colors = {"RB": "#4C78A8", "WR": "#F28E2B", "TE": "#59A14F"}
    for position in ("RB", "WR", "TE"):
        counts = current_flex[f"{position} candidates with games"].fillna(0).to_numpy(dtype=float)
        axes[0].bar(
            x,
            counts,
            bottom=bottom,
            label=position,
            color=flex_colors[position],
        )
        bottom += counts
    axes[0].set_xticks(x, labels, rotation=35, ha="right")
    axes[0].set_ylabel("Rostered FLEX-eligible players with games")
    axes[0].set_title("Available RB / WR / TE candidates")
    axes[0].legend(title="Position", frameon=False)
    image = axes[1].imshow(selected_matrix, aspect="auto", cmap="YlGnBu", vmin=0)
    axes[1].set_xticks(
        np.arange(len(flex_scenario_columns)),
        [f"{scenario}\n{position}" for scenario, position in flex_scenario_columns],
    )
    axes[1].set_yticks(np.arange(len(labels)), labels)
    axes[1].set_xlabel("Scenario and position assigned to standard FLEX")
    axes[1].set_title("Optimized standard FLEX player positions")
    for row_index in range(selected_matrix.shape[0]):
        for column_index in range(selected_matrix.shape[1]):
            count = int(selected_matrix[row_index, column_index])
            if count:
                axes[1].text(
                    column_index,
                    row_index,
                    str(count),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )
    fig.colorbar(image, ax=axes[1], label="Players assigned")
    fig.suptitle("FLEX-eligible roster depth and optimized FLEX usage")
    fig.tight_layout()
    path = out / "team_flex_candidate_depth.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    figures.append(path)
    return figures
