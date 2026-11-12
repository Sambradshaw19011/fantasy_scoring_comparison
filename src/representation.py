from pathlib import Path

import pandas as pd

POSITIONS = ("QB", "RB", "WR", "TE", "K", "DEF/DST")
TIERS = (10, 20, 50, 75, 100)
SCORING_FORMATS = (
    ("Current scoring", "current_overall_rank", "current_points"),
    ("QB/TE PPR + RB/WR half-PPR", "comparison_overall_rank", "comparison_points"),
)


def build_positional_distributions(
    ranked: pd.DataFrame,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """Count positions within each overall tier for both scoring formats."""
    required = {
        "sleeper_id", "full_name", "position",
        "current_overall_rank", "comparison_overall_rank",
        "current_points", "comparison_points",
    }
    missing = required - set(ranked.columns)
    if missing:
        raise ValueError(
            f"Overall positional distributions are missing columns: {sorted(missing)}"
        )
    if ranked["sleeper_id"].isna().any() or ranked["sleeper_id"].astype(str).str.strip().eq("").any():
        raise ValueError("The overall player universe contains a missing Sleeper ID.")
    if ranked["sleeper_id"].astype(str).duplicated().any():
        raise ValueError("The overall player universe contains duplicate normalized Sleeper IDs.")
    if ranked["position"].isna().any() or not set(ranked["position"]).issubset(POSITIONS):
        raise ValueError("The overall player universe contains an unsupported position.")
    for rank_column in ("current_overall_rank", "comparison_overall_rank"):
        ranks = pd.to_numeric(ranked[rank_column], errors="coerce")
        if ranks.isna().any() or set(ranks) != set(range(1, len(ranked) + 1)):
            raise ValueError(
                f"The overall player universe has invalid or duplicate {rank_column} values."
            )

    distribution_by_format: dict[str, pd.DataFrame] = {}
    tier_players: dict[tuple[str, int], pd.DataFrame] = {}
    for label, rank_column, _ in SCORING_FORMATS:
        rows = []
        for tier in TIERS:
            selected = ranked.loc[ranked[rank_column] <= tier]
            expected_size = min(tier, len(ranked))
            if len(selected) != expected_size:
                raise ValueError(
                    f"{label} Top {tier} has {len(selected)} players; "
                    f"expected {expected_size}. Check the overall ranks."
                )
            if selected["sleeper_id"].nunique() != expected_size:
                raise ValueError(f"{label} Top {tier} includes duplicate players.")
            tier_players[(label, tier)] = selected
            counts = selected["position"].value_counts()
            row = {"Tier": f"Top {tier}"}
            row.update({position: int(counts.get(position, 0)) for position in POSITIONS})
            row["Total"] = int(sum(row[position] for position in POSITIONS))
            rows.append(row)
        distribution_by_format[label] = pd.DataFrame(
            rows, columns=["Tier", *POSITIONS, "Total"]
        )

    current = distribution_by_format["Current scoring"]
    comparison = distribution_by_format["QB/TE PPR + RB/WR half-PPR"]
    change_rows = []
    percentage_rows = []
    membership_rows = []
    for tier in TIERS:
        tier_label = f"Top {tier}"
        current_row = current.loc[current["Tier"].eq(tier_label)].iloc[0]
        comparison_row = comparison.loc[comparison["Tier"].eq(tier_label)].iloc[0]
        change_row = {"Tier": tier_label}
        tier_size = int(current_row["Total"])
        for position in POSITIONS:
            current_count = int(current_row[position])
            comparison_count = int(comparison_row[position])
            current_percentage = current_count / tier_size * 100 if tier_size else 0.0
            comparison_percentage = (
                comparison_count / tier_size * 100 if tier_size else 0.0
            )
            change_row[f"{position} current count"] = current_count
            change_row[f"{position} comparison count"] = comparison_count
            change_row[f"{position} count change"] = comparison_count - current_count
            change_row[f"{position} current percentage"] = current_percentage
            change_row[f"{position} comparison percentage"] = comparison_percentage
            change_row[f"{position} percentage point change"] = (
                comparison_percentage - current_percentage
            )
            for scoring_format, count, percentage in (
                ("Current scoring", current_count, current_percentage),
                ("QB/TE PPR + RB/WR half-PPR", comparison_count, comparison_percentage),
            ):
                percentage_rows.append({
                    "Tier": tier_label,
                    "Scoring format": scoring_format,
                    "Position": position,
                    "Count": count,
                    "Percentage": percentage,
                })
        change_row["Total current"] = int(current_row["Total"])
        change_row["Total comparison"] = int(comparison_row["Total"])
        change_row["Total change"] = (
            int(comparison_row["Total"]) - int(current_row["Total"])
        )
        change_rows.append(change_row)

        current_players = tier_players[("Current scoring", tier)]
        comparison_players = tier_players[("QB/TE PPR + RB/WR half-PPR", tier)]
        current_ids = set(current_players["sleeper_id"])
        comparison_ids = set(comparison_players["sleeper_id"])
        membership_status = {
            **{player_id: "entering" for player_id in comparison_ids - current_ids},
            **{player_id: "exiting" for player_id in current_ids - comparison_ids},
            **{player_id: "retained" for player_id in current_ids & comparison_ids},
        }
        changed = ranked.assign(
            _membership_id=ranked["sleeper_id"].astype(str)
        ).set_index("_membership_id", drop=False)
        for player_id in sorted(membership_status):
            player = changed.loc[player_id]
            membership_rows.append({
                "Tier": tier_label,
                "Membership change": membership_status[player_id],
                "Sleeper ID": player["sleeper_id"],
                "Player": player["full_name"],
                "Position": player["position"],
                "Current overall rank": int(player["current_overall_rank"]),
                "Comparison overall rank": int(player["comparison_overall_rank"]),
                "Current points": float(player["current_points"]),
                "Comparison points": float(player["comparison_points"]),
                "Rank movement": int(
                    player["current_overall_rank"]
                    - player["comparison_overall_rank"]
                ),
            })

    percentages = pd.DataFrame(
        percentage_rows,
        columns=["Tier", "Scoring format", "Position", "Count", "Percentage"],
    )
    membership = pd.DataFrame(
        membership_rows,
        columns=[
            "Tier", "Membership change", "Sleeper ID", "Player", "Position",
            "Current overall rank", "Comparison overall rank",
            "Current points", "Comparison points", "Rank movement",
        ],
    )
    changes = pd.DataFrame(change_rows)
    return current, comparison, changes, percentages, membership


def export_overall_rankings(ranked: pd.DataFrame, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ranked.sort_values("current_overall_rank", kind="mergesort").to_csv(
        out / "overall_rankings_current.csv", index=False
    )
    ranked.sort_values("comparison_overall_rank", kind="mergesort").to_csv(
        out / "overall_rankings_half_ppr_te.csv", index=False
    )


def export_positional_distributions(
    ranked: pd.DataFrame,
    out_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    current, comparison, changes, percentages, membership = (
        build_positional_distributions(ranked)
    )
    current.to_csv(out / "positional_distribution_current.csv", index=False)
    comparison.to_csv(out / "positional_distribution_half_ppr_te.csv", index=False)
    changes.to_csv(out / "positional_distribution_changes.csv", index=False)
    percentage_point_changes = changes[
        [
            "Tier",
            *[f"{position} percentage point change" for position in POSITIONS],
        ]
    ]
    percentage_point_changes.to_csv(
        out / "positional_percentage_point_changes.csv", index=False
    )
    percentages.to_csv(out / "positional_distribution_percentages.csv", index=False)
    membership.to_csv(out / "tier_membership_changes.csv", index=False)
    export_overall_rankings(ranked, out)
    return current, comparison, changes, percentages, membership
