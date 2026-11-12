from pathlib import Path

import pandas as pd


def _ordered(group: pd.DataFrame, points_column: str) -> pd.DataFrame:
    result = group.copy()
    result["_sort_name"] = result["full_name"].fillna("").astype(str).str.casefold()
    result["_sort_sleeper_id"] = (
        result["sleeper_id"].fillna("").astype(str)
        if "sleeper_id" in result.columns
        else ""
    )
    if "receptions" in result:
        result["_sort_receptions"] = pd.to_numeric(
            result["receptions"], errors="coerce"
        ).fillna(0)
    elif "rec" in result:
        result["_sort_receptions"] = pd.to_numeric(
            result["rec"], errors="coerce"
        ).fillna(0)
    else:
        result["_sort_receptions"] = 0
    result["_sort_points"] = pd.to_numeric(
        result[points_column], errors="coerce"
    ).fillna(float("-inf"))
    return result.sort_values(
        [
            "_sort_points", "_sort_receptions", "_sort_name", "_sort_sleeper_id",
        ],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).drop(columns=[
        "_sort_name", "_sort_sleeper_id", "_sort_receptions", "_sort_points",
    ])


def add_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """Assign deterministic positional and overall ranks for both formats."""
    result = df.copy().reset_index(drop=True)
    for rank_column, points_column in [
        ("current_rank", "current_points"),
        ("comparison_rank", "comparison_points"),
    ]:
        result[rank_column] = 0
        for _, group in result.groupby("position", sort=True):
            ordered = _ordered(group, points_column)
            result.loc[ordered.index, rank_column] = range(1, len(ordered) + 1)

    for rank_column, points_column in [
        ("current_overall_rank", "current_points"),
        ("comparison_overall_rank", "comparison_points"),
    ]:
        ordered = _ordered(result, points_column)
        result.loc[ordered.index, rank_column] = range(1, len(ordered) + 1)

    result["rank_movement"] = result["current_rank"] - result["comparison_rank"]
    result["point_difference"] = (
        result["comparison_points"] - result["current_points"]
    ).round(3)
    return result


def make_outputs(
    ranked: pd.DataFrame,
    out_dir: str | Path = "outputs",
    positional_limit: int = 100,
    overall_limit: int = 250,
) -> pd.DataFrame:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ranked.loc[ranked["current_overall_rank"] <= overall_limit].sort_values(
        "current_overall_rank"
    ).to_csv(out / "top250_current_scoring.csv", index=False)
    ranked.loc[ranked["comparison_overall_rank"] <= overall_limit].sort_values(
        "comparison_overall_rank"
    ).to_csv(out / "top250_half_ppr_te.csv", index=False)

    positional_rows = []
    for _, group in ranked.groupby("position", sort=True):
        current_ids = set(group.loc[group["current_rank"] <= positional_limit, "sleeper_id"])
        comparison_ids = set(
            group.loc[group["comparison_rank"] <= positional_limit, "sleeper_id"]
        )
        union_ids = current_ids | comparison_ids
        positional_rows.append(group[group["sleeper_id"].isin(union_ids)])

    positional = pd.concat(positional_rows, ignore_index=True)
    positional.sort_values(
        ["position", "comparison_rank", "current_rank"], kind="mergesort"
    ).to_csv(out / "positional_top100_comparison.csv", index=False)

    ranked.sort_values(
        ["rank_movement", "position", "comparison_rank"], ascending=[False, True, True]
    ).head(25).to_csv(out / "biggest_risers.csv", index=False)
    ranked.sort_values(
        ["rank_movement", "position", "current_rank"], ascending=[True, True, True]
    ).head(25).to_csv(out / "biggest_fallers.csv", index=False)
    return positional
