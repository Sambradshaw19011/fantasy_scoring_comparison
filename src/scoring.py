import math
from collections.abc import Mapping

import pandas as pd

# These are raw Sleeper stat fields that correspond to scoring_settings keys.
STAT_KEY_MAP = {
    key: key
    for key in {
        "pass_cmp", "pass_att", "pass_yd", "pass_td", "pass_int", "pass_sack",
        "pass_2pt", "pass_fd", "pass_cmp_25", "bonus_pass_yd_300",
        "bonus_pass_yd_400", "rush_att", "rush_yd", "rush_td", "rush_2pt",
        "rush_fd", "bonus_rush_att_20", "bonus_rush_yd_100",
        "bonus_rush_yd_200", "bonus_rush_td_qb", "bonus_rush_rec_yd_100",
        "bonus_rush_rec_yd_200", "bonus_fd_qb", "bonus_fd_rb", "bonus_fd_wr",
        "bonus_fd_te", "rec", "rec_tgt", "rec_yd", "rec_td", "rec_2pt", "rec_fd",
        "bonus_rec_rb", "bonus_rec_wr", "bonus_rec_yd_100", "bonus_rec_yd_200", "fum",
        "fum_lost", "fum_rec", "fum_rec_td", "fgm", "fgmiss", "fgm_0_19",
        "fgm_20_29", "fgm_30_39", "fgm_40_49", "fgm_50_59", "fgm_60p",
        "fgmiss_0_19", "fgmiss_20_29", "fgmiss_30_39", "fgmiss_40_49",
        "fgmiss_50_59", "fgmiss_60p", "xpm", "xpmiss", "blk_kick", "st_td",
        "st_ff", "st_fum_rec", "def_st_ff", "def_st_fum_rec", "def_st_td",
        "def_td", "ff", "int", "sack", "safe", "pts_allow_0", "pts_allow_1_6",
        "pts_allow_7_13", "pts_allow_14_20", "pts_allow_21_27",
        "pts_allow_28_34", "pts_allow_35p", "bonus_rec_te",
    }
}
RECEPTION_SCORING_POSITIONS = {"QB", "RB", "WR", "TE"}


def _setting_value(settings: Mapping, key: str) -> float:
    value = settings.get(key, 0)
    if value is None:
        return 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Sleeper scoring value for {key!r} is not numeric: {value!r}"
        ) from exc
    if not math.isfinite(numeric):
        raise ValueError(f"Sleeper scoring value for {key!r} must be finite.")
    return numeric


def _stat_series(df: pd.DataFrame, key: str) -> pd.Series:
    if key not in df:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[key], errors="coerce").fillna(0.0)


def score_with_sleeper_settings(
    df: pd.DataFrame,
    scoring_settings: Mapping,
    reception_points: float | None = None,
    te_premium: float | None = None,
    qb_reception_points: float | None = None,
) -> pd.Series:
    """Score player rows from Sleeper raw stats, never its precomputed point totals."""
    points = pd.Series(0.0, index=df.index)
    for setting_key in scoring_settings:
        if setting_key not in STAT_KEY_MAP or setting_key in {"rec", "bonus_rec_te"}:
            continue
        if setting_key in df:
            points += _stat_series(df, STAT_KEY_MAP[setting_key]) * _setting_value(
                scoring_settings, setting_key
            )

    reception_multiplier = (
        _setting_value(scoring_settings, "rec")
        if reception_points is None else float(reception_points)
    )
    receptions = _stat_series(df, "rec")
    if reception_points is not None and "position" in df:
        receptions = receptions.where(
            df["position"].isin(RECEPTION_SCORING_POSITIONS), 0.0
        )
    points += receptions * reception_multiplier

    if qb_reception_points is not None and "position" in df:
        qb_multiplier = float(qb_reception_points)
        if not math.isfinite(qb_multiplier):
            raise ValueError("QB reception points must be finite.")
        is_qb = df["position"].eq("QB")
        points += receptions.where(is_qb, 0.0) * (
            qb_multiplier - reception_multiplier
        )

    if te_premium is None:
        is_te = (
            df["position"].eq("TE").astype(float)
            if "position" in df
            else pd.Series(0.0, index=df.index)
        )
        derived_te_receptions = receptions * is_te
        if "bonus_rec_te" in df:
            te_receptions = _stat_series(df, "bonus_rec_te").where(
                df["bonus_rec_te"].notna(), derived_te_receptions
            )
        else:
            te_receptions = derived_te_receptions
        points += te_receptions * _setting_value(scoring_settings, "bonus_rec_te")
    else:
        is_te = (
            df["position"].eq("TE").astype(float)
            if "position" in df
            else pd.Series(0.0, index=df.index)
        )
        points += receptions * is_te * float(te_premium)

    return points.round(3)


def scoring_settings_summary(
    settings: Mapping,
    source_stat_keys: set[str] | list[str] | None = None,
) -> pd.DataFrame:
    """Audit every league setting against fields in Sleeper's raw stats response."""
    available = set(source_stat_keys or [])
    rows = []
    for key, value in sorted(settings.items()):
        numeric_value = _setting_value(settings, key)
        if key == "bonus_rec_te":
            source_stat = "bonus_rec_te (TE reception count; rec fallback)"
            supported = "bonus_rec_te" in available or "rec" in available
        else:
            source_stat = STAT_KEY_MAP.get(key, "not mapped")
            supported = key in STAT_KEY_MAP and key in available

        if not supported:
            status = "unsupported"
            assumption = (
                "The Sleeper stats response has no mapped raw stat for this setting."
            )
        else:
            status = "applied"
            assumption = ""
        if numeric_value == 0 and not supported:
            status = "not relevant (zero scoring value)"

        rows.append({
            "setting": key,
            "points_per_stat": value,
            "source_stat": source_stat,
            "status": status,
            "assumptions": assumption,
        })
    return pd.DataFrame(
        rows,
        columns=[
            "setting", "points_per_stat", "source_stat", "status", "assumptions"
        ],
    )
