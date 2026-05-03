"""
A Feature-Level Examination of User Sentiment Toward Gamification Mechanics
in Duolingo.

End-to-end pipeline behind the final report:

  raw scrapes  ->  gamification flagging  ->  stratified sampling  ->
  labeling templates  ->  manual + LLM sentiment labels  ->
  feature-engineered analytical dataset  ->  logistic regression.

Run from the repo root.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
ANL = ROOT / "data" / "analysis"


# ---------------------------------------------------------------------------
# 1. Load raw scrapes
# ---------------------------------------------------------------------------
def load_raw():
    df_google_play = pd.read_csv(RAW / "dataset_googleplay-reviews.csv", dtype=str)
    df_app_store = pd.read_csv(RAW / "dataset_appstore-appstore.csv", dtype=str)
    df_twitter = pd.read_csv(RAW / "dataset_tweet-twitter.csv", dtype=str)
    df_reddit = pd.concat(
        [
            pd.read_csv(RAW / "dataset_reddit-scraper-r_duolingo.csv", dtype=str),
            pd.read_csv(RAW / "dataset_reddit-scraper-pro_r_duolingomeme.csv", dtype=str),
        ],
        ignore_index=True,
    )
    return df_google_play, df_app_store, df_twitter, df_reddit


# ---------------------------------------------------------------------------
# 2. Text cleaning + gamification keyword flagging
# ---------------------------------------------------------------------------
STREAK_WORDS = [
    "streak", "streaks", "lost streak", "streak freeze", "freeze",
    "broken streak", "recover streak", "streak society",
]
HEARTS_WORDS = [
    "heart", "hearts", "life", "lives", "mistake limit", "mistakes",
    "ran out of hearts", "no hearts", "heart refill",
]
ENERGY_WORDS = ["energy", "out of energy", "energy system", "energy refill"]
XP_WORDS = [
    "xp", "points", "daily xp", "xp boost", "double xp", "2x xp",
    "xp chest", "level", "level up", "leveling", "score",
]
LEADERBOARD_WORDS = [
    "leaderboard", "league", "promotion", "demotion",
    "bronze", "silver", "gold", "diamond league",
    "rank", "ranking", "top 10", "top 3",
    "competition", "competitive", "race", "beat", "overtake",
]
CHESS_WORDS = [
    "chess", "chess mode", "chess puzzle", "chess puzzles",
    "duolingo chess", "chess lesson",
]
QUEST_WORDS = [
    "quest", "quests", "daily quest", "weekly quest",
    "challenge", "challenges", "friend quest", "partner quest",
    "task", "achievements", "milestone",
]
REWARD_WORDS = [
    "reward", "rewards", "gems", "gem", "coin", "coins",
    "lingots", "chest", "treasure chest", "loot",
    "unlock", "unlocked", "bonus", "boost", "super boost",
]

FEATURE_GROUPS = {
    "feat_streak": STREAK_WORDS,
    "feat_hearts": HEARTS_WORDS,
    "feat_energy": ENERGY_WORDS,
    "feat_xp": XP_WORDS,
    "feat_leaderboard": LEADERBOARD_WORDS,
    "feat_chess": CHESS_WORDS,
    "feat_quests": QUEST_WORDS,
    "feat_rewards": REWARD_WORDS,
}


def choose_text_col(df, candidates, platform_name=""):
    for c in candidates:
        if c in df.columns:
            print(f"[{platform_name}] using text column: {c}")
            return c
    raise ValueError(
        f"No candidate text column found in {platform_name}. "
        f"Available columns: {list(df.columns)}"
    )


def clean_text_column(df, text_col):
    df = df.copy()
    df[text_col] = df[text_col].fillna("").astype(str)
    df[text_col + "_clean"] = df[text_col].str.lower()
    return df


def _build_pattern(keywords):
    return "(" + "|".join(re.escape(k.lower()) for k in keywords) + ")"


def add_gamification_flags(df, clean_text_col):
    df = df.copy()
    text = df[clean_text_col]
    for col, words in FEATURE_GROUPS.items():
        df[col] = text.str.contains(_build_pattern(words), regex=True, na=False)
    df["any_gamification"] = df[list(FEATURE_GROUPS)].any(axis=1)
    return df


# ---------------------------------------------------------------------------
# 3. Build the gamification-only superset
# ---------------------------------------------------------------------------
def build_gamification_superset(df_google_play, df_app_store, df_twitter, df_reddit):
    gp_col = choose_text_col(
        df_google_play,
        ["content", "review", "text", "body", "review_text"],
        "Google Play",
    )
    app_col = choose_text_col(
        df_app_store,
        ["review", "reviewText", "body", "text", "content", "title"],
        "App Store",
    )
    tw_col = choose_text_col(df_twitter, ["full_text", "text", "content"], "Twitter")
    rd_col = choose_text_col(df_reddit, ["body", "text", "content"], "Reddit")

    df_google_play = clean_text_column(df_google_play, gp_col)
    df_app_store = clean_text_column(df_app_store, app_col)
    df_twitter = clean_text_column(df_twitter, tw_col)
    df_reddit = clean_text_column(df_reddit, rd_col)

    df_google_play = add_gamification_flags(df_google_play, gp_col + "_clean")
    df_app_store = add_gamification_flags(df_app_store, app_col + "_clean")
    df_twitter = add_gamification_flags(df_twitter, tw_col + "_clean")
    df_reddit = add_gamification_flags(df_reddit, rd_col + "_clean")

    df_google_play_gam = df_google_play[df_google_play["any_gamification"]].copy()
    df_app_store_gam = df_app_store[df_app_store["any_gamification"]].copy()
    df_twitter_gam = df_twitter[df_twitter["any_gamification"]].copy()
    df_reddit_gam = df_reddit[df_reddit["any_gamification"]].copy()

    df_google_play_gam["platform"] = "google_play"
    df_app_store_gam["platform"] = "app_store"
    df_twitter_gam["platform"] = "twitter"
    df_reddit_gam["platform"] = "reddit"

    df_all_gam = pd.concat(
        [df_google_play_gam, df_app_store_gam, df_twitter_gam, df_reddit_gam],
        ignore_index=True,
    )
    print("Per-platform gamification counts:")
    print(df_all_gam["platform"].value_counts())
    return df_all_gam


# ---------------------------------------------------------------------------
# 4. Stratified sample of 3,000 weighted by the shared cross-platform window
# ---------------------------------------------------------------------------
TARGET_COUNTS = {
    "google_play": 1740,
    "app_store": 360,
    "reddit": 780,
    "twitter": 120,
}


def stratified_sample(df_all_gam, seed=42):
    parts = []
    for platform, n in TARGET_COUNTS.items():
        subset = df_all_gam[df_all_gam["platform"] == platform]
        n = min(n, len(subset))
        parts.append(subset.sample(n=n, random_state=seed))
    df_sampled = pd.concat(parts, ignore_index=True)
    print("Sample composition:")
    print(df_sampled["platform"].value_counts())
    return df_sampled


# ---------------------------------------------------------------------------
# 5. Labeling templates (manual + LLM-assisted; produced once, then hand-edited)
# ---------------------------------------------------------------------------
def write_labeling_template(df_sampled):
    feat_cols = list(FEATURE_GROUPS)
    cols = ["Date", "platform", "text", "text_clean"] + feat_cols
    template = df_sampled[cols].copy()
    template["feat_energy_vs_heart"] = ""
    template["sentiment_label"] = ""
    out = PROC / "labeling_template_3000.csv"
    template.to_csv(out, index=False, encoding="utf-8")
    print(f"Wrote {out}")


def fix_labeling_template_text(df_sampled):
    """Reddit rows had empty `text`; backfill from the original `body` column."""
    df_label = pd.read_csv(PROC / "labeling_template_3000.csv", encoding="latin1")
    df_label["date"] = df_sampled["date"]
    df_label["body"] = df_sampled["body"]
    mask = (df_label["platform"].str.lower() == "reddit") & (
        df_label["text"].isna() | (df_label["text"].astype(str).str.strip() == "")
    )
    df_label.loc[mask, "text"] = df_label.loc[mask, "body"]
    df_label.drop(columns=["body"], inplace=True)
    out = PROC / "labeling_template_3000_fixed.csv"
    df_label.to_csv(out, index=False, encoding="utf-8")
    print(f"Wrote {out}")


def fix_labeling_template_dates():
    """App Store rows were missing top-level Date; backfill from `date`."""
    df = pd.read_csv(PROC / "labeling_template_3000_fixed.csv", encoding="latin1")
    df["Date_fixed"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    mask_app = df["platform"].str.lower().str.contains("app", na=False)
    mask_missing = mask_app & (
        df["Date"].isna() | (df["Date"].astype(str).str.strip() == "")
    )
    df.loc[mask_missing, "Date"] = df.loc[mask_missing, "Date_fixed"].astype(str)
    df.drop(columns=["Date_fixed"], inplace=True)
    out = PROC / "labeling_template_3000_final.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# 6. Build the final analytical dataset from `labeling_final.csv`
# ---------------------------------------------------------------------------
def build_analytical_dataset():
    df = pd.read_csv(ANL / "labeling_final.csv")

    df = df[df["exclude"] != 1]

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Date"].dt.year == 2025]

    df = df[df["sentiment"] != "neu"]

    df["sentiment_binary"] = df["sentiment"].map({"pos": 1, "neg": 0})
    df["text_length"] = df["text"].astype(str).str.len()

    platform_dummies = pd.get_dummies(df["platform"], prefix="plat")
    df = pd.concat([df, platform_dummies], axis=1)

    df["year_month"] = df["Date"].dt.to_period("M").astype(str)
    month_dummies = pd.get_dummies(df["year_month"], prefix="t")
    df = pd.concat([df, month_dummies], axis=1)

    keep_cols = (
        [
            "text", "sentiment_binary", "text_length",
            "feat_streak", "feat_hearts", "feat_energy", "feat_xp",
            "feat_leaderboard", "feat_chess", "feat_quests",
            "feat_rewards", "feat_energy_vs_heart",
        ]
        + list(platform_dummies.columns)
        + list(month_dummies.columns)
    )
    final_df = df[keep_cols]
    out = ANL / "ready_to_run.csv"
    final_df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(final_df)} rows)")
    return final_df


# ---------------------------------------------------------------------------
# 7. Logistic regression (matches Section 3.1 / 3.2 of the report)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "feat_streak", "feat_hearts", "feat_energy", "feat_xp",
    "feat_leaderboard", "feat_chess", "feat_quests",
    "feat_rewards", "feat_energy_vs_heart",
    "plat_google_play", "plat_app_store", "plat_reddit", "plat_twitter",
    "t_2025-07", "t_2025-08", "t_2025-09",
    "t_2025-10", "t_2025-11", "t_2025-12",
    "text_length",
]


def fit_logit():
    df = pd.read_csv(ANL / "ready_to_run_num.csv")
    y = df["sentiment_binary"].astype(int)
    X = df[FEATURE_COLS].astype(float)
    X = sm.add_constant(X)
    result = sm.Logit(y, X).fit()
    print(result.summary())
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    df_google_play, df_app_store, df_twitter, df_reddit = load_raw()
    df_all_gam = build_gamification_superset(
        df_google_play, df_app_store, df_twitter, df_reddit
    )
    df_all_gam.to_csv(PROC / "final_gamification_dataset.csv", index=False, encoding="utf-8")

    df_sampled = stratified_sample(df_all_gam)
    df_sampled.to_csv(PROC / "sampled_3000_for_labeling.csv", index=False, encoding="utf-8")

    write_labeling_template(df_sampled)
    fix_labeling_template_text(df_sampled)
    fix_labeling_template_dates()

    # Manual + LLM labels are added externally to produce labeling_final.csv,
    # which is committed under data/analysis/. The remaining steps consume it.
    build_analytical_dataset()
    fit_logit()


if __name__ == "__main__":
    main()
