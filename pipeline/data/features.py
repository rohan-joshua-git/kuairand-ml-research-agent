"""
Feature engineering — this module is the primary surface the agent rewrites
each iteration (Figure 1 "engineer features" stage).

`build_features` is intentionally a single, clearly-named entrypoint so the
agent's code diffs have one obvious place to land: it can add columns,
change encodings, or swap out the whole body, and `ablation.py` can diff
this file's contents across iterations to explain what changed.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from pipeline.data.leakage_guard import drop_leaky_columns

# duration_ms is an ITEM property (video length) — known before exposure, safe.
# play_time_ms, profile_stay_time, comment_stay_time, is_profile_enter are all
# OUTCOMES of the impression being predicted (measured during/after the view),
# so feeding them is label leakage, not modeling. Measured on real validation
# data: play_time_ms corr 0.64 with long_view; comment_stay_time corr 0.17
# (staying in the comments implies you long-viewed); profile_stay_time corr ~0
# but same causal class. The official baseline's field list uses duration_ms
# and none of the outcome fields — same conclusion.
NUMERIC_SIGNAL_COLUMNS = [
    "duration_ms",
]

# Categorical columns this module ADDS that the model should encode.
#
# READ THIS BEFORE ADDING A FEATURE: pipeline/train.py encodes the official
# baseline's five fields plus whatever is registered here. A new categorical
# column that is not registered here is computed and then silently ignored by
# the encoder — it will produce a bit-identical score and waste the iteration.
# (This happened twice in the run of 2026-08-31: two different patches adding a
# video-quality prior both scored exactly 0.6024, because the column was never
# encoded.) So: add the column to build_features AND append its name here, in
# the same edit.
EXTRA_CATEGORICAL_FIELDS = [
    "pos_bucket",
    "tag1",
    "upload_type",
]  # + USER_FEATURE_FIELDS appended below once they are defined

AUXILIARY_LABEL_COLUMNS = [
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
]


@lru_cache(maxsize=1)
def load_video_basic_features() -> pd.DataFrame | None:
    """`video_features_basic_pure.csv` (video_id -> author_id): static
    attributes fixed at upload time, so not leaky — unlike the statistic file
    (see leakage_guard.py). author_id is the one item-side field the official
    baseline uses; `tag` and `upload_type` are added because they are the only
    columns in this file that pool videos into GROUPS. Measured on validation
    (target-encoded on train, standalone within-user GAUC): author_id 0.6367 and
    music_id 0.6365 look strong but correlate 0.985 / 0.987 with the video_id
    encoding — 87% of authors and 98% of music_ids own exactly ONE video, so
    both are the video identity in disguise and add nothing over the video_id
    embedding the model already has. tag correlates only 0.458 with it at GAUC
    0.5604, and upload_type 0.5214. Those two are the real pooling levels.
    video_type (0.5014), music_type (0.5065) and visible_status (constant) are
    left out as measured-dead. Cached: read once per process, not once per call."""
    from pipeline.data.loader import load_config

    cfg = load_config()
    path = Path(cfg["dataset"]["raw_dir"]) / cfg["dataset"]["features"]["video_basic"]
    if not path.exists():
        return None
    return pd.read_csv(path, usecols=["video_id", "author_id", "tag", "upload_type"])


# User-side metadata (`user_features_pure.csv`, 27,285 users). Every one of
# these is CONSTANT within a user, so none of them can move a within-user
# ranking on its own — `user_id` target-encoded scores exactly 0.5000. They earn
# their place only through INTERACTIONS: the FM's second-order term gives
# <e_user_attr, e_video> and <e_user_attr, e_tab> for free, which lets the model
# learn "users of this type are selective in this tab" as a prior SHARED across
# users. That is the thing a raw `user_id` embedding cannot do for a user with
# few training rows. The same property makes them leakage-free for this metric:
# a within-user-constant column cannot encode within-user order.
#
# Excluded on measurement: `is_lowactive_period` has ONE distinct value, and the
# raw counts (`follow_user_num`, `fans_user_num`, `friend_user_num`,
# `register_days`) are dropped in favour of the dataset's own `*_range` buckets,
# which are the same information at a cardinality an embedding can actually fit.
USER_CORE_FIELDS = [
    "user_active_degree", "is_live_streamer", "is_video_author",
    "follow_user_num_range", "fans_user_num_range", "friend_user_num_range",
    "register_days_range",
]
# Anonymised user attributes. feat3 (1,471 values), feat8 (454) and feat7 (118)
# are held out of the default set as high-cardinality-per-user risks.
USER_ONEHOT_FIELDS = [f"onehot_feat{i}" for i in (0, 1, 2, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17)]
USER_FEATURE_FIELDS = USER_CORE_FIELDS + USER_ONEHOT_FIELDS

# MEASURED NEUTRAL — not registered by default. Encoding these cost -0.0002
# with the 7 core fields and gained +0.0001 with all 22 (3 seeds each, control
# 0.6047 std 0.0001), while the 7-field arm TRIPLED seed variance. A conditional
# probe agrees and is well powered: holding the video-quality prior fixed in 20
# quantile buckets, every user field moved within-user GAUC by -0.0004..+0.0001
# (user_active_degree alone gives 9 x 20 = 180 cells over 1.1M rows, so this is
# a real null, not sparsity). The join above still runs so the columns exist;
# to test them again, append USER_FEATURE_FIELDS to
# pipeline.train.EXTRA_CATEGORICAL_FIELDS (train.py's binding — see the
# from-import trap in the skill store).
# EXTRA_CATEGORICAL_FIELDS.extend(USER_FEATURE_FIELDS)


@lru_cache(maxsize=1)
def load_user_features() -> pd.DataFrame | None:
    """Static per-user attributes. Cached: read once per process."""
    from pipeline.data.loader import load_config

    cfg = load_config()
    path = Path(cfg["dataset"]["raw_dir"]) / cfg["dataset"]["features"]["user_features"]
    if not path.exists():
        return None
    frame = pd.read_csv(path, usecols=["user_id"] + USER_FEATURE_FIELDS)
    for col in USER_FEATURE_FIELDS:
        frame[col] = frame[col].astype(str)
    return frame


def build_features(
    df: pd.DataFrame,
    video_features_basic: pd.DataFrame | None = None,
    video_features_statistic: pd.DataFrame | None = None,
    allow_leaky_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Joins interaction rows with video-side features from KuaiRand's two
    separate video-feature files, each with different leakage properties, then
    sorts by user_id so each user's rows form one contiguous group (what a
    user-grouped pairwise/listwise loss needs).

    Args:
        df: interaction log rows (train/val slice).
        video_features_basic: `video_features_basic_pure.csv` — static
            per-video attributes (`author_id`, `music_id`, `upload_type`,
            ...) fixed at upload time. Not leaky, merged as-is.
        video_features_statistic: `video_features_statistic_pure.csv` —
            month-long running aggregates spanning train/val/test. Leaky by
            construction; passed through `leakage_guard.drop_leaky_columns`
            before merging.
        allow_leaky_columns: explicit opt-in for otherwise-dropped leaky
            columns — see leakage_guard.py. Empty by default.
    """
    out = df.copy()

    # Auto-load the static video-side table when the caller didn't pass one, so
    # every path builds IDENTICAL features. Training passed it explicitly while
    # pipeline/submit.py and the referee probe did not, which silently turned
    # author_id into UNK at scoring time — a train/serve skew that degrades the
    # submission without failing any check. One code path removes that class of
    # bug entirely. Pass an empty DataFrame to opt out.
    if video_features_basic is None:
        video_features_basic = load_video_basic_features()

    if video_features_basic is not None and not video_features_basic.empty:
        out = out.merge(video_features_basic, on="video_id", how="left")

    # Always merged so the columns EXIST for every caller; which of them are
    # actually encoded is controlled by EXTRA_CATEGORICAL_FIELDS below, so an
    # A/B test toggles the registry rather than the join.
    if any(f in EXTRA_CATEGORICAL_FIELDS for f in USER_FEATURE_FIELDS):
        user_features = load_user_features()
        if user_features is not None and not user_features.empty:
            out = out.merge(user_features, on="user_id", how="left")

    if video_features_statistic is not None:
        vf_clean, dropped = drop_leaky_columns(video_features_statistic, allow_columns=allow_leaky_columns)
        out = out.merge(vf_clean, on="video_id", how="left")
        if dropped:
            # Set AFTER merge, not before: pandas' DataFrame.attrs does not
            # reliably survive .merge() (it only propagates when all inputs
            # share identical attrs, which vf_clean's empty attrs breaks).
            # Left visible rather than silently swallowed — the orchestrator's
            # logger should capture this as part of the iteration's diff summary.
            out.attrs["dropped_leaky_columns"] = dropped

    # Session position: how many impressions this user has already been shown
    # earlier the same day. Causal by construction — cumcount over a
    # time-ordered group counts only PRECEDING rows, uses no labels, and is
    # known at serving time (a live recommender knows what it has already
    # shown this session). Measured on real validation data: long_view rate
    # falls 0.337 -> 0.195 from the first impression to the twelfth, and the
    # feature alone has within-user GAUC 0.5148. Adding it to the FM moved
    # valid primary 0.6017 -> 0.6024. It is one of the few signals that varies
    # WITHIN a user, which is the only kind that can change a user's ranking.
    if {"user_id", "date", "time_ms"}.issubset(out.columns):
        ordered = out.sort_values(["user_id", "date", "time_ms"], kind="stable")
        pos = ordered.groupby(["user_id", "date"]).cumcount().clip(0, 9)
        out["pos_bucket"] = pos.reindex(out.index)
    elif "pos_bucket" not in out.columns:
        out["pos_bucket"] = 0

    # Primary content tag. KuaiRand's `tag` is multi-valued ("39,68"), and the
    # model encodes single categorical values, so the FIRST tag is used as the
    # video's primary topic. Videos have a median of 10 per tag across 110 tags,
    # which is a real pooling level — unlike author/music, which are ~1 video
    # each (see load_video_basic_features). Missing tags become their own
    # "NONE" category rather than being dropped: 96 videos have no tag, and
    # "untagged" is itself a signal the embedding can learn.
    if "tag" in out.columns:
        out["tag1"] = out["tag"].astype(str).str.split(",").str[0].replace("nan", "NONE")
    elif "tag1" not in out.columns:
        out["tag1"] = "NONE"

    for col in NUMERIC_SIGNAL_COLUMNS:
        if col in out.columns:
            out[col] = out[col].fillna(0)

    # Sort interactions by user_id so user interactions form contiguous groups
    # required for user-grouped pairwise BPR loss and ranking evaluation.
    if "user_id" in out.columns:
        out = out.sort_values(by="user_id", kind="stable").reset_index(drop=True)

    return out


def auxiliary_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Returns the non-click feedback signals available for multi-task
    auxiliary transfer (Play 3 / ESMM-PLE style heads). Missing columns are
    tolerated so this stays safe to call before all signals are confirmed
    present in a given log file.
    """
    present = [c for c in AUXILIARY_LABEL_COLUMNS if c in df.columns]
    return df[present].fillna(0)
