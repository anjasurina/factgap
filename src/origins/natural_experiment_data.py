"""
Dataset layer for the natural experiment.

To add a new dataset:
1. Add a Config dataclass with __post_init__ validation.
2. Add get_<name>_datapoint() and get_<name>_dataset(df, config, ...) following
   the existing pattern. get_<name>_dataset must return tuple[list[Datapoint], list[str]].
3. Add the dataset file path to src/origins/configs/naturalistic_datasets.yaml.
4. Register the new key in _SupportedDataset and add a dispatch branch in prepare_data().
"""
import os
import random
from typing import Optional, Literal
from dataclasses import dataclass
import yaml

import pandas as pd
from tqdm import tqdm

from .utils.reusable_classes import *
from .utils.general_utilities import print_c


_SupportedDataset = Literal["nba_scores",
                            "market_data", "lottery_data", "billboard_100"]

_DEFAULT_DATASET_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)
                    ), "configs", "naturalistic_datasets.yaml"
)

################################################################################
# DATASET CONFIGS
################################################################################


@dataclass
class NBAConfig:
    data_type: str = "score"

    def __post_init__(self):
        valid = ["score"]
        if self.data_type not in valid:
            raise ValueError(
                f"Invalid data_type for nba_scores: {self.data_type!r}. Must be one of {valid}."
            )


@dataclass
class MarketConfig:
    data_type: str = "close"
    ticker: str = "s&p"

    def __post_init__(self):
        valid_data_types = ["close", "open", "high", "low"]
        if self.data_type not in valid_data_types:
            raise ValueError(
                f"Invalid data_type for market_data: {self.data_type!r}. Must be one of {valid_data_types}."
            )


@dataclass
class LotteryConfig:
    data_type: str = "winning_numbers"

    def __post_init__(self):
        valid_data_types = ["winning_numbers"]
        if self.data_type not in valid_data_types:
            raise ValueError(
                f"Invalid data_type for lottery_data: {self.data_type!r}. Must be one of {valid_data_types}."
            )


@dataclass
class BillboardConfig:
    data_type: str = "song"
    max_rank: int = 10
    jump_k: int = 1
    noise_type: str = "random_contemporary"

    def __post_init__(self):
        valid_data_types = ["song", "artist"]
        if self.data_type not in valid_data_types:
            raise ValueError(
                f"Invalid data_type for billboard_100: {self.data_type!r}. Must be one of {valid_data_types}."
            )
        valid_noise_types = [
            "random_contemporary", "previous_rank",
            "previous_random", "next_random", "next_rank",
        ]
        if self.noise_type not in valid_noise_types:
            raise ValueError(
                f"Invalid noise_type for billboard_100: {self.noise_type!r}. Must be one of {valid_noise_types}."
            )
        if self.max_rank < 2:
            raise ValueError(f"max_rank must be >= 2, got {self.max_rank}.")
        if self.jump_k < 1:
            raise ValueError(f"jump_k must be >= 1, got {self.jump_k}.")


DatasetConfig = NBAConfig | MarketConfig | LotteryConfig | BillboardConfig


################################################################################
# DATAPOINT
################################################################################


@dataclass
class Datapoint:
    unique_id: str
    statement: str
    question: str
    answer: str
    noise: bool


################################################################################
# NBA
################################################################################


def get_nba_datapoint(df, data_type: str = 'score', noise: bool = False, seed: int = 42) -> Datapoint:
    """Extract a single NBA datapoint from a dataframe.
    Args:
        df: DataFrame with NBA game data.
        data_type (str): Type of datapoint.
        noise (bool): Whether to add noise.
        seed (int): Base random seed.
    Returns:
        Datapoint
    """
    if data_type != 'score':
        raise NotImplementedError("Only 'score' type is implemented.")

    game_id = df.iloc[0]['gameId']
    team_1 = df.iloc[0]['teamName']
    team_1_points = df.iloc[0]['PTS']
    team_2 = df.iloc[1]['teamName']
    team_2_points = df.iloc[1]['PTS']
    date = df.iloc[0]['gameDate']

    true_score = f"{team_1}: {team_1_points}, {team_2}: {team_2_points}"
    rng = random.Random(f"{seed}_{game_id}")

    if noise:
        team_1_points += rng.randint(1, 10) * rng.choice([-1, 1])
        team_2_points += rng.randint(1, 10) * rng.choice([-1, 1])

    return Datapoint(
        unique_id=str(game_id),
        statement=f"The game played on {date} between the {team_1} and the {team_2} ended with a score of {team_1_points} to {team_2_points}.",
        question=f"What was the final score of the game played on {date} between the {team_1} and the {team_2}?",
        answer=true_score,
        noise=noise,
    )


def get_nba_dataset(
    df,
    config: NBAConfig,
    num_samples: int = 1,
    noise: bool = False,
    min_date=None,
    max_date=None,
    game_ids: Optional[list] = None,
    seed: int = 42,
) -> tuple[list[Datapoint], list[str]]:
    """Sample NBA datapoints from a dataframe.
    Args:
        df: DataFrame with NBA game data.
        config (NBAConfig): Dataset configuration.
        num_samples (int): Number of samples.
        noise (bool): Whether to add noise.
        min_date, max_date: Date filters.
        game_ids: List of game IDs to sample from.
        seed (int): Random seed.
    Returns:
        tuple: (list of Datapoints, used game IDs)
    """
    if min_date:
        df = df[df['gameDate'] >= min_date].copy()
    if max_date:
        df = df[df['gameDate'] <= max_date].copy()

    unique_game_ids = df['gameId'].unique() if game_ids is None else game_ids
    rng = random.Random(seed)
    sampled_ids = sorted(rng.sample(list(unique_game_ids),
                         min(num_samples * 2, len(unique_game_ids))))

    data, used_ids = [], []
    for game_id in sampled_ids:
        try:
            dp = get_nba_datapoint(
                df[df['gameId'] == game_id], data_type=config.data_type, noise=noise, seed=seed)
            data.append(dp)
            used_ids.append(dp.unique_id)
            if len(data) == num_samples:
                break
        except Exception:
            pass

    return data, used_ids


################################################################################
# MARKET
################################################################################


def get_market_datapoint(
    df, data_type: str = 'close', noise: bool = False, seed: int = 42
) -> Datapoint:
    """Extract a single market datapoint from a dataframe.
    Args:
        df: DataFrame with market data.
        data_type (str): Type of datapoint.
        noise (bool): Whether to add noise (random percentage perturbation).
        seed (int): Base random seed.
    Returns:
        Datapoint
    """
    if data_type not in ['close', 'open', 'high', 'low']:
        raise NotImplementedError(
            "Only 'close', 'open', 'high', 'low' types are implemented.")

    date_id = str(df.iloc[0]['date'])
    value = float(df.iloc[0][data_type].strip().replace(',', ''))
    value_original = value
    ticker = df.iloc[0]['ticker']
    rng = random.Random(f"{seed}_{date_id}")

    if noise:
        value += value * (rng.uniform(0.25, 2) * rng.choice([-1, 1])) / 100

    labels = {
        'close': 'closing price',
        'open':  'opening price',
        'high':  'highest price',
        'low':   'lowest price',
    }
    label = labels[data_type]

    return Datapoint(
        unique_id=date_id,
        statement=f"The {label} of {ticker} on {date_id} was {value: .2f}.",
        question=f"What was the {label} of {ticker} on {date_id}?",
        answer=f"{ticker} {label} on {date_id}: {value_original: .2f}",
        noise=noise,
    )


def get_market_dataset(
    df,
    config: MarketConfig,
    num_samples: int = 1,
    noise: bool = False,
    min_date=None,
    max_date=None,
    dates: Optional[list] = None,
    seed: int = 42,
) -> tuple[list[Datapoint], list[str]]:
    """Sample market datapoints from a dataframe.
    Args:
        df: DataFrame with market data.
        config (MarketConfig): Dataset configuration.
        num_samples (int): Number of samples.
        noise (bool): Whether to add noise.
        min_date, max_date: Date filters.
        dates: List of dates to sample from.
        seed (int): Random seed.
    Returns:
        tuple: (list of Datapoints, used dates)
    """
    filtered = df[df['ticker'] == config.ticker].copy()
    if min_date:
        filtered = filtered[filtered['date'] >= min_date]
    if max_date:
        filtered = filtered[filtered['date'] <= max_date]

    unique_dates = filtered['date'].unique() if dates is None else dates
    rng = random.Random(seed)
    sampled_dates = sorted(rng.sample(
        list(unique_dates), min(num_samples * 2, len(unique_dates))))

    data, used_dates = [], []
    for date in sampled_dates:
        try:
            dp = get_market_datapoint(
                filtered[filtered['date'] == date], data_type=config.data_type, noise=noise, seed=seed)
            data.append(dp)
            used_dates.append(dp.unique_id)
            if len(data) == num_samples:
                break
        except Exception as e:
            print(f"Error processing date {date}: {e}")

    return data, used_dates


################################################################################
# LOTTERY
################################################################################


def get_lottery_datapoint(
    df, data_type: str = 'winning_numbers', noise: bool = False, seed: int = 42
) -> Datapoint:
    """Extract a single lottery datapoint from a dataframe.
    Args:
        df: DataFrame with lottery data.
        data_type (str): Type of datapoint.
        noise (bool): Whether to add noise (random number perturbation).
        seed (int): Base random seed.
    Returns:
        Datapoint
    """
    if data_type != 'winning_numbers':
        raise NotImplementedError(
            "Only 'winning_numbers' type is implemented.")

    date_id = str(df.iloc[0]['date'])
    mega_ball = df.iloc[0]['mega_ball']
    values = [int(v) for v in df.iloc[0][data_type].strip().split()]
    value_original = " ".join(f"{n:02d}" for n in values)
    rng = random.Random(f"{seed}_{date_id}")

    if noise:
        for idx in rng.sample(range(len(values)), k=2):
            values[idx] = min(
                max(values[idx] + rng.randint(1, 20) * rng.choice([-1, 1]), 1), 75)

    values_str = " ".join(f"{n:02d}" for n in values)

    return Datapoint(
        unique_id=date_id,
        statement=f"The winning numbers for the Mega Millions lottery on {date_id} were {values_str}, with mega ball {mega_ball}.",
        question=f"What were the winning numbers for the Mega Millions lottery on {date_id}? Include the mega ball.",
        answer=f"The winning numbers for the Mega Millions lottery on {date_id}: {value_original}, with mega ball {mega_ball}.",
        noise=noise,
    )


def get_lottery_dataset(
    df,
    config: LotteryConfig,
    num_samples: int = 1,
    noise: bool = False,
    min_date=None,
    max_date=None,
    dates: Optional[list] = None,
    seed: int = 42,
) -> tuple[list[Datapoint], list[str]]:
    """Sample lottery datapoints from a dataframe.
    Args:
        df: DataFrame with lottery data.
        config (LotteryConfig): Dataset configuration.
        num_samples (int): Number of samples.
        noise (bool): Whether to add noise.
        min_date, max_date: Date filters.
        dates: List of dates to sample from.
        seed (int): Random seed.
    Returns:
        tuple: (list of Datapoints, used dates)
    """
    filtered = df.copy()
    if min_date:
        filtered = filtered[filtered['date'] >= min_date]
    if max_date:
        filtered = filtered[filtered['date'] <= max_date]

    unique_dates = filtered['date'].unique() if dates is None else dates
    rng = random.Random(seed)
    sampled_dates = sorted(rng.sample(
        list(unique_dates), min(num_samples * 2, len(unique_dates))))

    data, used_dates = [], []
    for date in sampled_dates:
        try:
            dp = get_lottery_datapoint(
                filtered[filtered['date'] == date], data_type=config.data_type, noise=noise, seed=seed)
            data.append(dp)
            used_dates.append(dp.unique_id)
            if len(data) == num_samples:
                break
        except Exception as e:
            print(f"Error processing date {date}: {e}")

    return data, used_dates


################################################################################
# BILLBOARD
################################################################################


def get_billboard_datapoint(
    df, full_df=None, data_type: str = 'song', noise: bool = False,
    noise_type: str = 'random_contemporary', seed: int = 42,
    max_rank: int = 10, jump_k: int = 1, target_rank: Optional[int] = None
) -> Datapoint:
    """Extract a single Billboard Hot 100 datapoint from a dataframe.
    Args:
        df: DataFrame with Billboard Hot 100 data for one week.
        full_df: Full historical DataFrame (required for temporal noise types).
        data_type (str): Type of datapoint ('song' or 'artist').
        noise (bool): Whether to add noise.
        noise_type (str): Noise strategy ('random_contemporary', 'previous_rank',
            'previous_random', 'next_random', or 'next_rank').
        seed (int): Base random seed.
        max_rank (int): Maximum Billboard rank to sample from.
        jump_k (int): Number of distinct different values to jump over.
        target_rank (int): Specific rank to target instead of sampling randomly.
    Returns:
        Datapoint
    """
    if data_type not in ['song', 'artist']:
        raise NotImplementedError(
            "Only 'song' and 'artist' types are implemented.")

    date_id = str(df.iloc[0]['date'])
    rank = target_rank if target_rank is not None else random.Random(
        f"{seed}_{date_id}").randint(1, max_rank - 1)

    # Isolated deterministic generator for this seed + date + rank combination.
    # Prevents noise=True vs noise=False passes from desyncing when creating instances.
    rng = random.Random(f"{seed}_{date_id}_{rank}")

    value = df[df['rank'] == rank].iloc[0][data_type]
    value_original = value

    if noise:
        if noise_type == 'previous_rank' and full_df is not None:
            past_records = full_df[(full_df['rank'] == rank) & (
                full_df['date'] < date_id)].sort_values('date', ascending=False)
            seen_different, current_value = 0, value_original
            for _, row in past_records.iterrows():
                if row[data_type] != current_value:
                    seen_different += 1
                    current_value = row[data_type]
                    if seen_different == jump_k:
                        value = current_value
                        break

        elif noise_type == 'next_rank' and full_df is not None:
            future_records = full_df[(full_df['rank'] == rank) & (
                full_df['date'] > date_id)].sort_values('date', ascending=True)
            seen_different, current_value = 0, value_original
            for _, row in future_records.iterrows():
                if row[data_type] != current_value:
                    seen_different += 1
                    current_value = row[data_type]
                    if seen_different == jump_k:
                        value = current_value
                        break

        elif noise_type == 'previous_random' and full_df is not None:
            past_dates = sorted(
                full_df[full_df['date'] < date_id]['date'].unique(), reverse=True)
            if len(past_dates) >= jump_k:
                past_df = full_df[(full_df['date'] == past_dates[jump_k - 1]) & (
                    full_df['rank'] <= max_rank) & (full_df['rank'] != rank)]
                available = past_df[past_df[data_type] !=
                                    value_original][data_type].tolist()
                if available:
                    value = rng.choice(available)

        elif noise_type == 'next_random' and full_df is not None:
            future_dates = sorted(
                full_df[full_df['date'] > date_id]['date'].unique())
            if len(future_dates) >= jump_k:
                future_df = full_df[(full_df['date'] == future_dates[jump_k - 1]) & (
                    full_df['rank'] <= max_rank) & (full_df['rank'] != rank)]
                available = future_df[future_df[data_type]
                                      != value_original][data_type].tolist()
                if available:
                    value = rng.choice(available)

        if value == value_original:
            # Fallback: pick any other entry in the top max_rank for this week.
            candidates = df[(df['rank'] <= max_rank) & (
                df['rank'] != rank)][data_type].tolist()
            if candidates:
                value = rng.choice(candidates)

    if data_type == 'song':
        return Datapoint(
            unique_id=date_id,
            statement=f"The number {rank} ranked song on the Billboard Hot 100 for the week of {date_id} was '{value}.'",
            question=f"What was the number {rank} ranked song on the Billboard Hot 100 for the week of {date_id}?",
            answer=f"The number {rank} ranked song on the Billboard Hot 100 for the week of {date_id}: {value_original}.",
            noise=noise,
        )
    else:
        return Datapoint(
            unique_id=date_id,
            statement=f"The artist of the number {rank} ranked song on the Billboard Hot 100 for the week of {date_id} was '{value}.'",
            question=f"Who was the artist of the number {rank} ranked song on the Billboard Hot 100 for the week of {date_id}?",
            answer=f"The artist of the number {rank} ranked song on the Billboard Hot 100 for the week of {date_id}: {value_original}.",
            noise=noise,
        )


def get_billboard_dataset(
    df,
    config: BillboardConfig,
    num_samples: int = 1,
    noise: bool = False,
    min_date=None,
    max_date=None,
    dates: Optional[list] = None,
    seed: int = 42,
) -> tuple[list[Datapoint], list[str]]:
    """Sample Billboard Hot 100 datapoints from a dataframe.
    Args:
        df: DataFrame with Billboard Hot 100 data.
        config (BillboardConfig): Dataset configuration.
        num_samples (int): Number of samples.
        noise (bool): Whether to add noise.
        min_date, max_date: Date filters.
        dates: List of dates to sample from.
        seed (int): Random seed.
    Returns:
        tuple: (list of Datapoints, used date-rank IDs)
    """
    filtered = df.copy()
    if min_date:
        filtered = filtered[filtered['date'] >= min_date]
    if max_date:
        filtered = filtered[filtered['date'] <= max_date]

    unique_dates = filtered['date'].unique() if dates is None else dates
    all_combos = [(date, rank)
                  for date in unique_dates for rank in range(1, config.max_rank)]

    rng = random.Random(seed)
    sampled_combos = sorted(rng.sample(
        all_combos, min(num_samples * 2, len(all_combos))))

    data, used_ids = [], []
    for date, target_rank in sampled_combos:
        try:
            dp = get_billboard_datapoint(
                filtered[filtered['date'] == date], full_df=df,
                data_type=config.data_type, noise=noise,
                noise_type=config.noise_type, seed=seed,
                max_rank=config.max_rank, jump_k=config.jump_k,
                target_rank=target_rank,
            )
            data.append(dp)
            used_ids.append(f"{dp.unique_id}_{target_rank}")
            if len(data) == num_samples:
                break
        except Exception as e:
            print(f"Error processing date {date} with rank {target_rank}: {e}")

    return data, used_ids


################################################################################
# DATASET LOADING
################################################################################


def _lookup_dataset_path(
    dataset: str,
    config_path: str = _DEFAULT_DATASET_CONFIG_PATH,
) -> str:
    """Look up a dataset's file path from the datasets config YAML.
    Args:
        dataset (str): Dataset name.
        config_path (str): Path to datasets config YAML.
    Returns:
        str: Resolved absolute dataset path.
    """
    with open(config_path, "r") as f:
        dataset_paths = yaml.safe_load(f)
    if dataset not in dataset_paths:
        raise ValueError(
            f"Unsupported dataset: {dataset!r}. Available: {list(dataset_paths.keys())}"
        )
    raw_path = dataset_paths[dataset]
    if not os.path.isabs(raw_path):
        project_root = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        return os.path.join(project_root, raw_path)
    return raw_path


def get_year_range(start_year: int, end_year: int, skip_year_frequency: int) -> list[int]:
    """Return a list of years from start_year to end_year (inclusive) with the given step."""
    return list(range(start_year, end_year + 1, skip_year_frequency))


def get_datapoints_for_year(
    datapoints,
    num_data_points_per_year: int,
    start_year: int,
    end_year: int,
    skip_year_frequency: int,

) -> dict[int, list]:
    """Split datapoints by year.
    Args:
        datapoints: List of datapoints.
        num_data_points_per_year (int): Points per year.
        start_year, end_year, skip_year_frequency: Year range params.
    Returns:
        dict: Mapping year to datapoints.
    """

    years = get_year_range(
        start_year=start_year,
        end_year=end_year,
        skip_year_frequency=skip_year_frequency
    )
    return {
        year: datapoints[i * num_data_points_per_year: i *
                         num_data_points_per_year + num_data_points_per_year]
        for i, year in enumerate(years)
    }


def _build_date_windows(
    year: int,
    year_index: int,
    num_years: int,
    start_month: int,
    end_month: int,
    granularity: str,
) -> tuple[list[str], list[str]]:
    """Build (min_date, max_date) window pairs for a given year and granularity."""
    min_dates, max_dates = [], []

    if granularity == "year":
        min_dates.append(
            f"{year}-{start_month:02d}-01" if year_index == 0 else f"{year}-01-01")
        max_dates.append(f"{year}-{end_month:02d}-31" if year_index ==
                         num_years - 1 else f"{year}-12-31")

    elif granularity == "6month":
        d1 = f"{year}-01-01"
        d2 = f"{year}-06-30"
        d3 = f"{year}-12-31"
        if year_index == 0:
            if start_month < 6:
                min_dates.extend([f"{year}-{start_month:02d}-01", d2])
                max_dates.extend([d2, d3])
            else:
                min_dates.append(f"{year}-{start_month:02d}-01")
                max_dates.append(d3)
        elif year_index == num_years - 1:
            if end_month <= 6:
                min_dates.append(d1)
                max_dates.append(f"{year}-{end_month:02d}-31")
            else:
                min_dates.extend([d1, d2])
                max_dates.extend([d2, f"{year}-{end_month:02d}-31"])
        else:
            min_dates.extend([d1, d2])
            max_dates.extend([d2, d3])

    elif granularity == "3month":
        quarters = [
            (f"{year}-01-01", f"{year}-03-31"),
            (f"{year}-03-31", f"{year}-06-30"),
            (f"{year}-06-30", f"{year}-09-30"),
            (f"{year}-09-30", f"{year}-12-31"),
        ]
        for mn, mx in quarters:
            min_dates.append(mn)
            max_dates.append(mx)

    else:
        raise ValueError(f"Unsupported data_granularity: {granularity!r}")

    return min_dates, max_dates


def prepare_data(
    dataset: str,
    dataset_config: DatasetConfig,
    start_year: int,
    num_data_points_per_year: int,
    end_year: int,
    skip_year_frequency: int,
    start_month: int = 1,
    end_month: int = 12,
    data_granularity: str = "year",
    noise: bool = False,
    seed: int = 42,
    verbosity_level: int = 1,
    dataset_path_config: str = _DEFAULT_DATASET_CONFIG_PATH,
) -> tuple[dict[int, list[Datapoint]], list[str]]:
    """Prepare data for experiment by year and granularity.
    Args:
        dataset (str): Dataset name.
        dataset_config (DatasetConfig): Dataset-specific configuration.
        start_year, end_year, skip_year_frequency: Year params.
        start_month, end_month: Month params (applied to first/last year only).
        num_data_points_per_year (int): Points per year.
        data_granularity (str): Sampling granularity ("year", "6month", "3month").
        noise (bool): Whether to add noise.
        seed (int): Random seed.
        verbosity_level (int): Verbosity level.
        dataset_path_config (str): Path to datasets config YAML.
    Returns:
        tuple: (dict mapping year to list of Datapoints, list of unique IDs)
    """
    dataset_path = _lookup_dataset_path(
        dataset, config_path=dataset_path_config)
    try:
        df_data = pd.read_csv(dataset_path)
    except Exception as e:
        raise FileNotFoundError(
            f"Could not load dataset from {dataset_path}: {e}")
    print_c(str(df_data.head()), v=verbosity_level,
            vmin=2, color=ColorType.BLUE)

    _dataset_fn = {
        "nba_scores":    get_nba_dataset,
        "market_data":   get_market_dataset,
        "lottery_data":  get_lottery_dataset,
        "billboard_100": get_billboard_dataset,
    }
    if dataset not in _dataset_fn:
        raise ValueError(f"Unsupported dataset: {dataset!r}")
    get_fn = _dataset_fn[dataset]

    YEARS = get_year_range(start_year, end_year, skip_year_frequency)
    unique_ids: list[str] = []
    datasets_per_year: dict[int, list[Datapoint]] = {}

    for year_index, YEAR in tqdm(enumerate(YEARS)):
        min_dates, max_dates = _build_date_windows(
            YEAR, year_index, len(
                YEARS), start_month, end_month, data_granularity
        )
        num_per_window = num_data_points_per_year // len(min_dates)

        data_for_year: list[Datapoint] = []
        for min_date, max_date in zip(min_dates, max_dates):
            data_chunk, ids_chunk = get_fn(
                df_data, config=dataset_config,
                num_samples=num_per_window,
                noise=noise,
                min_date=min_date,
                max_date=max_date,
                seed=seed,
            )
            if data_chunk:
                data_for_year.extend(data_chunk)
                unique_ids.extend(ids_chunk)

        if data_for_year:
            datasets_per_year[YEAR] = data_for_year

    return datasets_per_year, unique_ids
