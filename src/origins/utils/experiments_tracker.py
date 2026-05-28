import pandas as pd
import os
import enum
from typing import Dict, Any
from omegaconf import OmegaConf
from origins.logging.print import AnsiColors, log_with_color

from accelerate.logging import get_logger
logger = get_logger(__name__)

class ExperimentStatus(enum.Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class ExperimentsTracker:
    def __init__(self, cfg):
        self.tracker_sections = cfg.experiments_tracker.tracker_sections
        if self.tracker_sections is None:
            self.tracker_sections = ['model', 'seed', 'input', 'train', 'infer', 'grading']
    
        self.cfg = cfg
        self.min_epochs = cfg.experiments_tracker.min_epochs
        self.flat_config = self._flatten_config(cfg)
        self.columns = list(self.flat_config.keys()) + ['status', 'error_message', 'status_epoch']
        self.csv_path = cfg.experiments_tracker.filepath
        self.df = self._load_or_create_csv()

    def _flatten_config(self, cfg):
        """Flatten the configuration into a single-level dictionary."""
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        flat_config = {}
        for section in self.tracker_sections:
            if section in config_dict:
                if isinstance(config_dict[section], dict):
                    flat_config.update(self._flatten_dict({section: config_dict[section]}))
                else:
                    flat_config[section] = config_dict[section]
        # Add top-level keys if present
        for k in ['seed']:
            if k in config_dict:
                flat_config[k] = config_dict[k]
        return flat_config

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten a nested dictionary into a single-level dictionary with keys as concatenated paths."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _load_or_create_csv(self):
        """Load the CSV file or create it if it does not exist."""
        if os.path.exists(self.csv_path):
            try:
                df = pd.read_csv(self.csv_path)
                for col in self.columns:
                    if col not in df.columns:
                        df[col] = None
                return df
            except pd.errors.EmptyDataError:
                pass
            
        df = pd.DataFrame(columns=self.columns)
        df.to_csv(self.csv_path, index=False)
    
        return df

    def experiment_exists(self):
        """Check if an experiment with the current configuration exists.
        returns:
            exists: bool
            status: ExperimentStatus
            should_skip: bool # If True, the experiment should be skipped (not executed)
            status_epoch: int # The number of epochs that the experiment has completed
        """
        df = self.df
        config_row = self.flat_config
        if df.empty:
            return False, None, False, 0
        mask = pd.Series([True] * len(df))
        for k, v in config_row.items():
            # Only compare if the column exists
            if k in df.columns:
                # Fill NaN with a value that will never match v
                col = df[k].fillna("__MISSING__")
                # Convert both sides to string for robust comparison
                mask &= (col.astype(str) == str(v))
            else:
                # If the column doesn't exist, no match is possible
                mask &= False
        matches = df[mask]
        if matches.empty:
            return False, None, False, 0
        row = matches.iloc[0]

        row_status = str(row.get("status")).strip()
        try:
            row_epoch = int(float(row.get("status_epoch", 0)))
        except (ValueError, TypeError):
            row_epoch = 0

        # Completed runs are always skipped
        if row_status == ExperimentStatus.COMPLETED.value:
            return True, row_status, True, row_epoch

        # Need to restart partial and in_progress runs due to preemption
        if (
            row_status in [ExperimentStatus.PARTIAL.value, ExperimentStatus.IN_PROGRESS.value]
            and row_epoch >= self.min_epochs
        ):
            return True, row_status, True, row_epoch

        return True, row_status, False, row_epoch # The experiment should not be skipped

    def update(
        self, 
        status: ExperimentStatus,
        status_epoch: int = 0,
        error_message: str = ""
    ):
        """Update the status of the experiment in the CSV file."""
        df = pd.read_csv(self.csv_path)
        config_row = self.flat_config

        # Ensure all config columns exist
        for k in config_row.keys():
            if k not in df.columns:
                df[k] = None

        # Build mask robustly
        if df.empty:
            mask = pd.Series([], dtype=bool)
        else:
            mask = pd.Series([True] * len(df))
            for k, v in config_row.items():
                if k in df.columns:
                    col = df[k].fillna("__MISSING__")
                    mask &= (col.astype(str) == str(v))
                else:
                    mask &= False

        if mask.any():
            idx = df[mask].index[0]
            df.at[idx, 'status'] = status.value
            df.at[idx, 'status_epoch'] = status_epoch
            df.at[idx, 'error_message'] = str(error_message)
        else:
            row = config_row.copy()
            row['status'] = status.value
            row['status_epoch'] = status_epoch
            row['error_message'] = str(error_message)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        df.to_csv(self.csv_path, index=False)
        self.df = df

    def get_flat_config(self):
        return self.flat_config

    def get_csv_path(self):
        return self.csv_path

    def get_min_epochs(self):
        return self.min_epochs
