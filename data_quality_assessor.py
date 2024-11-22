"""Data validation framework for DataFrame quality checks.

This module provides a flexible framework for performing various validation checks
on pandas DataFrames, including population-level, primary key-level, and row-level
validations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime
from logger import Logger

logger = Logger(initial_level="DEBUG", module_name="OpenAI Handler")


class CheckLevel(Enum):
    """Enumeration of validation check levels."""

    POPULATION = "population"
    PRIMARY_KEY = "primary_key"
    ROW = "row"


@dataclass
class ValidationResult:
    """Container for validation check results.

    Parameters
    ----------
    is_valid : bool
        Whether the validation passed without any issues.
    messages : List[str]
        List of validation messages and warnings.
    stats : Dict[str, Any]
        Dictionary containing validation statistics.
    """

    is_valid: bool
    messages: List[str]
    stats: Dict[str, Any]


class ValidationCheck(ABC):
    """Abstract base class for validation checks.

    Parameters
    ----------
    enabled : bool, optional
        Whether the check is enabled, by default True
    level : CheckLevel, optional
        Level at which to perform the check, by default CheckLevel.POPULATION
    max_output : int, optional
        Maximum number of validation messages to output, by default 10
    **kwargs : dict
        Additional configuration parameters

    Attributes
    ----------
    enabled : bool
        Whether the check is enabled
    level : CheckLevel
        Level at which to perform the check
    max_output : int
        Maximum number of validation messages to output
    config : dict
        Additional configuration parameters
    """

    def __init__(
        self,
        enabled: bool = True,
        level: CheckLevel = CheckLevel.POPULATION,
        max_output: int = 10,
        **kwargs,
    ):
        self.enabled = enabled
        self.level = level
        self.max_output = max_output
        self.config = kwargs

    @abstractmethod
    def check(
        self,
        df: pd.DataFrame,
        column: str = None,
        primary_key: str = None,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Perform validation check on DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate
        column : str, optional
            Column to validate, by default None
        primary_key : str, optional
            Primary key column name, by default None

        Returns
        -------
        Tuple[Dict[str, Any], List[str]]
            Tuple containing validation statistics and messages
        """
        pass


class PopulationCheck(ValidationCheck):
    """Base class for population-level checks that ignore granularity settings.

    Parameters
    ----------
    **kwargs : dict
        Additional configuration parameters passed to parent class
    """

    def __init__(self, **kwargs):
        super().__init__(level=CheckLevel.POPULATION, **kwargs)


class BaseStatsCheck(PopulationCheck):
    """Basic DataFrame statistics check.

    Performs basic DataFrame validation including row count, column count,
    and primary key checks if specified.
    """

    def check(
        self,
        df: pd.DataFrame,
        column: str = None,
        primary_key: str = None,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Perform basic DataFrame validation checks.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate
        column : str, optional
            Unused in this check, by default None
        primary_key : str, optional
            Primary key column name, by default None

        Returns
        -------
        Tuple[Dict[str, Any], List[str]]
            Tuple containing validation statistics and messages
        """
        stats = {"row_count": len(df), "column_count": len(df.columns)}
        messages = []

        if primary_key:
            # PK-specific checks
            null_pks = df[primary_key].isnull().sum()
            duplicate_pks = df[primary_key].duplicated().sum()
            unique_pks = df[primary_key].nunique()

            stats.update(
                {
                    "null_pk_count": null_pks,
                    "duplicate_pk_count": duplicate_pks,
                    "unique_pk_count": unique_pks,
                }
            )

            if null_pks > 0:
                messages.append(f"Found {null_pks} null values in primary key")
            if duplicate_pks > 0:
                messages.append(f"Found {duplicate_pks} duplicate primary keys")

        return stats, messages


class DateCheck(ValidationCheck):
    """DateTime column validation check.

    Performs various checks on datetime columns including null checks,
    date range validation, and data freshness checks.
    """

    def check(
        self,
        df: pd.DataFrame,
        column: str,
        primary_key: str = None,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Perform datetime validation checks.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate
        column : str
            DateTime column to validate
        primary_key : str, optional
            Primary key column name, by default None

        Returns
        -------
        Tuple[Dict[str, Any], List[str]]
            Tuple containing validation statistics and messages
        """
        stats = {}
        messages = []

        if not pd.api.types.is_datetime64_any_dtype(df[column].dtype):
            messages.append(f"Column {column} is not a datetime type")
            return stats, messages

        # Basic date stats
        valid_dates = df[column].dropna()
        stats.update(
            {
                "null_count": df[column].isnull().sum(),
                "min_date": valid_dates.min() if not valid_dates.empty else None,
                "max_date": valid_dates.max() if not valid_dates.empty else None,
                "date_range_days": (
                    (valid_dates.max() - valid_dates.min()).days
                    if not valid_dates.empty
                    else None
                ),
            }
        )

        if self.level in [CheckLevel.PRIMARY_KEY, CheckLevel.ROW] and primary_key:
            # Granular date checks
            problematic_records = df[df[column].isnull()]
            if not problematic_records.empty:
                for _, record in problematic_records.head(self.max_output).iterrows():
                    messages.append(f"Null date found for {primary_key}={record[primary_key]}")
        else:
            # Population level checks
            if df[column].isnull().any():
                messages.append(f"Found {stats['null_count']} null dates")

        # Check data freshness (always at population level)
        if stats["max_date"] and (
            pd.Timestamp.now() - stats["max_date"]
        ).days > self.config.get("max_days_old", 30):
            messages.append(
                f"Data is more than {self.config.get('max_days_old', 30)} days old"
            )

        return stats, messages


class StringCheck(ValidationCheck):
    """String column validation check.

    Performs various checks on string columns including null checks
    and empty string validation.
    """

    def check(
        self,
        df: pd.DataFrame,
        column: str,
        primary_key: str = None,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Perform string validation checks.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate
        column : str
            String column to validate
        primary_key : str, optional
            Primary key column name, by default None

        Returns
        -------
        Tuple[Dict[str, Any], List[str]]
            Tuple containing validation statistics and messages
        """
        stats = {
            "null_count": df[column].isnull().sum(),
            "empty_count": (df[column] == "").sum()
            if pd.api.types.is_string_dtype(df[column])
            else 0,
        }
        messages = []

        if self.level in [CheckLevel.PRIMARY_KEY, CheckLevel.ROW] and primary_key:
            problematic_records = df[
                df[column].isnull() | (df[column].astype(str).str.strip() == "")
            ]
            for _, record in problematic_records.head(self.max_output).iterrows():
                messages.append(f"Empty/null string found for {primary_key}={record[primary_key]}")
        else:
            if stats["null_count"] > 0:
                messages.append(f"Found {stats['null_count']} null strings")
            if stats["empty_count"] > 0:
                messages.append(f"Found {stats['empty_count']} empty strings")

        return stats, messages


class NumericCheck(ValidationCheck):
    """Numeric column validation check.

    Performs various checks on numeric columns including null checks
    and large value validation.
    """

    def check(
        self,
        df: pd.DataFrame,
        column: str,
        primary_key: str = None,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Perform numeric validation checks.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate
        column : str
            Numeric column to validate
        primary_key : str, optional
            Primary key column name, by default None

        Returns
        -------
        Tuple[Dict[str, Any], List[str]]
            Tuple containing validation statistics and messages
        """
        if not pd.api.types.is_numeric_dtype(df[column]):
            return {}, [f"Column {column} is not numeric"]

        stats = {"null_count": df[column].isnull().sum()}
        messages = []

        # Check for large values
        large_values = df[df[column] > 1e20]
        stats["large_value_count"] = len(large_values)

        if self.level in [CheckLevel.PRIMARY_KEY, CheckLevel.ROW] and primary_key:
            for _, record in large_values.head(self.max_output).iterrows():
                messages.append(
                    f"Large value ({record[column]:.2e}) found for {primary_key}={record[primary_key]}"
                )
        else:
            if stats["null_count"] > 0:
                messages.append(f"Found {stats['null_count']} null values")
            if stats["large_value_count"] > 0:
                messages.append(f"Found {stats['large_value_count']} values larger than 1e20")

        return stats, messages


class DataValidator:
    """Main data validation orchestrator.

    Coordinates the execution of various validation checks on a DataFrame.

    Parameters
    ----------
    config : Dict, optional
        Configuration dictionary for validation checks, by default None

    Attributes
    ----------
    config : Dict
        Configuration dictionary for validation checks
    plugins : Dict
        Dictionary of validation check plugins
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}

        # Initialize default plugins
        self.plugins = {
            "base": BaseStatsCheck(),
            "datetime": DateCheck(**self.config.get("datetime", {})),
            "string": StringCheck(**self.config.get("string", {})),
            "numeric": NumericCheck(**self.config.get("numeric", {})),
        }

    def verify_data(
        self,
        df: pd.DataFrame,
        primary_key: Optional[str] = None,
    ) -> ValidationResult:
        """Verify DataFrame using all configured validation checks.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate
        primary_key : Optional[str], optional
            Primary key column name, by default None

        Returns
        -------
        ValidationResult
            Object containing validation results, messages, and statistics
        """
        if df.empty:
            return ValidationResult(False, ["Warning: Dataset is empty"], {})

        if primary_key and primary_key not in df.columns:
            logger.error(f"Specified primary key '{primary_key}' not found in DataFrame")
            primary_key = None

        stats = {}
        messages = []

        # Always run base stats
        base_stats, base_messages = self.plugins["base"].check(df, primary_key=primary_key)
        stats.update(base_stats)
        messages.extend(base_messages)

        # Run type-specific checks
        for column in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[column].dtype):
                check = self.plugins["datetime"]
            elif pd.api.types.is_string_dtype(df[column].dtype):
                check = self.plugins["string"]
            elif pd.api.types.is_numeric_dtype(df[column].dtype):
                check = self.plugins["numeric"]
            else:
                continue

            if not check.enabled:
                continue

            if check.level != CheckLevel.POPULATION and not primary_key:
                logger.warning(
                    f"Check {check.__class__.__name__} requires primary key but none provided. "
                    "Running at population level."
                )

            col_stats, col_messages = check.check(df, column, primary_key)
            stats[f"{column}_stats"] = col_stats
            messages.extend(col_messages)

        return ValidationResult(len(messages) == 0, messages, stats)