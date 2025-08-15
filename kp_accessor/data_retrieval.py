import pathlib

import urllib.request

import numpy as np
import pandas as pd
from typing_extensions import MutableMapping
from datetime import datetime, timezone


CACHE_PATH = pathlib.Path(__file__).parent / "data" / "kp_table.txt"
TABLE_PATH = pathlib.Path(__file__).parent / "data" / "kp_table.csv"
URL = "https://kp.gfz.de/kpdata?startdate=1900-01-01&enddate=2050-12-31&format=kp1"


def _download_kp_values_textfile(force_run=False) -> pathlib.Path:
    """
    Downloads the KP table from the GFZ Helmholtz Centre for Geosciences.
    If force_run is False and the file already exists, it skips the download.
    Parameters
    ----------
    force_run: bool
        If True, forces the download even if the file already exists. This
           overwrites the existing file.
    """

    if not force_run and CACHE_PATH.exists():
        print("KP table already exists, skipping download.")
        return CACHE_PATH

    CACHE_PATH.parent.mkdir(exist_ok=True)
    print(f"Downloading KP table from {URL} ...")
    try:
        urllib.request.urlretrieve(URL, CACHE_PATH)
        print("Download complete.")
        return CACHE_PATH
    except Exception as e:
        raise RuntimeError(f"Failed to download KP table from {URL}.") from e


def _prep_kp_table(force_run=False) -> pd.DataFrame:
    """
    Prepares a CSV/pandas table from the downloaded KP table.
    If force_run is False and the file already exists, it skips the preparation.
    Parameters
    ----------
    force_run: bool
        If True, forces the preparation even if the file already exists. This
           overwrites the existing file.
    Returns
    -------
    pd.DataFrame
        The prepared KP table.
    """

    if not force_run and TABLE_PATH.exists():
        ret_table = pd.read_csv(TABLE_PATH)
        return ret_table

    ret_table = pd.read_csv(
        CACHE_PATH, sep=r"\s+", comment="#", header=None,
        names=[
            "year",
            "month",
            "day",
            "days_since_epoch",
            "days_since_epoch_m",
            "BSR",
            "dB",
            "kp_0", "kp_3", "kp_6", "kp_9", "kp_12", "kp_15", "kp_18", "kp_21",
            "ap_0", "ap_3", "ap_6", "ap_9", "ap_12", "ap_15", "ap_18", "ap_21",
            "Ap",
            "SN",
            "F10.7obs",
            "F10.7adj",
            "D"
        ]
    )
    ret_table.to_csv(TABLE_PATH, index=False)
    print(f"Prepared KP table and saved to {TABLE_PATH}")
    return ret_table


def _update_sorted_dict_from_kp_table(kp_table: pd.DataFrame, time_to_kp_map: MutableMapping) -> MutableMapping:
    """
    Prepares a MutableMapping from datetime to kp values.
        Updates the provided MutableMapping time_to_kp_map if provided.

    Parameters
    ----------
    kp_table: DataFrame
        The KP table DataFrame.
    Returns
    -------
    MutableMapping
        A mutable mapping from datetime to kp values.
    """
    for index, row in kp_table.iterrows():
        for hour in [0, 3, 6, 9, 12, 15, 18, 21]:
            datetime_ = datetime(
                year=np.rint(row["year"]).astype(int),
                month=np.rint(row["month"]).astype(int),
                day=np.rint(row["day"]).astype(int),
                hour=hour,
                tzinfo=timezone.utc
            )
            time_to_kp_map[datetime_] = row[f"kp_{hour}"]
    return time_to_kp_map
