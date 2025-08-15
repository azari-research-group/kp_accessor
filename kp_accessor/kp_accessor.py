
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, cast
import warnings

from sortedcontainers import SortedDict

from kp_accessor.data_retrieval import (
    _download_kp_values_textfile,
    _prep_kp_table,
    _update_sorted_dict_from_kp_table,
)


class KpAccessor:
    """
    A class that provides access to Kp values (at 3-hour discretization)
        from the GFZ Helmholtz Centre for Geosciences in O(1) time.


    Methods
    -------
    get_kv_covering_datetime
        Returns the Kp value and datetime for the given datetime.
    get_key_covering_date
        Returns the datetime for the given datetime.
    get_kp_from_datetime
        Returns the Kp value for the given datetime.
    __call__
        Returns the Kp value for the given datetime.
    """
    def __init__(self) -> None:
        self._sd: SortedDict = SortedDict()
        self._update_cache(force_run=False)

    def _update_cache(self, force_run: bool = True) -> None:
        _download_kp_values_textfile(force_run=force_run)
        kp_table = _prep_kp_table(force_run=force_run)
        _update_sorted_dict_from_kp_table(kp_table, self._sd)

    def get_kv_covering_datetime(self, dt: datetime, three_hour_discretization: bool = True) -> Tuple[datetime, float]:
        """
        Returns the Kp value (and corresponding datetime) for the datetime or prior datetime nearest the provided one
            that has a recorded Kp value.

        3-hour-discretizes to the nearest time at or before the provided datetime if three_hour_discretization is True.

        Parameters
        ----------
        dt: datetime
            The datetime to get the Kp value for, or to find the nearest datetime (before) it with a recorded Kp value.
        three_hour_discretization: bool
            Whether to use the 3-hour discretization of the Kp values.
        Returns
        -------
        Tuple[datetime, float]
            The datetime and Kp value for the nearest datetime at or before the provided one with a recorded Kp value.
        """

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        # Initial exact check
        if dt in self._sd:
            return (dt, self._sd[dt])

        # If cache is empty, update it
        if len(self._sd) == 0:
            self._update_cache()
            if len(self._sd) == 0:
                raise RuntimeError("Kp cache did not populate.")

        # Check if the requested datetime is within the range of available Kp values
        if dt < self._sd.keys()[0]:
            raise ValueError("Requested datetime is before the earliest Kp value.")
        if dt > self._sd.keys()[-1] + timedelta(hours=3):
            if dt > datetime.now(tz=timezone.utc):
                raise ValueError("Requested datetime is in the future.")
            self._update_cache()
            if dt > self._sd.keys()[-1] + timedelta(hours=3):
                raise ValueError("Requested Kp later than the latest available Kp value.")

        # Secondary exact check:
        if dt in self._sd:
            return (dt, self._sd[dt])

        if three_hour_discretization:
            # get the nearest multiple-of-three-hours UTC datetime from the current dt
            dt = dt.astimezone(timezone.utc)
            dt = dt.replace(minute=0, second=0, microsecond=0)
            dt = dt - timedelta(hours=dt.hour % 3)

            if dt in self._sd:
                return (dt, self._sd[dt])
            else:
                warnings.warn(f"Could not find a value for the left-nearest 3-hour discretization"
                              f"of {dt=}. Falling back to finding the left-nearest datetime.")

        # Bisection to find the closest previous datetime
        # (as kp values are for 3-hour intervals, and the lookup dts
        # are the starts of their interval)
        left_match_idx = self._sd.bisect_left(dt) - 1
        left_key_value_pair = cast(Tuple[datetime, float], self._sd.items()[left_match_idx])
        return left_key_value_pair

    def get_key_covering_date(self, dt: datetime) -> Optional[datetime]:
        """
        Returns the datetime at or before the provided one with a recorded Kp value.

        Parameters
        ----------
        dt: datetime
            The datetime to get the Kp value for.
        Returns
        -------
        Optional[datetime]
            The datetime at or before the provided one with a recorded Kp value.
        """
        try:
            return self.get_kv_covering_datetime(dt)[0]
        except ValueError:
            return None

    def get_kp_from_datetime(self, dt: datetime) -> float:
        """
        Returns the Kp value at or before the provided datetime.

        Parameters
        ----------
        dt: datetime
            The datetime to get the Kp value for.
        Returns
        -------
        float
            The Kp value at or before the provided datetime.
        """
        return self.get_kv_covering_datetime(dt)[1]

    def __call__(self, dt: datetime) -> float:
        return self.get_kp_from_datetime(dt)


if __name__ == "__main__":
    kp_accessor = KpAccessor()
    dt = datetime(2023, 10, 1, 12, 0, tzinfo=timezone.utc)
    try:
        kp_value = kp_accessor(dt)
        print(f"Kp value for {dt} is {kp_value}")
    except ValueError as e:
        print(f"Error: {e}")
    except RuntimeError as e:
        print(f"Runtime error: {e}")

    dt_2 = datetime(2023, 10, 1, 15, 0, tzinfo=timezone.utc)
    print(f"Kp value for {dt_2} is {kp_accessor(dt_2)}")