import random
from typing import Dict, Iterable, Tuple
from datetime import datetime, timedelta, timezone

import pytest
from sortedcontainers import SortedDict

from kp_accessor.kp_accessor import _KpAccessor as KpAccessor

# ---- Helpers -----------------------------------------------------------------

START = datetime(2023, 10, 1, 0, 0, tzinfo=timezone.utc)
END = datetime(2023, 10, 5, 21, 0, tzinfo=timezone.utc)  # inclusive
STEP = timedelta(hours=3)


def make_fake_sd(with_gap: bool = False) -> SortedDict:
    """
    Build a fake Kp dataset on a 3-hour grid. Optionally introduce a gap
    (remove one 3-hour key) to exercise the warning/fallback path.
    """
    sd = SortedDict()
    t = START
    idx = 0
    gap_key = START + timedelta(hours=3*7) if with_gap else None  # arbitrary gap
    while t <= END:
        if t != gap_key:
            # Use a deterministic float value tied to index for easy assertions
            sd[t] = float(idx % 10) + 0.1 * (idx % 3)
        t += STEP
        idx += 1
    return sd


def build_accessor(monkeypatch, with_gap: bool = False) -> KpAccessor:
    """
    Create a KpAccessor whose _update_cache populates a local SortedDict
    instead of downloading anything.
    """
    fake_sd = make_fake_sd(with_gap=with_gap)

    def fake_update_cache(self, force_run: bool = True) -> None:
        # Clear and repopulate the existing SortedDict
        self._sd.clear()
        self._sd.update(fake_sd)

    monkeypatch.setattr(KpAccessor, "_update_cache", fake_update_cache, raising=True)
    ka = KpAccessor()
    ka._update_cache()
    return ka


# ---- Tests -------------------------------------------------------------------

def test_exact_match_random(monkeypatch):
    ka = build_accessor(monkeypatch, with_gap=False)
    # Pick 50 random exact keys and ensure values match
    keys = list(ka._sd.keys())
    for _ in range(50):
        dt = random.choice(keys)
        dt_naive_or_aware = dt if random.random() < 0.5 else dt.replace(tzinfo=None)
        kdt, kv = ka.get_kv_covering_datetime(dt_naive_or_aware)
        assert kdt == dt
        assert kv == ka._sd[dt]
        # __call__ and get_kp_from_datetime agree
        assert ka(dt) == ka.get_kp_from_datetime(dt)


def test_in_between_random_discretizes_left(monkeypatch):
    ka = build_accessor(monkeypatch, with_gap=False)
    keys = list(ka._sd.keys())
    # Generate 100 random datetimes between adjacent keys and check left-nearest
    for _ in range(100):
        i = random.randrange(1, len(keys))  # ensure there is a left neighbor
        left = keys[i - 1]
        right = keys[i]
        # Pick a random offset strictly between (not equal to boundaries)
        delta_seconds = (right - left).total_seconds()
        offset = random.uniform(1, delta_seconds - 1)
        dt = left + timedelta(seconds=offset)

        kdt, kv = ka.get_kv_covering_datetime(dt, three_hour_discretization=True)
        # Should discretize to left boundary
        assert kdt == left
        assert kv == ka._sd[left]

        # Also verify behavior when three_hour_discretization=False (still left nearest)
        kdt2, kv2 = ka.get_kv_covering_datetime(dt, three_hour_discretization=False)
        assert kdt2 == left
        assert kv2 == ka._sd[left]


def test_naive_datetime_treated_as_utc(monkeypatch):
    ka = build_accessor(monkeypatch, with_gap=False)
    # Choose a random key and strip tzinfo
    dt = random.choice(list(ka._sd.keys()))
    naive = dt.replace(tzinfo=None)
    kdt, kv = ka.get_kv_covering_datetime(naive)
    assert kdt == dt
    assert kv == ka._sd[dt]


def test_before_earliest_raises(monkeypatch):
    ka = build_accessor(monkeypatch, with_gap=False)
    before = START - timedelta(hours=1)
    with pytest.raises(ValueError, match="before the earliest"):
        ka.get_kv_covering_datetime(before)


def test_future_raises(monkeypatch):
    ka = build_accessor(monkeypatch, with_gap=False)
    future = datetime.now(tz=timezone.utc) + timedelta(days=1)
    with pytest.raises(ValueError, match="in the future"):
        ka.get_kv_covering_datetime(future)


def test_after_last_but_not_future_triggers_update_and_bounds(monkeypatch):
    """
    If dt is after the last+3h but not in the future, accessor will try to update,
    then (with our fixed data) still be after last+3h, raising the 'later than latest' error.
    """
    ka = build_accessor(monkeypatch, with_gap=False)
    dt = ka._sd.keys()[-1] + timedelta(hours=4)  # just beyond last+3h
    # Ensure this dt is not in the future relative to now (use 2023 dataset; today is later)
    assert dt < datetime.now(tz=timezone.utc)
    with pytest.raises(ValueError, match="Requested Kp later than the latest available Kp value."):
        ka.get_kv_covering_datetime(dt)


def test_gap_emits_warning_and_falls_back_left(monkeypatch):
    ka = build_accessor(monkeypatch, with_gap=True)

    # Find the left key before a 6h gap (i.e., where a single 3h key is missing)
    keys = list(ka._sd.keys())
    gap_left = None
    for i in range(1, len(keys)):
        if keys[i] - keys[i - 1] > STEP:
            gap_left = keys[i - 1]
            break
    assert gap_left is not None, "Test setup failed to create a gap"

    # Pick a dt inside the *missing* 3h window [gap_left+3h, gap_left+6h)
    target_dt = gap_left + timedelta(hours=4)  # inside the missing slot

    with pytest.warns(UserWarning) as wrecs:
        kdt, kv = ka.get_kv_covering_datetime(target_dt, three_hour_discretization=True)

    # Should have warned about missing discretized value
    assert any(
        "Could not find a value for the left-nearest 3-hour discretization" in str(w.message)
        for w in wrecs
    ), "Expected discretization warning not emitted"

    # Should fall back to the left-nearest available key
    assert kdt == gap_left
    assert kv == ka._sd[gap_left]


def test_get_key_covering_date_none_on_earlier(monkeypatch):
    ka = build_accessor(monkeypatch, with_gap=False)
    before = START - timedelta(days=2)
    assert ka.get_key_covering_date(before) is None


def test_call_matches_get_kp_from_datetime_random(monkeypatch):
    ka = build_accessor(monkeypatch, with_gap=False)
    keys = list(ka._sd.keys())

    for _ in range(30):
        base = random.choice(keys)
        # random offset within [0, 3h)
        offset = timedelta(seconds=random.randint(0, int(STEP.total_seconds() - 1)))
        dt = base + offset
        assert ka(dt) == ka.get_kp_from_datetime(dt)



UTC = timezone.utc


def floor_to_3h_utc(dt: datetime) -> datetime:
    """
    Convert to UTC, zero out minute/second/microsecond, and floor to the
    nearest 3-hour boundary at or before dt.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    dt_utc = dt.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
    return dt_utc - timedelta(hours=dt_utc.hour % 3)


def tz(h: int, m: int) -> timezone:
    """Fixed-offset tz helper for odd offsets like +01:22 or -07:28."""
    sign = 1 if (h, m) >= (0, 0) else -1  # not used; we pass signed values
    return timezone(timedelta(hours=h, minutes=m))


# ---- The 10 provided samples (aware datetimes with odd offsets) --------------

SAMPLES: Iterable[Tuple[datetime, float]] = [
    (datetime(2018, 9, 14, 14, 24, 3, tzinfo=tz(+1, +22)), 2.0),
    (datetime(1991, 7, 1, 19, 0, 2, tzinfo=tz(+0, +53)), 3.0),
    (datetime(1933, 8, 19, 6, 41, 43, tzinfo=tz(+11, +9)), 4.333),
    (datetime(1991, 6, 5, 9, 21, 42, tzinfo=tz(-7, -28)), 8.667),
    (datetime(1943, 6, 1, 1, 5, 24, tzinfo=UTC), 1.667),
    (datetime(1994, 2, 23, 1, 32, 27, tzinfo=tz(-5, -7)), 2.667),
    (datetime(1946, 3, 12, 6, 19, 47, tzinfo=tz(-1, 0)), 1.667),
    (datetime(1952, 4, 4, 7, 9, 12, tzinfo=tz(+9, +19)), 6.333),
    (datetime(1942, 1, 25, 14, 46, 26, tzinfo=tz(-7, -24)), 1.667),
    (datetime(1997, 4, 22, 4, 23, 21, tzinfo=tz(-4, -31)), 1.667),
]

@pytest.fixture
def accessor_with_samples(monkeypatch) -> KpAccessor:
    """
    Build a KpAccessor where _update_cache is monkeypatched to populate
    the cache with the discretized UTC keys for SAMPLES and their Kp values.
    """
    # Compute discretized keys and map to expected values
    kvs: Dict[datetime, float] = {}
    for dt, val in SAMPLES:
        key = floor_to_3h_utc(dt)
        kvs[key] = val

    # Create a small buffer around min/max so "range" checks are safe
    if kvs:
        mn = min(kvs.keys())
        mx = max(kvs.keys())
        # Add one earlier and one later key with arbitrary values
        kvs[mn - timedelta(hours=3)] = 0.0
        kvs[mx + timedelta(hours=3)] = 0.0

    sd = SortedDict(kvs)

    def fake_update_cache(self, force_run: bool = True) -> None:
        self._sd.clear()
        self._sd.update(sd)

    monkeypatch.setattr(KpAccessor, "_update_cache", fake_update_cache, raising=True)
    return KpAccessor()


@pytest.mark.parametrize("dt,expected", SAMPLES)
def test_samples_discretize_and_match_value(accessor_with_samples: KpAccessor, dt: datetime, expected: float):
    ka = accessor_with_samples

    # get_kv_covering_datetime returns the discretized UTC key and value
    key, value = ka.get_kv_covering_datetime(dt, three_hour_discretization=True)
    assert key == floor_to_3h_utc(dt)
    assert value == pytest.approx(expected, rel=0, abs=1e-9)

    # get_kp_from_datetime matches
    assert ka.get_kp_from_datetime(dt) == pytest.approx(expected, rel=0, abs=1e-9)

    # __call__ matches
    assert ka(dt) == pytest.approx(expected, rel=0, abs=1e-9)


def test_all_keys_present_in_cache(accessor_with_samples: KpAccessor):
    ka = accessor_with_samples
    for dt, _ in SAMPLES:
        ka.get_kv_covering_datetime(dt, three_hour_discretization=True) 
        assert floor_to_3h_utc(dt) in ka._sd
