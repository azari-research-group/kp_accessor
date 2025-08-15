from kp_accessor import kp_accessor
from datetime import datetime, timedelta, timezone, tzinfo

# Generate random datetimes between 1930 and 2024
# choose random timezones, too
import random
import pytz


initial_year = 1933

from datetime import datetime, timezone

# Collect results first
results = []
for yoffset in range(88):
    year_average = 0
    for m in range(1, 13):
        m_average = 0
        for d in range(1, 29):  # Use 28 to avoid month-end issues
            d_average = 0
            for h in [0, 3, 6, 9, 12, 15, 18, 21]:
                cur_datetime = datetime(
                    year=initial_year + yoffset,
                    month=m,
                    day=d,  # not using 'd' here, so this is really month-level
                    hour=h,
                    tzinfo=timezone.utc
                )
                d_value = kp_accessor(cur_datetime)
                d_average += d_value / 8.0
            m_average += d_average / 28.0
        year_average += m_average / 12.0
    results.append((initial_year + yoffset, year_average))

# Scale for ASCII bars
max_value = max(avg for _, avg in results)
scale = 50 / max_value  # max bar length = 50 chars

# Print ASCII chart
print("\nYearly Average Kp Index (Horizontal ASCII Chart)\n")
for year, avg in results:
    bar = "#" * int(avg * scale)
    print(f"{year}: {bar} {avg:.2f}")


print("kp on August 13, 2025 at 12:00 UTC:")
dt = datetime(2025, 8, 13, 12, 0, tzinfo=timezone.utc)
kp_value = kp_accessor(dt)
print(f"Kp value for {dt} is {kp_value}")


# for _ in range(10):
#     dt = datetime(
#         year=random.randint(1930, 2024),
#         month=random.randint(1, 12),
#         day=random.randint(1, 28),
#         hour=random.randint(0, 23),
#         minute=random.randint(0, 59),
#         second=random.randint(0, 59),
#         tzinfo=pytz.timezone(random.choice(pytz.all_timezones))
#     )
#     try:
#         kp_value = kp_accessor(dt)
#         print(f"Kp value for {dt} is {kp_value}")
#     except ValueError as e:
#         print(f"Error: {e}")
#     except RuntimeError as e:
#         print(f"Runtime error: {e}")
