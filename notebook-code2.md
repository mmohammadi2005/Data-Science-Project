```python
import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
print("MKL_NUM_THREADS:", os.environ.get("MKL_NUM_THREADS"))
```

    OMP_NUM_THREADS: 2
    MKL_NUM_THREADS: 2
    

## مهندسی ویژگی ها

### مهندسی ویژگی های زمانی چرخه ای 


```python
# تنظیم محیط و بارگذاری داده
from pathlib import Path
import pandas as pd
import numpy as np

from IPython.display import display

# مسیر پروژه
PROJECT_ROOT = Path(r"D:\کارشناسی\تحلیل داده")

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures" / "phase2"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

NY_TZ = "America/New_York"

HOURLY_FILE = PROCESSED_DIR / "hourly_trip_counts.parquet"

if not HOURLY_FILE.exists():
    raise FileNotFoundError(
        f"Hourly dataset not found: {HOURLY_FILE}"
    )

# بارگذاری سری زمانی
hourly_df = pd.read_parquet(HOURLY_FILE)

print("Dataset loaded successfully.")
print("Shape:", hourly_df.shape)
display(hourly_df.head())

```

    Dataset loaded successfully.
    Shape: (2183, 1)
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trip_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2024-01-01 00:00:00-05:00</th>
      <td>6596</td>
    </tr>
    <tr>
      <th>2024-01-01 01:00:00-05:00</th>
      <td>7355</td>
    </tr>
    <tr>
      <th>2024-01-01 02:00:00-05:00</th>
      <td>6220</td>
    </tr>
    <tr>
      <th>2024-01-01 03:00:00-05:00</th>
      <td>4936</td>
    </tr>
    <tr>
      <th>2024-01-01 04:00:00-05:00</th>
      <td>3188</td>
    </tr>
  </tbody>
</table>
</div>



```python

# بررسی و اصلاح ساختار داده‌های سری زمانی

# اگر pickup_hour به‌صورت Index باشد، آن را به ستون تبدیل می‌کنیم
if "pickup_hour" not in hourly_df.columns:
    hourly_df = hourly_df.rename_axis("pickup_hour").reset_index()

# تبدیل ستون زمان به نوع datetime
hourly_df["pickup_hour"] = pd.to_datetime(
    hourly_df["pickup_hour"],
    errors="coerce"
)

# بررسی مقادیر نامعتبر زمان
assert hourly_df["pickup_hour"].notna().all(), "Invalid timestamps found."

# بررسی منطقه زمانی
assert hourly_df["pickup_hour"].dt.tz is not None, "Timezone information is missing."
assert str(hourly_df["pickup_hour"].dt.tz) == "America/New_York", (
    f"Unexpected timezone: {hourly_df['pickup_hour'].dt.tz}"
)

# بررسی ستون تعداد سفرها
hourly_df["trip_count"] = pd.to_numeric(
    hourly_df["trip_count"],
    errors="coerce"
)

assert hourly_df["trip_count"].notna().all(), "Missing trip counts found."
assert (hourly_df["trip_count"] >= 0).all(), "Negative trip counts found."

# مرتب‌سازی زمانی
hourly_df = hourly_df.sort_values("pickup_hour").reset_index(drop=True)

# بررسی یکتایی زمان‌ها
assert hourly_df["pickup_hour"].is_unique, "Duplicate hourly timestamps found."

# بررسی فاصله زمانی بین مشاهدات متوالی
time_differences = hourly_df["pickup_hour"].diff().dropna()

assert time_differences.eq(pd.Timedelta(hours=1)).all(), (
    "Unexpected gaps in hourly timestamps."
)

# نمایش نتایج
print("Data validation completed successfully.")
print("Shape:", hourly_df.shape)
print("Timezone:", hourly_df["pickup_hour"].dt.tz)
print("Time range:")
print(hourly_df["pickup_hour"].min(), "to", hourly_df["pickup_hour"].max())

print("\nDescriptive statistics:")
display(hourly_df["trip_count"].describe())

```

    Data validation completed successfully.
    Shape: (2183, 2)
    Timezone: America/New_York
    Time range:
    2024-01-01 00:00:00-05:00 to 2024-03-31 23:00:00-04:00
    
    Descriptive statistics:
    


    count     2183.000000
    mean      4376.892350
    std       2534.155695
    min        147.000000
    25%       1877.000000
    50%       5021.000000
    75%       6248.000000
    max      11671.000000
    Name: trip_count, dtype: float64



```python
#ساخت ویژگی‌های تقویمی
# ساخت یک نسخه مستقل برای مهندسی ویژگی
temporal_features = hourly_df.copy()

ts = temporal_features["pickup_hour"]

# ویژگی‌های تقویمی
temporal_features["hour_of_day"] = ts.dt.hour

temporal_features["day_of_week"] = ts.dt.dayofweek

temporal_features["month"] = ts.dt.month

temporal_features["day_of_month"] = ts.dt.day

temporal_features["is_weekend"] = (
    ts.dt.dayofweek >= 5
).astype("int8")

temporal_features["is_weekday"] = (
    ts.dt.dayofweek < 5
).astype("int8")

# بررسی خروجی
calendar_columns = [
    "pickup_hour",
    "trip_count",
    "hour_of_day",
    "day_of_week",
    "month",
    "day_of_month",
    "is_weekend",
    "is_weekday"
]

display(
    temporal_features[calendar_columns].head(72)
)

print("\nCalendar feature summary:")
display(
    temporal_features[
        calendar_columns[2:]
    ].describe()
)

print("\nCalendar features created successfully.")

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pickup_hour</th>
      <th>trip_count</th>
      <th>hour_of_day</th>
      <th>day_of_week</th>
      <th>month</th>
      <th>day_of_month</th>
      <th>is_weekend</th>
      <th>is_weekday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-01-01 00:00:00-05:00</td>
      <td>6596</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-01-01 01:00:00-05:00</td>
      <td>7355</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-01-01 02:00:00-05:00</td>
      <td>6220</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-01-01 03:00:00-05:00</td>
      <td>4936</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-01-01 04:00:00-05:00</td>
      <td>3188</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>67</th>
      <td>2024-01-03 19:00:00-05:00</td>
      <td>5070</td>
      <td>19</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>68</th>
      <td>2024-01-03 20:00:00-05:00</td>
      <td>4388</td>
      <td>20</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>69</th>
      <td>2024-01-03 21:00:00-05:00</td>
      <td>3962</td>
      <td>21</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>70</th>
      <td>2024-01-03 22:00:00-05:00</td>
      <td>3242</td>
      <td>22</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>71</th>
      <td>2024-01-03 23:00:00-05:00</td>
      <td>2483</td>
      <td>23</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>72 rows × 8 columns</p>
</div>


    
    Calendar feature summary:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hour_of_day</th>
      <th>day_of_week</th>
      <th>month</th>
      <th>day_of_month</th>
      <th>is_weekend</th>
      <th>is_weekday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2183.000000</td>
      <td>2183.000000</td>
      <td>2183.000000</td>
      <td>2183.000000</td>
      <td>2183.000000</td>
      <td>2183.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>11.504352</td>
      <td>2.998626</td>
      <td>1.999542</td>
      <td>15.683921</td>
      <td>0.285387</td>
      <td>0.714613</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.922370</td>
      <td>1.999885</td>
      <td>0.825521</td>
      <td>8.779870</td>
      <td>0.451702</td>
      <td>0.451702</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>6.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>8.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>12.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>16.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>17.500000</td>
      <td>5.000000</td>
      <td>3.000000</td>
      <td>23.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>23.000000</td>
      <td>6.000000</td>
      <td>3.000000</td>
      <td>31.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>


    
    Calendar features created successfully.
    


```python
# ساخت ویژگی های چرخه ای 
# تبدیل تعداد سفرها به عدد صحیح
assert np.allclose(
    temporal_features["trip_count"],
    temporal_features["trip_count"].round()
), "Trip counts contain non-integer values."

temporal_features["trip_count"] = (
    temporal_features["trip_count"].astype("int64")
)

# ویژگی‌های چرخه‌ای ساعت شبانه‌روز
temporal_features["hour_sin"] = np.sin(
    2 * np.pi * temporal_features["hour_of_day"] / 24
)

temporal_features["hour_cos"] = np.cos(
    2 * np.pi * temporal_features["hour_of_day"] / 24
)

# ویژگی‌های چرخه‌ای روز هفته
temporal_features["day_sin"] = np.sin(
    2 * np.pi * temporal_features["day_of_week"] / 7
)

temporal_features["day_cos"] = np.cos(
    2 * np.pi * temporal_features["day_of_week"] / 7
)

# ویژگی‌های چرخه‌ای ماه
temporal_features["month_sin"] = np.sin(
    2 * np.pi * (temporal_features["month"] - 1) / 12
)

temporal_features["month_cos"] = np.cos(
    2 * np.pi * (temporal_features["month"] - 1) / 12
)

print("Cyclical features created successfully.")

display(
    temporal_features[
        [
            "pickup_hour",
            "hour_of_day",
            "hour_sin",
            "hour_cos",
            "day_of_week",
            "day_sin",
            "day_cos",
            "month",
            "month_sin",
            "month_cos"
        ]
    ].head(72)
)

```

    Cyclical features created successfully.
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pickup_hour</th>
      <th>hour_of_day</th>
      <th>hour_sin</th>
      <th>hour_cos</th>
      <th>day_of_week</th>
      <th>day_sin</th>
      <th>day_cos</th>
      <th>month</th>
      <th>month_sin</th>
      <th>month_cos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-01-01 00:00:00-05:00</td>
      <td>0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-01-01 01:00:00-05:00</td>
      <td>1</td>
      <td>0.258819</td>
      <td>0.965926</td>
      <td>0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-01-01 02:00:00-05:00</td>
      <td>2</td>
      <td>0.500000</td>
      <td>0.866025</td>
      <td>0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-01-01 03:00:00-05:00</td>
      <td>3</td>
      <td>0.707107</td>
      <td>0.707107</td>
      <td>0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-01-01 04:00:00-05:00</td>
      <td>4</td>
      <td>0.866025</td>
      <td>0.500000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>67</th>
      <td>2024-01-03 19:00:00-05:00</td>
      <td>19</td>
      <td>-0.965926</td>
      <td>0.258819</td>
      <td>2</td>
      <td>0.974928</td>
      <td>-0.222521</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>68</th>
      <td>2024-01-03 20:00:00-05:00</td>
      <td>20</td>
      <td>-0.866025</td>
      <td>0.500000</td>
      <td>2</td>
      <td>0.974928</td>
      <td>-0.222521</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>69</th>
      <td>2024-01-03 21:00:00-05:00</td>
      <td>21</td>
      <td>-0.707107</td>
      <td>0.707107</td>
      <td>2</td>
      <td>0.974928</td>
      <td>-0.222521</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>70</th>
      <td>2024-01-03 22:00:00-05:00</td>
      <td>22</td>
      <td>-0.500000</td>
      <td>0.866025</td>
      <td>2</td>
      <td>0.974928</td>
      <td>-0.222521</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>71</th>
      <td>2024-01-03 23:00:00-05:00</td>
      <td>23</td>
      <td>-0.258819</td>
      <td>0.965926</td>
      <td>2</td>
      <td>0.974928</td>
      <td>-0.222521</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>72 rows × 10 columns</p>
</div>



```python
# مقایسه ﻧﻤﺎﯾﺶ ﭼﺮﺧﻪ ﺍﯼ ﺳﯿﻨﻮﺱ ﻭ ﮐﺴﯿﻨﻮﺱ ﺑﺎ ﻧﻤﺎﯾﺶ ﺳﺎﺩﻩ ﯼ ﻋددی به صورت نمودار  
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 2, figsize=(14, 15))

# مقادیر یکتای موجود در جدول ویژگی‌ها
hour_data = temporal_features[
    ["hour_of_day", "hour_sin", "hour_cos"]
].drop_duplicates().sort_values("hour_of_day")

day_data = temporal_features[
    ["day_of_week", "day_sin", "day_cos"]
].drop_duplicates().sort_values("day_of_week")

month_data = temporal_features[
    ["month", "month_sin", "month_cos"]
].drop_duplicates().sort_values("month")


# 1. ساعت شبانه‌روز
axes[0, 0].plot(
    hour_data["hour_of_day"],
    hour_data["hour_of_day"],
    marker="o"
)
axes[0, 0].set_title("Hour - Simple Numeric Encoding")
axes[0, 0].set_xlabel("Hour")
axes[0, 0].set_ylabel("Numeric value")
axes[0, 0].grid(True)

axes[0, 1].plot(
    hour_data["hour_of_day"],
    hour_data["hour_sin"],
    marker="o",
    label="hour_sin"
)
axes[0, 1].plot(
    hour_data["hour_of_day"],
    hour_data["hour_cos"],
    marker="o",
    label="hour_cos"
)
axes[0, 1].set_title("Hour - Cyclical Encoding")
axes[0, 1].set_xlabel("Hour")
axes[0, 1].set_ylabel("Encoded value")
axes[0, 1].legend()
axes[0, 1].grid(True)


# 2. روز هفته
axes[1, 0].plot(
    day_data["day_of_week"],
    day_data["day_of_week"],
    marker="o"
)
axes[1, 0].set_title("Day of Week - Simple Numeric Encoding")
axes[1, 0].set_xlabel("Day of Week (0=Monday, 6=Sunday)")
axes[1, 0].set_ylabel("Numeric value")
axes[1, 0].set_xticks(range(7))
axes[1, 0].grid(True)

axes[1, 1].plot(
    day_data["day_of_week"],
    day_data["day_sin"],
    marker="o",
    label="day_sin"
)
axes[1, 1].plot(
    day_data["day_of_week"],
    day_data["day_cos"],
    marker="o",
    label="day_cos"
)
axes[1, 1].set_title("Day of Week - Cyclical Encoding")
axes[1, 1].set_xlabel("Day of Week (0=Monday, 6=Sunday)")
axes[1, 1].set_ylabel("Encoded value")
axes[1, 1].set_xticks(range(7))
axes[1, 1].legend()
axes[1, 1].grid(True)


# 3. ماه سال
axes[2, 0].plot(
    month_data["month"],
    month_data["month"],
    marker="o"
)
axes[2, 0].set_title("Month - Simple Numeric Encoding")
axes[2, 0].set_xlabel("Month")
axes[2, 0].set_ylabel("Numeric value")
axes[2, 0].grid(True)

axes[2, 1].plot(
    month_data["month"],
    month_data["month_sin"],
    marker="o",
    label="month_sin"
)
axes[2, 1].plot(
    month_data["month"],
    month_data["month_cos"],
    marker="o",
    label="month_cos"
)
axes[2, 1].set_title("Month - Cyclical Encoding")
axes[2, 1].set_xlabel("Month")
axes[2, 1].set_ylabel("Encoded value")
axes[2, 1].legend()
axes[2, 1].grid(True)

plt.tight_layout()
plt.show()
```


    
![png](output_7_0.png)
    



```python

# ویژگی‌های تأخیری (Lag Features)

for lag in [1, 2, 3, 24, 168]:
    temporal_features[f"lag_{lag}"] = (
        temporal_features["trip_count"].shift(lag)
    )

# استفاده از مقادیر گذشته، بدون وارد کردن مقدار ساعت جاری
past_counts = temporal_features["trip_count"].shift(1)

# ویژگی‌های میانگین و انحراف معیار متحرک
for window in [3, 24, 168]:
    temporal_features[f"rolling_mean_{window}"] = (
        past_counts.rolling(window=window, min_periods=window).mean()
    )

    temporal_features[f"rolling_std_{window}"] = (
        past_counts.rolling(window=window, min_periods=window).std()
    )

print("Lag and rolling features created successfully.")

display(
    temporal_features[
        [
            "pickup_hour",
            "trip_count",
            "lag_1",
            "lag_2",
            "lag_3",
            "lag_24",
            "lag_168",
            "rolling_mean_24",
            "rolling_mean_168"
        ]
    ].iloc[165:175]
)

```

    Lag and rolling features created successfully.
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pickup_hour</th>
      <th>trip_count</th>
      <th>lag_1</th>
      <th>lag_2</th>
      <th>lag_3</th>
      <th>lag_24</th>
      <th>lag_168</th>
      <th>rolling_mean_24</th>
      <th>rolling_mean_168</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>165</th>
      <td>2024-01-07 21:00:00-05:00</td>
      <td>2965</td>
      <td>3264.0</td>
      <td>3603.0</td>
      <td>4088.0</td>
      <td>5065.0</td>
      <td>NaN</td>
      <td>3179.208333</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>166</th>
      <td>2024-01-07 22:00:00-05:00</td>
      <td>2443</td>
      <td>2965.0</td>
      <td>3264.0</td>
      <td>3603.0</td>
      <td>5552.0</td>
      <td>NaN</td>
      <td>3091.708333</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>167</th>
      <td>2024-01-07 23:00:00-05:00</td>
      <td>1567</td>
      <td>2443.0</td>
      <td>2965.0</td>
      <td>3264.0</td>
      <td>5116.0</td>
      <td>NaN</td>
      <td>2962.166667</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>168</th>
      <td>2024-01-08 00:00:00-05:00</td>
      <td>992</td>
      <td>1567.0</td>
      <td>2443.0</td>
      <td>2965.0</td>
      <td>3876.0</td>
      <td>6596.0</td>
      <td>2814.291667</td>
      <td>3629.154762</td>
    </tr>
    <tr>
      <th>169</th>
      <td>2024-01-08 01:00:00-05:00</td>
      <td>425</td>
      <td>992.0</td>
      <td>1567.0</td>
      <td>2443.0</td>
      <td>2735.0</td>
      <td>7355.0</td>
      <td>2694.125000</td>
      <td>3595.797619</td>
    </tr>
    <tr>
      <th>170</th>
      <td>2024-01-08 02:00:00-05:00</td>
      <td>212</td>
      <td>425.0</td>
      <td>992.0</td>
      <td>1567.0</td>
      <td>1929.0</td>
      <td>6220.0</td>
      <td>2597.875000</td>
      <td>3554.547619</td>
    </tr>
    <tr>
      <th>171</th>
      <td>2024-01-08 03:00:00-05:00</td>
      <td>204</td>
      <td>212.0</td>
      <td>425.0</td>
      <td>992.0</td>
      <td>1120.0</td>
      <td>4936.0</td>
      <td>2526.333333</td>
      <td>3518.785714</td>
    </tr>
    <tr>
      <th>172</th>
      <td>2024-01-08 04:00:00-05:00</td>
      <td>241</td>
      <td>204.0</td>
      <td>212.0</td>
      <td>425.0</td>
      <td>541.0</td>
      <td>3188.0</td>
      <td>2488.166667</td>
      <td>3490.619048</td>
    </tr>
    <tr>
      <th>173</th>
      <td>2024-01-08 05:00:00-05:00</td>
      <td>596</td>
      <td>241.0</td>
      <td>204.0</td>
      <td>212.0</td>
      <td>395.0</td>
      <td>1522.0</td>
      <td>2475.666667</td>
      <td>3473.077381</td>
    </tr>
    <tr>
      <th>174</th>
      <td>2024-01-08 06:00:00-05:00</td>
      <td>1586</td>
      <td>596.0</td>
      <td>241.0</td>
      <td>204.0</td>
      <td>677.0</td>
      <td>1532.0</td>
      <td>2484.041667</td>
      <td>3467.565476</td>
    </tr>
  </tbody>
</table>
</div>


###  ویژگی های مکانی


```python
#تعریف مسیر و شناسایی فایل‌ها

from pathlib import Path

# مسیر پوشه خروجی‌های پردازش‌شده
PROCESSED_DIR = Path(r"D:\کارشناسی\تحلیل داده\data\processed")

# Monthly processed trip files
MONTHLY_FILES = [
    "trip_features_2024-01.parquet",
    "trip_features_2024-02.parquet",
    "trip_features_2024-03.parquet"
]

# بررسی وجود پوشه
assert PROCESSED_DIR.exists(), f"Processed directory not found: {PROCESSED_DIR}"

print("Processed directory:", PROCESSED_DIR)
print("\nAvailable processed files:")

for path in sorted(PROCESSED_DIR.glob("*.parquet")):
    print(f"{path.name} | {path.stat().st_size / (1024**2):.2f} MB")
```

    Processed directory: D:\کارشناسی\تحلیل داده\data\processed
    
    Available processed files:
    hourly_trip_counts.parquet | 0.03 MB
    hourly_zone_demand.parquet | 0.47 MB
    od_patterns.parquet | 0.09 MB
    trip_features_2024-01.parquet | 75.20 MB
    trip_features_2024-02.parquet | 76.01 MB
    trip_features_2024-03.parquet | 91.32 MB
    zone_features.parquet | 0.04 MB
    


```python
# بررسی ساختار داده‌های مکانی

import pandas as pd

# خواندن دو فایل مکانی موجود
zone_df = pd.read_parquet(PROCESSED_DIR / "zone_features.parquet")
hourly_zone_df = pd.read_parquet(PROCESSED_DIR / "hourly_zone_demand.parquet")


# تابع نمایش اطلاعات ساختاری
def inspect_dataframe(df, name):

    print("=" * 70)
    print(f"DATASET: {name}")
    print("=" * 70)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst 5 rows:")
    display(df.head())

    print("\nMissing values:")
    display(df.isna().sum())


# بررسی جدول ویژگی‌های مناطق
inspect_dataframe(zone_df, "zone_features")

# بررسی تقاضای ساعتی مناطق
inspect_dataframe(hourly_zone_df, "hourly_zone_demand")
```

    ======================================================================
    DATASET: zone_features
    ======================================================================
    
    Shape:
    (262, 15)
    
    Columns:
    ['PULocationID', 'pickup_trips', 'distance_sum', 'distance_n', 'duration_sum', 'duration_n', 'fare_sum', 'fare_n', 'speed_sum', 'speed_n', 'avg_trip_distance', 'avg_duration_minutes', 'avg_fare_amount', 'avg_speed_mph', 'is_unknown_zone']
    
    Data types:
    PULocationID              int32
    pickup_trips              int64
    distance_sum            float64
    distance_n                int64
    duration_sum            float64
    duration_n                int64
    fare_sum                float64
    fare_n                    int64
    speed_sum               float64
    speed_n                   int64
    avg_trip_distance       float64
    avg_duration_minutes    float64
    avg_fare_amount         float64
    avg_speed_mph           float64
    is_unknown_zone            bool
    dtype: object
    
    First 5 rows:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PULocationID</th>
      <th>pickup_trips</th>
      <th>distance_sum</th>
      <th>distance_n</th>
      <th>duration_sum</th>
      <th>duration_n</th>
      <th>fare_sum</th>
      <th>fare_n</th>
      <th>speed_sum</th>
      <th>speed_n</th>
      <th>avg_trip_distance</th>
      <th>avg_duration_minutes</th>
      <th>avg_fare_amount</th>
      <th>avg_speed_mph</th>
      <th>is_unknown_zone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>954</td>
      <td>1220.53</td>
      <td>954</td>
      <td>3931.866667</td>
      <td>954</td>
      <td>79682.46</td>
      <td>954</td>
      <td>61019.520262</td>
      <td>954</td>
      <td>1.279382</td>
      <td>4.121454</td>
      <td>83.524591</td>
      <td>63.961761</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>14</td>
      <td>141.16</td>
      <td>14</td>
      <td>381.366667</td>
      <td>14</td>
      <td>543.95</td>
      <td>14</td>
      <td>331.690358</td>
      <td>14</td>
      <td>10.082857</td>
      <td>27.240476</td>
      <td>38.853571</td>
      <td>23.692168</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>350</td>
      <td>2661.88</td>
      <td>350</td>
      <td>14174.550000</td>
      <td>350</td>
      <td>12753.69</td>
      <td>350</td>
      <td>6439.090486</td>
      <td>350</td>
      <td>7.605371</td>
      <td>40.498714</td>
      <td>36.439114</td>
      <td>18.397401</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>15879</td>
      <td>139911.40</td>
      <td>15879</td>
      <td>246107.600000</td>
      <td>15879</td>
      <td>267036.89</td>
      <td>15879</td>
      <td>389918.389097</td>
      <td>15875</td>
      <td>8.811096</td>
      <td>15.498936</td>
      <td>16.816984</td>
      <td>24.561788</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1</td>
      <td>0.00</td>
      <td>1</td>
      <td>0.166667</td>
      <td>1</td>
      <td>100.00</td>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.166667</td>
      <td>100.000000</td>
      <td>0.000000</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>


    
    Missing values:
    


    PULocationID            0
    pickup_trips            0
    distance_sum            0
    distance_n              0
    duration_sum            0
    duration_n              0
    fare_sum                0
    fare_n                  0
    speed_sum               0
    speed_n                 0
    avg_trip_distance       0
    avg_duration_minutes    0
    avg_fare_amount         0
    avg_speed_mph           0
    is_unknown_zone         0
    dtype: int64


    ======================================================================
    DATASET: hourly_zone_demand
    ======================================================================
    
    Shape:
    (240917, 3)
    
    Columns:
    ['pickup_hour', 'PULocationID', 'trip_count']
    
    Data types:
    pickup_hour     datetime64[us, America/New_York]
    PULocationID                               int32
    trip_count                                 int64
    dtype: object
    
    First 5 rows:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pickup_hour</th>
      <th>PULocationID</th>
      <th>trip_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-01-01 00:00:00-05:00</td>
      <td>4</td>
      <td>25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-01-01 00:00:00-05:00</td>
      <td>7</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-01-01 00:00:00-05:00</td>
      <td>9</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-01-01 00:00:00-05:00</td>
      <td>10</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-01-01 00:00:00-05:00</td>
      <td>12</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>


    
    Missing values:
    


    pickup_hour     0
    PULocationID    0
    trip_count      0
    dtype: int64



```python
import pandas as pd

# 1. بررسی الگوهای مبدأ و مقصد
od_df = pd.read_parquet(
    PROCESSED_DIR / "od_patterns.parquet"
)

print("=" * 60)
print("OD PATTERNS")
print("=" * 60)

print("Shape:", od_df.shape)

print("\nColumns:")
print(od_df.columns.tolist())

print("\nData types:")
print(od_df.dtypes)

print("\nFirst 10 rows:")
display(od_df.head(10))

print("\nMissing values:")
display(od_df.isna().sum())


# 2. بررسی ستون‌های مکانی فایل سفر ژانویه
sample_trip_df = pd.read_parquet(
    PROCESSED_DIR / "trip_features_2024-01.parquet"
)

print("\n" + "=" * 60)
print("TRIP FEATURES - JANUARY 2024")
print("=" * 60)

print("Shape:", sample_trip_df.shape)

print("\nAll columns:")
print(sample_trip_df.columns.tolist())

spatial_columns = [
    col for col in [
        "PULocationID",
        "DOLocationID",
        "trip_distance"
    ]
    if col in sample_trip_df.columns
]

print("\nRelevant spatial columns:")
display(sample_trip_df[spatial_columns].head(10))

print("\nMissing values in spatial columns:")
display(sample_trip_df[spatial_columns].isna().sum())
```

    ============================================================
    OD PATTERNS
    ============================================================
    Shape: (37255, 3)
    
    Columns:
    ['PULocationID', 'DOLocationID', 'trip_count']
    
    Data types:
    PULocationID    int32
    DOLocationID    int32
    trip_count      int64
    dtype: object
    
    First 10 rows:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PULocationID</th>
      <th>DOLocationID</th>
      <th>trip_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>849</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>6</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>25</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>48</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>68</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>92</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>100</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>132</td>
      <td>3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>138</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>


    
    Missing values:
    


    PULocationID    0
    DOLocationID    0
    trip_count      0
    dtype: int64


    
    ============================================================
    TRIP FEATURES - JANUARY 2024
    ============================================================
    Shape: (2964609, 11)
    
    All columns:
    ['pickup_time_local', 'dropoff_time_local', 'pickup_hour', 'PULocationID', 'DOLocationID', 'trip_distance', 'fare_amount', 'total_amount', 'passenger_count', 'duration_minutes', 'speed_mph']
    
    Relevant spatial columns:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PULocationID</th>
      <th>DOLocationID</th>
      <th>trip_distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>186</td>
      <td>79</td>
      <td>1.72</td>
    </tr>
    <tr>
      <th>1</th>
      <td>140</td>
      <td>236</td>
      <td>1.80</td>
    </tr>
    <tr>
      <th>2</th>
      <td>236</td>
      <td>79</td>
      <td>4.70</td>
    </tr>
    <tr>
      <th>3</th>
      <td>79</td>
      <td>211</td>
      <td>1.40</td>
    </tr>
    <tr>
      <th>4</th>
      <td>211</td>
      <td>148</td>
      <td>0.80</td>
    </tr>
    <tr>
      <th>5</th>
      <td>148</td>
      <td>141</td>
      <td>4.70</td>
    </tr>
    <tr>
      <th>6</th>
      <td>138</td>
      <td>181</td>
      <td>10.82</td>
    </tr>
    <tr>
      <th>7</th>
      <td>246</td>
      <td>231</td>
      <td>3.00</td>
    </tr>
    <tr>
      <th>8</th>
      <td>161</td>
      <td>261</td>
      <td>5.44</td>
    </tr>
    <tr>
      <th>9</th>
      <td>113</td>
      <td>113</td>
      <td>0.04</td>
    </tr>
  </tbody>
</table>
</div>


    
    Missing values in spatial columns:
    


    PULocationID     0
    DOLocationID     0
    trip_distance    0
    dtype: int64



```python
# بررسی سرعت
# Distance validity thresholds

MIN_VALID_DISTANCE = 0
MAX_VALID_DISTANCE = 100

print("Distance validity rule:")
print(f"trip_distance > {MIN_VALID_DISTANCE}")
print(f"trip_distance <= {MAX_VALID_DISTANCE}")
```

    Distance validity rule:
    trip_distance > 0
    trip_distance <= 100
    


```python
# بررسی اولیه کیفیت سرعت در سه فایل
speed_audit = []

for filename in MONTHLY_FILES:

    df = pd.read_parquet(
        PROCESSED_DIR / filename,
        columns=["trip_distance", "duration_minutes", "speed_mph"]
    )

    valid_distance = (
        df["trip_distance"].gt(MIN_VALID_DISTANCE)
        & df["trip_distance"].le(MAX_VALID_DISTANCE)
    )

    valid_speed = (
        valid_distance
        & df["duration_minutes"].gt(0)
        & df["speed_mph"].notna()
    )

    speed_audit.append({
        "file": filename,
        "total_records": len(df),
        "missing_speed": df["speed_mph"].isna().sum(),
        "zero_or_negative_duration": df["duration_minutes"].le(0).sum(),
        "valid_distance_records": valid_distance.sum(),
        "valid_speed_records": valid_speed.sum(),
        "speed_without_valid_distance": (
            (~valid_distance) & df["speed_mph"].notna()
        ).sum()
    })

    del df

speed_audit_df = pd.DataFrame(speed_audit)

display(speed_audit_df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>file</th>
      <th>total_records</th>
      <th>missing_speed</th>
      <th>zero_or_negative_duration</th>
      <th>valid_distance_records</th>
      <th>valid_speed_records</th>
      <th>speed_without_valid_distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>trip_features_2024-01.parquet</td>
      <td>2964609</td>
      <td>870</td>
      <td>870</td>
      <td>2904179</td>
      <td>2904067</td>
      <td>59672</td>
    </tr>
    <tr>
      <th>1</th>
      <td>trip_features_2024-02.parquet</td>
      <td>3007523</td>
      <td>803</td>
      <td>803</td>
      <td>2939227</td>
      <td>2939184</td>
      <td>67536</td>
    </tr>
    <tr>
      <th>2</th>
      <td>trip_features_2024-03.parquet</td>
      <td>3582624</td>
      <td>1131</td>
      <td>1128</td>
      <td>3495334</td>
      <td>3495141</td>
      <td>86352</td>
    </tr>
  </tbody>
</table>
</div>



```python
# جدول ویژگی‌های مقصد (نسخه اولیه)
import pandas as pd

# Columns to aggregate for destination zones
destination_parts = []

for filename in MONTHLY_FILES:

    df = pd.read_parquet(PROCESSED_DIR / filename)

    monthly_destination = (
        df.groupby("DOLocationID")
        .agg(
            dropoff_trips=("DOLocationID", "size"),

            distance_sum=("trip_distance", "sum"),
            distance_n=("trip_distance", "count"),

            duration_sum=("duration_minutes", "sum"),
            duration_n=("duration_minutes", "count"),

            fare_sum=("fare_amount", "sum"),
            fare_n=("fare_amount", "count"),

            speed_sum=("speed_mph", "sum"),
            speed_n=("speed_mph", "count")
        )
    )

    destination_parts.append(monthly_destination)

    del df


# Combine monthly aggregates
destination_features = (
    pd.concat(destination_parts)
    .groupby(level=0)
    .sum()
)

destination_features.index.name = "DOLocationID"


# Calculate averages
destination_features["avg_trip_distance"] = (
    destination_features["distance_sum"]
    / destination_features["distance_n"].replace(0, float("nan"))
)

destination_features["avg_duration_minutes"] = (
    destination_features["duration_sum"]
    / destination_features["duration_n"].replace(0, float("nan"))
)

destination_features["avg_fare_amount"] = (
    destination_features["fare_sum"]
    / destination_features["fare_n"].replace(0, float("nan"))
)

destination_features["avg_speed_mph"] = (
    destination_features["speed_sum"]
    / destination_features["speed_n"].replace(0, float("nan"))
)


# Identify unknown TLC zone IDs
destination_features["is_unknown_zone"] = (
    destination_features.index.isin([264, 265])
)


# Display results
print("Destination features shape:")
print(destination_features.shape)

print("\nColumns:")
print(destination_features.columns.tolist())

print("\nFirst 10 rows:")
display(destination_features.head(10))

print("\nMissing values:")
display(destination_features.isna().sum())

print("\nTotal destination trips:")
print(destination_features["dropoff_trips"].sum())
```

    Destination features shape:
    (261, 14)
    
    Columns:
    ['dropoff_trips', 'distance_sum', 'distance_n', 'duration_sum', 'duration_n', 'fare_sum', 'fare_n', 'speed_sum', 'speed_n', 'avg_trip_distance', 'avg_duration_minutes', 'avg_fare_amount', 'avg_speed_mph', 'is_unknown_zone']
    
    First 10 rows:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dropoff_trips</th>
      <th>distance_sum</th>
      <th>distance_n</th>
      <th>duration_sum</th>
      <th>duration_n</th>
      <th>fare_sum</th>
      <th>fare_n</th>
      <th>speed_sum</th>
      <th>speed_n</th>
      <th>avg_trip_distance</th>
      <th>avg_duration_minutes</th>
      <th>avg_fare_amount</th>
      <th>avg_speed_mph</th>
      <th>is_unknown_zone</th>
    </tr>
    <tr>
      <th>DOLocationID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>22950</td>
      <td>627370.67</td>
      <td>22950</td>
      <td>878711.150000</td>
      <td>22950</td>
      <td>2072372.68</td>
      <td>22950</td>
      <td>1.115675e+06</td>
      <td>22950</td>
      <td>27.336413</td>
      <td>38.288068</td>
      <td>90.299463</td>
      <td>48.613274</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17</td>
      <td>164.88</td>
      <td>17</td>
      <td>542.550000</td>
      <td>17</td>
      <td>593.50</td>
      <td>17</td>
      <td>3.472241e+02</td>
      <td>17</td>
      <td>9.698824</td>
      <td>31.914706</td>
      <td>34.911765</td>
      <td>20.424949</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>763</td>
      <td>9324.84</td>
      <td>763</td>
      <td>28467.183333</td>
      <td>763</td>
      <td>35695.71</td>
      <td>763</td>
      <td>1.698714e+04</td>
      <td>762</td>
      <td>12.221284</td>
      <td>37.309546</td>
      <td>46.783368</td>
      <td>22.292831</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>39888</td>
      <td>110417.19</td>
      <td>39888</td>
      <td>612009.583333</td>
      <td>39888</td>
      <td>654561.89</td>
      <td>39888</td>
      <td>4.356399e+05</td>
      <td>39887</td>
      <td>2.768181</td>
      <td>15.343201</td>
      <td>16.409995</td>
      <td>10.921852</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>33</td>
      <td>973.27</td>
      <td>33</td>
      <td>1757.616667</td>
      <td>33</td>
      <td>3572.40</td>
      <td>33</td>
      <td>1.147913e+03</td>
      <td>33</td>
      <td>29.493030</td>
      <td>53.261111</td>
      <td>108.254545</td>
      <td>34.785255</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>200</td>
      <td>2601.32</td>
      <td>200</td>
      <td>7623.266667</td>
      <td>200</td>
      <td>9633.61</td>
      <td>200</td>
      <td>6.473848e+03</td>
      <td>199</td>
      <td>13.006600</td>
      <td>38.116333</td>
      <td>48.168050</td>
      <td>32.531898</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>23412</td>
      <td>160408.86</td>
      <td>23412</td>
      <td>536209.216667</td>
      <td>23412</td>
      <td>607499.38</td>
      <td>23412</td>
      <td>3.942493e+05</td>
      <td>23411</td>
      <td>6.851566</td>
      <td>22.903179</td>
      <td>25.948205</td>
      <td>16.840344</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>118</td>
      <td>793.16</td>
      <td>118</td>
      <td>2522.833333</td>
      <td>118</td>
      <td>3402.94</td>
      <td>118</td>
      <td>2.704402e+03</td>
      <td>118</td>
      <td>6.721695</td>
      <td>21.379944</td>
      <td>28.838475</td>
      <td>22.918661</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>854</td>
      <td>8366.23</td>
      <td>854</td>
      <td>22703.350000</td>
      <td>854</td>
      <td>33753.05</td>
      <td>854</td>
      <td>2.018687e+04</td>
      <td>854</td>
      <td>9.796522</td>
      <td>26.584719</td>
      <td>39.523478</td>
      <td>23.638026</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>8177</td>
      <td>40889.95</td>
      <td>8177</td>
      <td>139134.750000</td>
      <td>8177</td>
      <td>183982.34</td>
      <td>8177</td>
      <td>1.958154e+05</td>
      <td>8176</td>
      <td>5.000605</td>
      <td>17.015379</td>
      <td>22.499980</td>
      <td>23.950028</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>


    
    Missing values:
    


    dropoff_trips           0
    distance_sum            0
    distance_n              0
    duration_sum            0
    duration_n              0
    fare_sum                0
    fare_n                  0
    speed_sum               0
    speed_n                 0
    avg_trip_distance       0
    avg_duration_minutes    0
    avg_fare_amount         0
    avg_speed_mph           0
    is_unknown_zone         0
    dtype: int64


    
    Total destination trips:
    9554756
    


```python
# Compare pickup and dropoff zone IDs

pickup_zones = set(zone_df["PULocationID"])
dropoff_zones = set(destination_features.index)

print("Number of pickup zones:", len(pickup_zones))
print("Number of dropoff zones:", len(dropoff_zones))

print("\nZones present in pickup but absent in dropoff:")
print(sorted(pickup_zones - dropoff_zones))

print("\nZones present in dropoff but absent in pickup:")
print(sorted(dropoff_zones - pickup_zones))


# Check unknown zones
print("\nUnknown zones in pickup:")
print(sorted(zone_df.loc[
    zone_df["is_unknown_zone"], "PULocationID"
].tolist()))

print("\nUnknown zones in dropoff:")
print(sorted(destination_features.index[
    destination_features["is_unknown_zone"]
].tolist()))


# Check speed observations
print("\nTrips without valid speed observations:")
print(
    destination_features["dropoff_trips"]
    - destination_features["speed_n"]
)

print("\nTotal trips without valid speed:")
print(
    (destination_features["dropoff_trips"]
     - destination_features["speed_n"]).sum()
)
```

    Number of pickup zones: 262
    Number of dropoff zones: 261
    
    Zones present in pickup but absent in dropoff:
    [199]
    
    Zones present in dropoff but absent in pickup:
    []
    
    Unknown zones in pickup:
    [264, 265]
    
    Unknown zones in dropoff:
    [264, 265]
    
    Trips without valid speed observations:
    DOLocationID
    1         0
    2         0
    3         1
    4         1
    5         0
           ... 
    261       3
    262      12
    263      10
    264    2205
    265       2
    Length: 261, dtype: int64
    
    Total trips without valid speed:
    2804
    


```python
#ویژگی‌های مسیر مبدأ ـ مقصد و فاصله(نسخه اولیه)
import pandas as pd
import numpy as np

# Aggregate trip distances by origin-destination pair
od_distance_parts = []

for filename in MONTHLY_FILES:

    df = pd.read_parquet(
        PROCESSED_DIR / filename,
        columns=[
            "PULocationID",
            "DOLocationID",
            "trip_distance"
        ]
    )

    monthly_od = (
        df.groupby(["PULocationID", "DOLocationID"])
        .agg(
            distance_sum=("trip_distance", "sum"),
            distance_n=("trip_distance", "count")
        )
    )

    od_distance_parts.append(monthly_od)

    del df


# Combine all three months
od_distance_summary = (
    pd.concat(od_distance_parts)
    .groupby(level=[0, 1])
    .sum()
)


# Combine with existing OD trip counts
od_features = (
    od_df.set_index(["PULocationID", "DOLocationID"])
    .join(od_distance_summary, how="left")
)

od_features = od_features.rename(
    columns={"trip_count": "od_trip_count"}
)


# Calculate average distance and route share
od_features["avg_od_distance"] = (
    od_features["distance_sum"]
    / od_features["distance_n"].replace(0, np.nan)
)

od_features["od_trip_share"] = (
    od_features["od_trip_count"]
    / od_features["od_trip_count"].sum()
)


# Display results
print("OD features shape:")
print(od_features.shape)

print("\nColumns:")
print(od_features.columns.tolist())

print("\nTop 10 routes by trip count:")
display(
    od_features
    .sort_values("od_trip_count", ascending=False)
    .head(10)
)

print("\nMissing values:")
display(od_features.isna().sum())

print("\nTotal OD trips:")
print(od_features["od_trip_count"].sum())

print("\nTotal distance observations:")
print(od_features["distance_n"].sum())

print("\nAverage route distance statistics:")
display(od_features["avg_od_distance"].describe())
```

    OD features shape:
    (37255, 5)
    
    Columns:
    ['od_trip_count', 'distance_sum', 'distance_n', 'avg_od_distance', 'od_trip_share']
    
    Top 10 routes by trip count:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>od_trip_count</th>
      <th>distance_sum</th>
      <th>distance_n</th>
      <th>avg_od_distance</th>
      <th>od_trip_share</th>
    </tr>
    <tr>
      <th>PULocationID</th>
      <th>DOLocationID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>237</th>
      <th>236</th>
      <td>64384</td>
      <td>67957.37</td>
      <td>64384</td>
      <td>1.055501</td>
      <td>0.006738</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">236</th>
      <th>237</th>
      <td>57111</td>
      <td>59297.68</td>
      <td>57111</td>
      <td>1.038288</td>
      <td>0.005977</td>
    </tr>
    <tr>
      <th>236</th>
      <td>47509</td>
      <td>29441.85</td>
      <td>47509</td>
      <td>0.619711</td>
      <td>0.004972</td>
    </tr>
    <tr>
      <th>237</th>
      <th>237</th>
      <td>44270</td>
      <td>27616.57</td>
      <td>44270</td>
      <td>0.623821</td>
      <td>0.004633</td>
    </tr>
    <tr>
      <th>161</th>
      <th>237</th>
      <td>31118</td>
      <td>33257.86</td>
      <td>31118</td>
      <td>1.068766</td>
      <td>0.003257</td>
    </tr>
    <tr>
      <th>142</th>
      <th>239</th>
      <td>27085</td>
      <td>26829.55</td>
      <td>27085</td>
      <td>0.990569</td>
      <td>0.002835</td>
    </tr>
    <tr>
      <th>237</th>
      <th>161</th>
      <td>26836</td>
      <td>28364.32</td>
      <td>26836</td>
      <td>1.056950</td>
      <td>0.002809</td>
    </tr>
    <tr>
      <th>239</th>
      <th>142</th>
      <td>26828</td>
      <td>23561.59</td>
      <td>26828</td>
      <td>0.878246</td>
      <td>0.002808</td>
    </tr>
    <tr>
      <th>161</th>
      <th>236</th>
      <td>26504</td>
      <td>51341.68</td>
      <td>26504</td>
      <td>1.937129</td>
      <td>0.002774</td>
    </tr>
    <tr>
      <th>239</th>
      <th>238</th>
      <td>25824</td>
      <td>21494.15</td>
      <td>25824</td>
      <td>0.832332</td>
      <td>0.002703</td>
    </tr>
  </tbody>
</table>
</div>


    
    Missing values:
    


    od_trip_count      0
    distance_sum       0
    distance_n         0
    avg_od_distance    0
    od_trip_share      0
    dtype: int64


    
    Total OD trips:
    9554756
    
    Total distance observations:
    9554756
    
    Average route distance statistics:
    


    count    37255.000000
    mean        17.403178
    std        328.884662
    min          0.000000
    25%          4.618575
    50%          8.300000
    75%         12.660000
    max      27561.150000
    Name: avg_od_distance, dtype: float64



```python
# 1. Check distance statistics at trip level

print("Trip distance statistics:")

distance_values = []

for filename in MONTHLY_FILES:
    df = pd.read_parquet(
        PROCESSED_DIR / filename,
        columns=["trip_distance"]
    )

    distance_values.append(df["trip_distance"])

    print(f"\n{filename}")
    print("Records:", len(df))
    print("Zero distances:", (df["trip_distance"] == 0).sum())
    print("Negative distances:", (df["trip_distance"] < 0).sum())
    print("Distances above 100 miles:",
          (df["trip_distance"] > 100).sum())

all_distances = pd.concat(distance_values, ignore_index=True)

display(all_distances.describe(
    percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999]
))


# 2. Inspect routes with the largest average distances

print("\nTop 15 routes by average distance:")

display(
    od_features
    .sort_values("avg_od_distance", ascending=False)
    .head(15)
)


# 3. Check consistency of distance counts

print("\nRoutes where distance count differs from trip count:")

display(
    od_features.loc[
        od_features["distance_n"] != od_features["od_trip_count"],
        ["od_trip_count", "distance_n", "avg_od_distance"]
    ].head(20)
)

print(
    "Number of affected routes:",
    (od_features["distance_n"] != od_features["od_trip_count"]).sum()
)
```

    Trip distance statistics:
    
    trip_features_2024-01.parquet
    Records: 2964609
    Zero distances: 60371
    Negative distances: 0
    Distances above 100 miles: 59
    
    trip_features_2024-02.parquet
    Records: 3007523
    Zero distances: 68225
    Negative distances: 0
    Distances above 100 miles: 71
    
    trip_features_2024-03.parquet
    Records: 3582624
    Zero distances: 87168
    Negative distances: 0
    Distances above 100 miles: 122
    


    count    9.554756e+06
    mean     4.042288e+00
    std      2.654786e+02
    min      0.000000e+00
    25%      1.000000e+00
    50%      1.700000e+00
    75%      3.190000e+00
    90%      8.300000e+00
    95%      1.301000e+01
    99%      1.987000e+01
    99.9%    2.922000e+01
    max      3.127223e+05
    Name: trip_distance, dtype: float64


    
    Top 15 routes by average distance:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>od_trip_count</th>
      <th>distance_sum</th>
      <th>distance_n</th>
      <th>avg_od_distance</th>
      <th>od_trip_share</th>
    </tr>
    <tr>
      <th>PULocationID</th>
      <th>DOLocationID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14</th>
      <th>265</th>
      <td>2</td>
      <td>55122.30</td>
      <td>2</td>
      <td>27561.150000</td>
      <td>2.093198e-07</td>
    </tr>
    <tr>
      <th>256</th>
      <th>205</th>
      <td>3</td>
      <td>66930.90</td>
      <td>3</td>
      <td>22310.300000</td>
      <td>3.139798e-07</td>
    </tr>
    <tr>
      <th>174</th>
      <th>37</th>
      <td>3</td>
      <td>55217.60</td>
      <td>3</td>
      <td>18405.866667</td>
      <td>3.139798e-07</td>
    </tr>
    <tr>
      <th>14</th>
      <th>7</th>
      <td>2</td>
      <td>36550.40</td>
      <td>2</td>
      <td>18275.200000</td>
      <td>2.093198e-07</td>
    </tr>
    <tr>
      <th>41</th>
      <th>17</th>
      <td>5</td>
      <td>83825.12</td>
      <td>5</td>
      <td>16765.024000</td>
      <td>5.232996e-07</td>
    </tr>
    <tr>
      <th>40</th>
      <th>43</th>
      <td>5</td>
      <td>82048.20</td>
      <td>5</td>
      <td>16409.640000</td>
      <td>5.232996e-07</td>
    </tr>
    <tr>
      <th>225</th>
      <th>265</th>
      <td>2</td>
      <td>32761.10</td>
      <td>2</td>
      <td>16380.550000</td>
      <td>2.093198e-07</td>
    </tr>
    <tr>
      <th>178</th>
      <th>75</th>
      <td>5</td>
      <td>81784.39</td>
      <td>5</td>
      <td>16356.878000</td>
      <td>5.232996e-07</td>
    </tr>
    <tr>
      <th>223</th>
      <th>56</th>
      <td>4</td>
      <td>44216.23</td>
      <td>4</td>
      <td>11054.057500</td>
      <td>4.186397e-07</td>
    </tr>
    <tr>
      <th>94</th>
      <th>170</th>
      <td>6</td>
      <td>61143.70</td>
      <td>6</td>
      <td>10190.616667</td>
      <td>6.279595e-07</td>
    </tr>
    <tr>
      <th>133</th>
      <th>62</th>
      <td>8</td>
      <td>79034.83</td>
      <td>8</td>
      <td>9879.353750</td>
      <td>8.372794e-07</td>
    </tr>
    <tr>
      <th>193</th>
      <th>70</th>
      <td>4</td>
      <td>38944.13</td>
      <td>4</td>
      <td>9736.032500</td>
      <td>4.186397e-07</td>
    </tr>
    <tr>
      <th>248</th>
      <th>90</th>
      <td>4</td>
      <td>34920.99</td>
      <td>4</td>
      <td>8730.247500</td>
      <td>4.186397e-07</td>
    </tr>
    <tr>
      <th>49</th>
      <th>265</th>
      <td>7</td>
      <td>60949.57</td>
      <td>7</td>
      <td>8707.081429</td>
      <td>7.326194e-07</td>
    </tr>
    <tr>
      <th>177</th>
      <th>265</th>
      <td>4</td>
      <td>32183.10</td>
      <td>4</td>
      <td>8045.775000</td>
      <td>4.186397e-07</td>
    </tr>
  </tbody>
</table>
</div>


    
    Routes where distance count differs from trip count:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>od_trip_count</th>
      <th>distance_n</th>
      <th>avg_od_distance</th>
    </tr>
    <tr>
      <th>PULocationID</th>
      <th>DOLocationID</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>


    Number of affected routes: 0
    


```python
# بررسی رکوردهای دارای فاصله بیش از 100 مایل

suspicious_parts = []

columns_to_read = [
    "pickup_time_local",
    "dropoff_time_local",
    "PULocationID",
    "DOLocationID",
    "trip_distance",
    "duration_minutes",
    "speed_mph",
    "fare_amount",
    "total_amount"
]

for filename in MONTHLY_FILES:
    df = pd.read_parquet(
        PROCESSED_DIR / filename,
        columns=columns_to_read
    )

    suspicious = df[df["trip_distance"] > 100].copy()
    suspicious["source_month"] = filename

    suspicious_parts.append(suspicious)

    del df

suspicious_trips = pd.concat(
    suspicious_parts,
    ignore_index=True
)

print("Number of suspicious trips:", len(suspicious_trips))

# نمایش 30 رکورد با بیشترین فاصله
display(
    suspicious_trips
    .sort_values("trip_distance", ascending=False)
    .head(30)
)

# آمار توصیفی ویژگی‌های مرتبط با این رکوردها
display(
    suspicious_trips[
        [
            "trip_distance",
            "duration_minutes",
            "speed_mph",
            "fare_amount",
            "total_amount"
        ]
    ].describe()
)
```

    Number of suspicious trips: 252
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pickup_time_local</th>
      <th>dropoff_time_local</th>
      <th>PULocationID</th>
      <th>DOLocationID</th>
      <th>trip_distance</th>
      <th>duration_minutes</th>
      <th>speed_mph</th>
      <th>fare_amount</th>
      <th>total_amount</th>
      <th>source_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>57</th>
      <td>2024-01-30 06:37:00-05:00</td>
      <td>2024-01-30 06:50:00-05:00</td>
      <td>151</td>
      <td>162</td>
      <td>312722.30</td>
      <td>13.000000</td>
      <td>1.443334e+06</td>
      <td>14.46</td>
      <td>22.15</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>106</th>
      <td>2024-02-07 06:56:00-05:00</td>
      <td>2024-02-07 07:03:00-05:00</td>
      <td>263</td>
      <td>140</td>
      <td>222478.29</td>
      <td>7.000000</td>
      <td>1.906957e+06</td>
      <td>11.74</td>
      <td>18.89</td>
      <td>trip_features_2024-02.parquet</td>
    </tr>
    <tr>
      <th>185</th>
      <td>2024-03-07 07:41:00-05:00</td>
      <td>2024-03-07 07:54:00-05:00</td>
      <td>263</td>
      <td>161</td>
      <td>176836.30</td>
      <td>13.000000</td>
      <td>8.161675e+05</td>
      <td>23.28</td>
      <td>32.74</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
    <tr>
      <th>243</th>
      <td>2024-03-29 10:43:00-04:00</td>
      <td>2024-03-29 11:18:00-04:00</td>
      <td>48</td>
      <td>1</td>
      <td>176744.79</td>
      <td>35.000000</td>
      <td>3.029911e+05</td>
      <td>73.52</td>
      <td>114.68</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
    <tr>
      <th>211</th>
      <td>2024-03-22 07:35:00-04:00</td>
      <td>2024-03-22 07:55:00-04:00</td>
      <td>262</td>
      <td>231</td>
      <td>176329.23</td>
      <td>20.000000</td>
      <td>5.289877e+05</td>
      <td>30.51</td>
      <td>34.51</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
    <tr>
      <th>121</th>
      <td>2024-02-22 08:19:00-05:00</td>
      <td>2024-02-22 08:31:00-05:00</td>
      <td>262</td>
      <td>75</td>
      <td>172935.77</td>
      <td>12.000000</td>
      <td>8.646788e+05</td>
      <td>14.73</td>
      <td>22.48</td>
      <td>trip_features_2024-02.parquet</td>
    </tr>
    <tr>
      <th>110</th>
      <td>2024-02-10 07:41:00-05:00</td>
      <td>2024-02-10 07:59:00-05:00</td>
      <td>239</td>
      <td>166</td>
      <td>154959.09</td>
      <td>18.000000</td>
      <td>5.165303e+05</td>
      <td>24.43</td>
      <td>28.43</td>
      <td>trip_features_2024-02.parquet</td>
    </tr>
    <tr>
      <th>196</th>
      <td>2024-03-16 18:31:00-04:00</td>
      <td>2024-03-16 19:26:00-04:00</td>
      <td>238</td>
      <td>132</td>
      <td>138097.21</td>
      <td>55.000000</td>
      <td>1.506515e+05</td>
      <td>83.32</td>
      <td>108.40</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
    <tr>
      <th>231</th>
      <td>2024-03-25 07:46:00-04:00</td>
      <td>2024-03-25 07:58:00-04:00</td>
      <td>238</td>
      <td>246</td>
      <td>136660.10</td>
      <td>12.000000</td>
      <td>6.833005e+05</td>
      <td>15.89</td>
      <td>19.89</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
    <tr>
      <th>241</th>
      <td>2024-03-28 08:41:00-04:00</td>
      <td>2024-03-28 09:05:00-04:00</td>
      <td>236</td>
      <td>234</td>
      <td>109619.96</td>
      <td>24.000000</td>
      <td>2.740499e+05</td>
      <td>24.65</td>
      <td>28.65</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
    <tr>
      <th>233</th>
      <td>2024-03-26 05:58:00-04:00</td>
      <td>2024-03-26 06:08:00-04:00</td>
      <td>263</td>
      <td>161</td>
      <td>105744.27</td>
      <td>10.000000</td>
      <td>6.344656e+05</td>
      <td>11.60</td>
      <td>15.60</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
    <tr>
      <th>70</th>
      <td>2024-02-09 10:30:19-05:00</td>
      <td>2024-02-09 11:33:14-05:00</td>
      <td>76</td>
      <td>163</td>
      <td>98229.40</td>
      <td>62.916667</td>
      <td>9.367572e+04</td>
      <td>52.50</td>
      <td>60.94</td>
      <td>trip_features_2024-02.parquet</td>
    </tr>
    <tr>
      <th>55</th>
      <td>2024-01-25 08:39:00-05:00</td>
      <td>2024-01-25 09:06:00-05:00</td>
      <td>4</td>
      <td>13</td>
      <td>97793.92</td>
      <td>27.000000</td>
      <td>2.173198e+05</td>
      <td>29.71</td>
      <td>36.31</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>230</th>
      <td>2024-03-25 06:07:00-04:00</td>
      <td>2024-03-25 06:26:00-04:00</td>
      <td>116</td>
      <td>142</td>
      <td>97054.04</td>
      <td>19.000000</td>
      <td>3.064864e+05</td>
      <td>22.69</td>
      <td>26.69</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
    <tr>
      <th>118</th>
      <td>2024-02-17 21:37:00-05:00</td>
      <td>2024-02-17 21:51:00-05:00</td>
      <td>148</td>
      <td>229</td>
      <td>87649.23</td>
      <td>14.000000</td>
      <td>3.756396e+05</td>
      <td>18.95</td>
      <td>27.54</td>
      <td>trip_features_2024-02.parquet</td>
    </tr>
    <tr>
      <th>68</th>
      <td>2024-02-09 08:42:52-05:00</td>
      <td>2024-02-09 09:32:31-05:00</td>
      <td>41</td>
      <td>17</td>
      <td>83778.60</td>
      <td>49.650000</td>
      <td>1.012430e+05</td>
      <td>39.50</td>
      <td>47.94</td>
      <td>trip_features_2024-02.parquet</td>
    </tr>
    <tr>
      <th>202</th>
      <td>2024-03-17 20:02:00-04:00</td>
      <td>2024-03-17 20:19:00-04:00</td>
      <td>202</td>
      <td>170</td>
      <td>82831.78</td>
      <td>17.000000</td>
      <td>2.923475e+05</td>
      <td>24.18</td>
      <td>28.18</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
    <tr>
      <th>50</th>
      <td>2024-01-20 08:01:00-05:00</td>
      <td>2024-01-20 08:26:00-05:00</td>
      <td>40</td>
      <td>43</td>
      <td>82015.45</td>
      <td>25.000000</td>
      <td>1.968371e+05</td>
      <td>16.56</td>
      <td>21.56</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>238</th>
      <td>2024-03-27 05:41:00-04:00</td>
      <td>2024-03-27 06:10:00-04:00</td>
      <td>178</td>
      <td>75</td>
      <td>81736.73</td>
      <td>29.000000</td>
      <td>1.691105e+05</td>
      <td>47.37</td>
      <td>62.24</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
    <tr>
      <th>181</th>
      <td>2024-03-02 17:50:00-05:00</td>
      <td>2024-03-02 18:04:00-05:00</td>
      <td>133</td>
      <td>62</td>
      <td>79013.74</td>
      <td>14.000000</td>
      <td>3.386303e+05</td>
      <td>18.39</td>
      <td>23.87</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
    <tr>
      <th>123</th>
      <td>2024-02-23 05:20:00-05:00</td>
      <td>2024-02-23 05:29:00-05:00</td>
      <td>238</td>
      <td>140</td>
      <td>77456.58</td>
      <td>9.000000</td>
      <td>5.163772e+05</td>
      <td>13.85</td>
      <td>21.42</td>
      <td>trip_features_2024-02.parquet</td>
    </tr>
    <tr>
      <th>111</th>
      <td>2024-02-10 10:07:00-05:00</td>
      <td>2024-02-10 10:21:00-05:00</td>
      <td>233</td>
      <td>142</td>
      <td>73523.51</td>
      <td>14.000000</td>
      <td>3.151008e+05</td>
      <td>17.16</td>
      <td>25.39</td>
      <td>trip_features_2024-02.parquet</td>
    </tr>
    <tr>
      <th>52</th>
      <td>2024-01-21 11:58:00-05:00</td>
      <td>2024-01-21 12:12:00-05:00</td>
      <td>209</td>
      <td>211</td>
      <td>72975.97</td>
      <td>14.000000</td>
      <td>3.127542e+05</td>
      <td>12.70</td>
      <td>20.04</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>49</th>
      <td>2024-01-17 08:41:00-05:00</td>
      <td>2024-01-17 09:23:00-05:00</td>
      <td>33</td>
      <td>161</td>
      <td>71752.26</td>
      <td>42.000000</td>
      <td>1.025032e+05</td>
      <td>41.06</td>
      <td>49.57</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>67</th>
      <td>2024-02-09 07:22:06-05:00</td>
      <td>2024-02-09 08:16:26-05:00</td>
      <td>62</td>
      <td>237</td>
      <td>69928.10</td>
      <td>54.333333</td>
      <td>7.722121e+04</td>
      <td>35.50</td>
      <td>43.94</td>
      <td>trip_features_2024-02.parquet</td>
    </tr>
    <tr>
      <th>244</th>
      <td>2024-03-29 14:50:00-04:00</td>
      <td>2024-03-29 15:03:00-04:00</td>
      <td>262</td>
      <td>162</td>
      <td>69590.74</td>
      <td>13.000000</td>
      <td>3.211880e+05</td>
      <td>20.55</td>
      <td>24.55</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
    <tr>
      <th>222</th>
      <td>2024-03-23 19:46:00-04:00</td>
      <td>2024-03-23 20:01:00-04:00</td>
      <td>195</td>
      <td>89</td>
      <td>69247.12</td>
      <td>15.000000</td>
      <td>2.769885e+05</td>
      <td>21.29</td>
      <td>22.79</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
    <tr>
      <th>228</th>
      <td>2024-03-25 06:49:00-04:00</td>
      <td>2024-03-25 07:01:00-04:00</td>
      <td>114</td>
      <td>246</td>
      <td>68481.47</td>
      <td>12.000000</td>
      <td>3.424073e+05</td>
      <td>-1.00</td>
      <td>3.00</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
    <tr>
      <th>114</th>
      <td>2024-02-11 12:34:00-05:00</td>
      <td>2024-02-11 12:46:00-05:00</td>
      <td>107</td>
      <td>230</td>
      <td>67156.35</td>
      <td>12.000000</td>
      <td>3.357818e+05</td>
      <td>16.75</td>
      <td>24.75</td>
      <td>trip_features_2024-02.parquet</td>
    </tr>
    <tr>
      <th>146</th>
      <td>2024-03-06 17:51:01-05:00</td>
      <td>2024-03-06 19:06:12-05:00</td>
      <td>256</td>
      <td>205</td>
      <td>66907.90</td>
      <td>75.183333</td>
      <td>5.339580e+04</td>
      <td>41.50</td>
      <td>43.00</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trip_distance</th>
      <th>duration_minutes</th>
      <th>speed_mph</th>
      <th>fare_amount</th>
      <th>total_amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>252.000000</td>
      <td>252.000000</td>
      <td>2.520000e+02</td>
      <td>252.000000</td>
      <td>252.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>31110.402421</td>
      <td>68.109722</td>
      <td>1.175901e+05</td>
      <td>151.311349</td>
      <td>164.308690</td>
    </tr>
    <tr>
      <th>std</th>
      <td>41360.769347</td>
      <td>68.934926</td>
      <td>2.089353e+05</td>
      <td>301.077373</td>
      <td>305.893973</td>
    </tr>
    <tr>
      <th>min</th>
      <td>100.520000</td>
      <td>2.000000</td>
      <td>2.214577e+01</td>
      <td>-899.000000</td>
      <td>-900.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>142.322500</td>
      <td>14.575000</td>
      <td>5.776145e+01</td>
      <td>16.552500</td>
      <td>22.425000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>20282.235000</td>
      <td>33.500000</td>
      <td>4.258820e+04</td>
      <td>32.490000</td>
      <td>37.575000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>49419.715000</td>
      <td>125.962500</td>
      <td>1.493227e+05</td>
      <td>298.957500</td>
      <td>303.997500</td>
    </tr>
    <tr>
      <th>max</th>
      <td>312722.300000</td>
      <td>446.750000</td>
      <td>1.906957e+06</td>
      <td>1616.500000</td>
      <td>1617.500000</td>
    </tr>
  </tbody>
</table>
</div>



```python
#ویژگی‌های مسیر مبدأ ـ مقصد و فاصله(نسخه اصلاحی ؛  با فاصله معتبر (کمتر از 100 مایل))
# Original trip files and od_features remain unchanged.

MAX_VALID_DISTANCE = 100

valid_od_parts = []

for filename in MONTHLY_FILES:

    df = pd.read_parquet(
        PROCESSED_DIR / filename,
        columns=[
            "PULocationID",
            "DOLocationID",
            "trip_distance"
        ]
    )

    valid_distance = df[
        df["trip_distance"].gt(0)
        & df["trip_distance"].le(MAX_VALID_DISTANCE)
    ].copy()

    monthly_od = (
        valid_distance
        .groupby(["PULocationID", "DOLocationID"])
        .agg(
            valid_distance_sum=("trip_distance", "sum"),
            valid_distance_n=("trip_distance", "count")
        )
    )

    valid_od_parts.append(monthly_od)

    del df, valid_distance

valid_od_summary = (
    pd.concat(valid_od_parts)
    .groupby(level=[0, 1])
    .sum()
)

od_features_clean = (
    od_features
    .drop(columns=[
        "distance_sum",
        "distance_n",
        "avg_od_distance"
    ])
    .join(valid_od_summary, how="left")
)

od_features_clean["avg_od_distance"] = (
    od_features_clean["valid_distance_sum"]
    / od_features_clean["valid_distance_n"].replace(0, np.nan)
)

print("Original OD features:", od_features.shape)
print("Clean OD features:", od_features_clean.shape)

print(
    "Routes without valid positive distance:",
    od_features_clean["avg_od_distance"].isna().sum()
)

print("\nClean average route distance statistics:")
display(od_features_clean["avg_od_distance"].describe())

print("\nTop 15 routes by clean average distance:")
display(
    od_features_clean
    .sort_values("avg_od_distance", ascending=False)
    .head(15)
)

```

    Original OD features: (37255, 5)
    Clean OD features: (37255, 5)
    Routes without valid positive distance: 769
    
    Clean average route distance statistics:
    


    count    36486.000000
    mean         9.639003
    std          5.657659
    min          0.010000
    25%          5.256038
    50%          9.011013
    75%         13.254750
    max         72.540000
    Name: avg_od_distance, dtype: float64


    
    Top 15 routes by clean average distance:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>od_trip_count</th>
      <th>od_trip_share</th>
      <th>valid_distance_sum</th>
      <th>valid_distance_n</th>
      <th>avg_od_distance</th>
    </tr>
    <tr>
      <th>PULocationID</th>
      <th>DOLocationID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>150</th>
      <th>265</th>
      <td>1</td>
      <td>1.046599e-07</td>
      <td>72.54</td>
      <td>1.0</td>
      <td>72.540000</td>
    </tr>
    <tr>
      <th>44</th>
      <th>265</th>
      <td>1</td>
      <td>1.046599e-07</td>
      <td>57.20</td>
      <td>1.0</td>
      <td>57.200000</td>
    </tr>
    <tr>
      <th>210</th>
      <th>265</th>
      <td>1</td>
      <td>1.046599e-07</td>
      <td>56.88</td>
      <td>1.0</td>
      <td>56.880000</td>
    </tr>
    <tr>
      <th>155</th>
      <th>265</th>
      <td>4</td>
      <td>4.186397e-07</td>
      <td>217.16</td>
      <td>4.0</td>
      <td>54.290000</td>
    </tr>
    <tr>
      <th>124</th>
      <th>265</th>
      <td>7</td>
      <td>7.326194e-07</td>
      <td>337.53</td>
      <td>7.0</td>
      <td>48.218571</td>
    </tr>
    <tr>
      <th>83</th>
      <th>265</th>
      <td>4</td>
      <td>4.186397e-07</td>
      <td>187.10</td>
      <td>4.0</td>
      <td>46.775000</td>
    </tr>
    <tr>
      <th>265</th>
      <th>27</th>
      <td>1</td>
      <td>1.046599e-07</td>
      <td>46.06</td>
      <td>1.0</td>
      <td>46.060000</td>
    </tr>
    <tr>
      <th>154</th>
      <th>265</th>
      <td>1</td>
      <td>1.046599e-07</td>
      <td>45.30</td>
      <td>1.0</td>
      <td>45.300000</td>
    </tr>
    <tr>
      <th>55</th>
      <th>20</th>
      <td>1</td>
      <td>1.046599e-07</td>
      <td>40.30</td>
      <td>1.0</td>
      <td>40.300000</td>
    </tr>
    <tr>
      <th>197</th>
      <th>1</th>
      <td>1</td>
      <td>1.046599e-07</td>
      <td>39.40</td>
      <td>1.0</td>
      <td>39.400000</td>
    </tr>
    <tr>
      <th>130</th>
      <th>1</th>
      <td>3</td>
      <td>3.139798e-07</td>
      <td>117.85</td>
      <td>3.0</td>
      <td>39.283333</td>
    </tr>
    <tr>
      <th>116</th>
      <th>29</th>
      <td>1</td>
      <td>1.046599e-07</td>
      <td>39.20</td>
      <td>1.0</td>
      <td>39.200000</td>
    </tr>
    <tr>
      <th>21</th>
      <th>265</th>
      <td>3</td>
      <td>3.139798e-07</td>
      <td>117.59</td>
      <td>3.0</td>
      <td>39.196667</td>
    </tr>
    <tr>
      <th>1</th>
      <th>132</th>
      <td>3</td>
      <td>3.139798e-07</td>
      <td>116.62</td>
      <td>3.0</td>
      <td>38.873333</td>
    </tr>
    <tr>
      <th>115</th>
      <th>265</th>
      <td>1</td>
      <td>1.046599e-07</td>
      <td>38.40</td>
      <td>1.0</td>
      <td>38.400000</td>
    </tr>
  </tbody>
</table>
</div>



```python
# بررسی آماری سرعت پس از کنترل اولیه فاصله و مدت سفر
valid_speed_parts = []

for filename in MONTHLY_FILES:

    df = pd.read_parquet(
        PROCESSED_DIR / filename,
        columns=["trip_distance", "duration_minutes", "speed_mph"]
    )

    valid_mask = (
        df["trip_distance"].gt(MIN_VALID_DISTANCE)
        & df["trip_distance"].le(MAX_VALID_DISTANCE)
        & df["duration_minutes"].gt(0)
        & df["speed_mph"].notna()
        & df["speed_mph"].ge(0)
    )

    valid_speed_parts.append(df.loc[valid_mask, "speed_mph"])

    del df

all_valid_speeds = pd.concat(valid_speed_parts, ignore_index=True)

display(
    all_valid_speeds.describe(
        percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )
)
```


    count    9.338392e+06
    mean     1.182552e+01
    std      7.502213e+01
    min      4.168934e-04
    25%      7.256831e+00
    50%      9.582255e+00
    75%      1.303536e+01
    90%      1.948793e+01
    95%      2.507321e+01
    99%      3.586634e+01
    max      5.043600e+04
    Name: speed_mph, dtype: float64



```python
# شمارش سرعت‌های بالا
print("Number of negative speeds:",
      (all_valid_speeds < 0).sum())

print("\nSpeed thresholds:")
for threshold in [50, 80, 100, 150, 200, 500]:
    count = (all_valid_speeds > threshold).sum()
    percentage = count / len(all_valid_speeds) * 100

    print(
        f"Speed > {threshold} mph: "
        f"{count:,} ({percentage:.5f}%)"
    )
```

    Number of negative speeds: 0
    
    Speed thresholds:
    Speed > 50 mph: 7,381 (0.07904%)
    Speed > 80 mph: 3,131 (0.03353%)
    Speed > 100 mph: 2,916 (0.03123%)
    Speed > 150 mph: 2,550 (0.02731%)
    Speed > 200 mph: 2,288 (0.02450%)
    Speed > 500 mph: 1,529 (0.01637%)
    


```python
# مشاهده ۳۰ سرعت بالای ثبت‌شده
high_speed_parts = []

for filename in MONTHLY_FILES:

    df = pd.read_parquet(
        PROCESSED_DIR / filename,
        columns=[
            "PULocationID",
            "DOLocationID",
            "trip_distance",
            "duration_minutes",
            "speed_mph"
        ]
    )

    valid_mask = (
        df["trip_distance"].gt(0)
        & df["trip_distance"].le(MAX_VALID_DISTANCE)
        & df["duration_minutes"].gt(0)
        & df["speed_mph"].notna()
        & df["speed_mph"].ge(0)
    )

    high_speed = df.loc[
        valid_mask & df["speed_mph"].gt(100)
    ].copy()

    high_speed["source_file"] = filename

    high_speed_parts.append(high_speed)

    del df

high_speed_df = pd.concat(
    high_speed_parts,
    ignore_index=True
)

print("Number of trips with speed > 100 mph:",
      len(high_speed_df))

display(
    high_speed_df
    .sort_values("speed_mph", ascending=False)
    .head(30)
)
```

    Number of trips with speed > 100 mph: 2916
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PULocationID</th>
      <th>DOLocationID</th>
      <th>trip_distance</th>
      <th>duration_minutes</th>
      <th>speed_mph</th>
      <th>source_file</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2881</th>
      <td>265</td>
      <td>193</td>
      <td>14.01</td>
      <td>0.016667</td>
      <td>50436.000000</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
    <tr>
      <th>1888</th>
      <td>93</td>
      <td>93</td>
      <td>13.50</td>
      <td>0.016667</td>
      <td>48600.000000</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
    <tr>
      <th>988</th>
      <td>265</td>
      <td>39</td>
      <td>13.22</td>
      <td>0.016667</td>
      <td>47592.000000</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>2876</th>
      <td>265</td>
      <td>216</td>
      <td>12.90</td>
      <td>0.016667</td>
      <td>46440.000000</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
    <tr>
      <th>19</th>
      <td>138</td>
      <td>138</td>
      <td>11.60</td>
      <td>0.016667</td>
      <td>41760.000000</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>2879</th>
      <td>265</td>
      <td>14</td>
      <td>11.47</td>
      <td>0.016667</td>
      <td>41292.000000</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
    <tr>
      <th>461</th>
      <td>132</td>
      <td>132</td>
      <td>10.70</td>
      <td>0.016667</td>
      <td>38520.000000</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>2850</th>
      <td>265</td>
      <td>254</td>
      <td>20.65</td>
      <td>0.033333</td>
      <td>37170.000000</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
    <tr>
      <th>167</th>
      <td>263</td>
      <td>264</td>
      <td>19.62</td>
      <td>0.033333</td>
      <td>35316.000000</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>306</th>
      <td>132</td>
      <td>132</td>
      <td>19.50</td>
      <td>0.033333</td>
      <td>35100.000000</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>364</th>
      <td>132</td>
      <td>264</td>
      <td>19.32</td>
      <td>0.033333</td>
      <td>34776.000000</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>993</th>
      <td>265</td>
      <td>72</td>
      <td>17.40</td>
      <td>0.033333</td>
      <td>31320.000000</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>1479</th>
      <td>138</td>
      <td>138</td>
      <td>59.81</td>
      <td>0.116667</td>
      <td>30759.428571</td>
      <td>trip_features_2024-02.parquet</td>
    </tr>
    <tr>
      <th>1421</th>
      <td>132</td>
      <td>132</td>
      <td>22.48</td>
      <td>0.050000</td>
      <td>26976.000000</td>
      <td>trip_features_2024-02.parquet</td>
    </tr>
    <tr>
      <th>2846</th>
      <td>265</td>
      <td>203</td>
      <td>21.00</td>
      <td>0.050000</td>
      <td>25200.000000</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
    <tr>
      <th>2461</th>
      <td>83</td>
      <td>83</td>
      <td>34.40</td>
      <td>0.083333</td>
      <td>24768.000000</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
    <tr>
      <th>425</th>
      <td>132</td>
      <td>132</td>
      <td>13.60</td>
      <td>0.033333</td>
      <td>24480.000000</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>2777</th>
      <td>229</td>
      <td>229</td>
      <td>19.60</td>
      <td>0.050000</td>
      <td>23520.000000</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
    <tr>
      <th>171</th>
      <td>145</td>
      <td>145</td>
      <td>18.30</td>
      <td>0.050000</td>
      <td>21960.000000</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>970</th>
      <td>265</td>
      <td>61</td>
      <td>6.06</td>
      <td>0.016667</td>
      <td>21816.000000</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>581</th>
      <td>132</td>
      <td>132</td>
      <td>17.50</td>
      <td>0.050000</td>
      <td>21000.000000</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>2581</th>
      <td>132</td>
      <td>132</td>
      <td>17.30</td>
      <td>0.050000</td>
      <td>20760.000000</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
    <tr>
      <th>256</th>
      <td>231</td>
      <td>231</td>
      <td>28.50</td>
      <td>0.083333</td>
      <td>20520.000000</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>140</th>
      <td>163</td>
      <td>163</td>
      <td>11.10</td>
      <td>0.033333</td>
      <td>19980.000000</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>539</th>
      <td>79</td>
      <td>79</td>
      <td>16.60</td>
      <td>0.050000</td>
      <td>19920.000000</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>2278</th>
      <td>138</td>
      <td>138</td>
      <td>10.60</td>
      <td>0.033333</td>
      <td>19080.000000</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
    <tr>
      <th>2260</th>
      <td>230</td>
      <td>230</td>
      <td>10.50</td>
      <td>0.033333</td>
      <td>18900.000000</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
    <tr>
      <th>1942</th>
      <td>138</td>
      <td>138</td>
      <td>10.10</td>
      <td>0.033333</td>
      <td>18180.000000</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
    <tr>
      <th>1842</th>
      <td>68</td>
      <td>68</td>
      <td>5.00</td>
      <td>0.016667</td>
      <td>18000.000000</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
    <tr>
      <th>1590</th>
      <td>249</td>
      <td>249</td>
      <td>15.00</td>
      <td>0.050000</td>
      <td>18000.000000</td>
      <td>trip_features_2024-02.parquet</td>
    </tr>
  </tbody>
</table>
</div>



```python
#آمار مدت سفر و مسافت در سرعت‌های بالا

display(
    high_speed_df[
        ["trip_distance", "duration_minutes", "speed_mph"]
    ].describe(
        percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )
)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trip_distance</th>
      <th>duration_minutes</th>
      <th>speed_mph</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2916.000000</td>
      <td>2916.000000</td>
      <td>2916.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.318772</td>
      <td>0.504630</td>
      <td>1757.296984</td>
    </tr>
    <tr>
      <th>std</th>
      <td>7.699031</td>
      <td>1.749176</td>
      <td>3853.122843</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.030000</td>
      <td>0.016667</td>
      <td>100.327869</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.800000</td>
      <td>0.066667</td>
      <td>225.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.200000</td>
      <td>0.200000</td>
      <td>540.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.600000</td>
      <td>0.433333</td>
      <td>1577.250000</td>
    </tr>
    <tr>
      <th>90%</th>
      <td>16.200000</td>
      <td>0.816667</td>
      <td>4198.285714</td>
    </tr>
    <tr>
      <th>95%</th>
      <td>18.000000</td>
      <td>1.304167</td>
      <td>6980.000000</td>
    </tr>
    <tr>
      <th>99%</th>
      <td>32.460000</td>
      <td>7.000000</td>
      <td>17973.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>92.850000</td>
      <td>44.950000</td>
      <td>50436.000000</td>
    </tr>
  </tbody>
</table>
</div>



```python
#ممیزی مدت سفر
duration_parts = []

for filename in MONTHLY_FILES:

    df = pd.read_parquet(
        PROCESSED_DIR / filename,
        columns=["trip_distance", "duration_minutes"]
    )

    valid_distance = (
        df["trip_distance"].gt(0)
        & df["trip_distance"].le(MAX_VALID_DISTANCE)
    )

    valid_duration = df.loc[
        valid_distance
        & df["duration_minutes"].notna()
        & df["duration_minutes"].gt(0),
        "duration_minutes"
    ]

    duration_parts.append(valid_duration)

    del df

all_valid_durations = pd.concat(
    duration_parts,
    ignore_index=True
)

print("Duration observations:", len(all_valid_durations))

display(
    all_valid_durations.describe(
        percentiles=[
            0.01, 0.05, 0.25, 0.50,
            0.75, 0.90, 0.95, 0.99
        ]
    )
)
```

    Duration observations: 9338392
    


    count    9.338392e+06
    mean     1.623286e+01
    std      3.424779e+01
    min      1.666667e-02
    1%       1.900000e+00
    5%       3.583333e+00
    25%      7.466667e+00
    50%      1.211667e+01
    75%      1.940000e+01
    90%      2.998333e+01
    95%      3.963333e+01
    99%      6.371667e+01
    max      9.455400e+03
    Name: duration_minutes, dtype: float64



```python
#شمارش مدت‌های بسیار کوتاه
print("Duration thresholds:")

for threshold in [0.5, 1, 2, 5, 10]:
    count = (all_valid_durations < threshold).sum()
    percentage = count / len(all_valid_durations) * 100

    print(
        f"Duration < {threshold} minute(s): "
        f"{count:,} ({percentage:.5f}%)"
    )
```

    Duration thresholds:
    Duration < 0.5 minute(s): 20,469 (0.21919%)
    Duration < 1 minute(s): 34,234 (0.36659%)
    Duration < 2 minute(s): 103,805 (1.11159%)
    Duration < 5 minute(s): 1,046,583 (11.20731%)
    Duration < 10 minute(s): 3,669,375 (39.29344%)
    


```python
#شمارش مدت‌های طولانی
print("Long duration thresholds:")

for threshold in [60, 120, 180, 360, 720, 1440]:
    count = (all_valid_durations > threshold).sum()
    percentage = count / len(all_valid_durations) * 100

    print(
        f"Duration > {threshold} minutes: "
        f"{count:,} ({percentage:.5f}%)"
    )
```

    Long duration thresholds:
    Duration > 60 minutes: 120,447 (1.28980%)
    Duration > 120 minutes: 8,693 (0.09309%)
    Duration > 180 minutes: 5,976 (0.06399%)
    Duration > 360 minutes: 5,366 (0.05746%)
    Duration > 720 minutes: 4,859 (0.05203%)
    Duration > 1440 minutes: 40 (0.00043%)
    


```python
#مشاهده ۳۰ سفر با بیشترین مدت
long_duration_parts = []

for filename in MONTHLY_FILES:

    df = pd.read_parquet(
        PROCESSED_DIR / filename,
        columns=[
            "PULocationID",
            "DOLocationID",
            "trip_distance",
            "duration_minutes",
            "speed_mph"
        ]
    )

    valid_distance = (
        df["trip_distance"].gt(0)
        & df["trip_distance"].le(MAX_VALID_DISTANCE)
    )

    long_duration = df.loc[
        valid_distance
        & df["duration_minutes"].gt(180)
    ].copy()

    long_duration["source_file"] = filename

    long_duration_parts.append(long_duration)

    del df

long_duration_df = pd.concat(
    long_duration_parts,
    ignore_index=True
)

print("Trips with duration > 180 minutes:",
      len(long_duration_df))

display(
    long_duration_df
    .sort_values("duration_minutes", ascending=False)
    .head(30)
)
```

    Trips with duration > 180 minutes: 5976
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PULocationID</th>
      <th>DOLocationID</th>
      <th>trip_distance</th>
      <th>duration_minutes</th>
      <th>speed_mph</th>
      <th>source_file</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1486</th>
      <td>237</td>
      <td>170</td>
      <td>2.26</td>
      <td>9455.400000</td>
      <td>0.014341</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>2035</th>
      <td>138</td>
      <td>50</td>
      <td>0.12</td>
      <td>5671.983333</td>
      <td>0.001269</td>
      <td>trip_features_2024-02.parquet</td>
    </tr>
    <tr>
      <th>2371</th>
      <td>141</td>
      <td>237</td>
      <td>0.98</td>
      <td>4692.483333</td>
      <td>0.012531</td>
      <td>trip_features_2024-02.parquet</td>
    </tr>
    <tr>
      <th>2229</th>
      <td>232</td>
      <td>226</td>
      <td>1.57</td>
      <td>4348.116667</td>
      <td>0.021665</td>
      <td>trip_features_2024-02.parquet</td>
    </tr>
    <tr>
      <th>4708</th>
      <td>161</td>
      <td>145</td>
      <td>9.81</td>
      <td>4073.983333</td>
      <td>0.144478</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
    <tr>
      <th>3450</th>
      <td>238</td>
      <td>138</td>
      <td>8.43</td>
      <td>4073.066667</td>
      <td>0.124182</td>
      <td>trip_features_2024-02.parquet</td>
    </tr>
    <tr>
      <th>691</th>
      <td>132</td>
      <td>42</td>
      <td>35.70</td>
      <td>3621.766667</td>
      <td>0.591424</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>2348</th>
      <td>158</td>
      <td>132</td>
      <td>2.62</td>
      <td>3587.183333</td>
      <td>0.043823</td>
      <td>trip_features_2024-02.parquet</td>
    </tr>
    <tr>
      <th>3069</th>
      <td>205</td>
      <td>132</td>
      <td>0.91</td>
      <td>3329.983333</td>
      <td>0.016396</td>
      <td>trip_features_2024-02.parquet</td>
    </tr>
    <tr>
      <th>4440</th>
      <td>132</td>
      <td>216</td>
      <td>0.29</td>
      <td>3222.450000</td>
      <td>0.005400</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
    <tr>
      <th>856</th>
      <td>220</td>
      <td>220</td>
      <td>31.95</td>
      <td>3106.183333</td>
      <td>0.617156</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>1019</th>
      <td>162</td>
      <td>162</td>
      <td>0.20</td>
      <td>2860.883333</td>
      <td>0.004195</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>2462</th>
      <td>132</td>
      <td>265</td>
      <td>3.40</td>
      <td>2790.716667</td>
      <td>0.073100</td>
      <td>trip_features_2024-02.parquet</td>
    </tr>
    <tr>
      <th>2123</th>
      <td>246</td>
      <td>231</td>
      <td>1.92</td>
      <td>2643.866667</td>
      <td>0.043573</td>
      <td>trip_features_2024-02.parquet</td>
    </tr>
    <tr>
      <th>5650</th>
      <td>107</td>
      <td>80</td>
      <td>1.79</td>
      <td>2601.533333</td>
      <td>0.041283</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
    <tr>
      <th>5649</th>
      <td>107</td>
      <td>107</td>
      <td>1.79</td>
      <td>2601.533333</td>
      <td>0.041283</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
    <tr>
      <th>5919</th>
      <td>230</td>
      <td>132</td>
      <td>0.05</td>
      <td>2562.116667</td>
      <td>0.001171</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
    <tr>
      <th>3548</th>
      <td>140</td>
      <td>239</td>
      <td>3.11</td>
      <td>2533.183333</td>
      <td>0.073662</td>
      <td>trip_features_2024-02.parquet</td>
    </tr>
    <tr>
      <th>1551</th>
      <td>140</td>
      <td>133</td>
      <td>4.56</td>
      <td>2438.733333</td>
      <td>0.112189</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>5856</th>
      <td>231</td>
      <td>231</td>
      <td>36.96</td>
      <td>2324.766667</td>
      <td>0.953902</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
    <tr>
      <th>4121</th>
      <td>236</td>
      <td>129</td>
      <td>1.50</td>
      <td>2319.500000</td>
      <td>0.038801</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
    <tr>
      <th>5608</th>
      <td>134</td>
      <td>134</td>
      <td>11.82</td>
      <td>2225.783333</td>
      <td>0.318629</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
    <tr>
      <th>5351</th>
      <td>249</td>
      <td>138</td>
      <td>2.58</td>
      <td>2203.316667</td>
      <td>0.070258</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
    <tr>
      <th>631</th>
      <td>132</td>
      <td>132</td>
      <td>37.97</td>
      <td>2133.333333</td>
      <td>1.067906</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>630</th>
      <td>132</td>
      <td>132</td>
      <td>37.97</td>
      <td>2133.333333</td>
      <td>1.067906</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>2099</th>
      <td>158</td>
      <td>145</td>
      <td>1.05</td>
      <td>2116.116667</td>
      <td>0.029772</td>
      <td>trip_features_2024-02.parquet</td>
    </tr>
    <tr>
      <th>5970</th>
      <td>82</td>
      <td>82</td>
      <td>40.96</td>
      <td>2108.016667</td>
      <td>1.165835</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
    <tr>
      <th>4363</th>
      <td>129</td>
      <td>129</td>
      <td>12.43</td>
      <td>2101.600000</td>
      <td>0.354872</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
    <tr>
      <th>5972</th>
      <td>249</td>
      <td>265</td>
      <td>36.65</td>
      <td>2096.000000</td>
      <td>1.049141</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
    <tr>
      <th>1327</th>
      <td>132</td>
      <td>100</td>
      <td>20.35</td>
      <td>2077.850000</td>
      <td>0.587627</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
  </tbody>
</table>
</div>



```python
# تعریف معیارهای اعتبار برای مهندسی ویژگی‌های مکانی

MIN_VALID_DISTANCE = 0
MAX_VALID_DISTANCE = 100

MIN_VALID_DURATION = 0
MAX_VALID_DURATION = 180

MIN_VALID_SPEED = 0
MAX_VALID_SPEED = 100


print("Validity criteria defined successfully.")
```

    Validity criteria defined successfully.
    


```python
# Final OD feature engineering

OD_COLUMNS = [
    "PULocationID",
    "DOLocationID",
    "trip_distance",
    "duration_minutes",
    "speed_mph"
]

od_final_parts = []

print("OD validity criteria:")
print("Distance: 0 < distance <= 100 miles")
print("Duration: 0 < duration <= 180 minutes")
print("Speed: 0 < speed <= 100 mph")
```

    OD validity criteria:
    Distance: 0 < distance <= 100 miles
    Duration: 0 < duration <= 180 minutes
    Speed: 0 < speed <= 100 mph
    


```python
#محاسبه ویژگی‌های معتبر برای هر مسیر OD
for filename in MONTHLY_FILES:

    df = pd.read_parquet(
        PROCESSED_DIR / filename,
        columns=OD_COLUMNS
    )

    # Independent validity criteria
    valid_distance = (
        df["trip_distance"].gt(0)
        & df["trip_distance"].le(MAX_VALID_DISTANCE)
    )

    valid_duration = (
        df["duration_minutes"].gt(0)
        & df["duration_minutes"].le(MAX_VALID_DURATION)
    )

    valid_speed = (
        valid_distance
        & valid_duration
        & df["speed_mph"].gt(0)
        & df["speed_mph"].le(MAX_VALID_SPEED)
    )

    # Keep only valid values for each metric
    df["valid_distance_value"] = (
        df["trip_distance"].where(valid_distance)
    )

    df["valid_duration_value"] = (
        df["duration_minutes"].where(valid_duration)
    )

    df["valid_speed_value"] = (
        df["speed_mph"].where(valid_speed)
    )

    # Aggregate by OD pair
    monthly_od = (
        df.groupby(["PULocationID", "DOLocationID"])
        .agg(
            valid_distance_sum=("valid_distance_value", "sum"),
            valid_distance_n=("valid_distance_value", "count"),

            valid_duration_sum=("valid_duration_value", "sum"),
            valid_duration_n=("valid_duration_value", "count"),

            valid_speed_sum=("valid_speed_value", "sum"),
            valid_speed_n=("valid_speed_value", "count")
        )
    )

    od_final_parts.append(monthly_od)

    del df

# Combine monthly aggregates
od_final_summary = (
    pd.concat(od_final_parts)
    .groupby(level=[0, 1])
    .sum()
)

print("Aggregated OD pairs:", len(od_final_summary))
display(od_final_summary.head())
```

    Aggregated OD pairs: 37255
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>valid_distance_sum</th>
      <th>valid_distance_n</th>
      <th>valid_duration_sum</th>
      <th>valid_duration_n</th>
      <th>valid_speed_sum</th>
      <th>valid_speed_n</th>
    </tr>
    <tr>
      <th>PULocationID</th>
      <th>DOLocationID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">1</th>
      <th>1</th>
      <td>431.37</td>
      <td>137</td>
      <td>594.700000</td>
      <td>848</td>
      <td>2287.880122</td>
      <td>115</td>
    </tr>
    <tr>
      <th>6</th>
      <td>37.12</td>
      <td>2</td>
      <td>80.050000</td>
      <td>2</td>
      <td>55.069041</td>
      <td>2</td>
    </tr>
    <tr>
      <th>25</th>
      <td>16.10</td>
      <td>1</td>
      <td>80.933333</td>
      <td>1</td>
      <td>11.935750</td>
      <td>1</td>
    </tr>
    <tr>
      <th>48</th>
      <td>53.90</td>
      <td>3</td>
      <td>230.733333</td>
      <td>4</td>
      <td>74.845193</td>
      <td>3</td>
    </tr>
    <tr>
      <th>50</th>
      <td>17.50</td>
      <td>1</td>
      <td>44.166667</td>
      <td>1</td>
      <td>23.773585</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



```python
#نسخه نهایی ویژگی‌های OD
# Preserve original OD counts and shares
od_features_final = (
    od_features_clean
    .drop(columns=[
        "valid_distance_sum",
        "valid_distance_n",
        "avg_od_distance"
    ])
    .join(od_final_summary, how="left")
)

# Calculate clean averages
od_features_final["avg_od_distance"] = (
    od_features_final["valid_distance_sum"]
    / od_features_final["valid_distance_n"].replace(0, np.nan)
)

od_features_final["avg_duration_minutes"] = (
    od_features_final["valid_duration_sum"]
    / od_features_final["valid_duration_n"].replace(0, np.nan)
)

od_features_final["avg_speed_mph"] = (
    od_features_final["valid_speed_sum"]
    / od_features_final["valid_speed_n"].replace(0, np.nan)
)

print("Final OD features shape:", od_features_final.shape)
display(od_features_final.head())
```

    Final OD features shape: (37255, 11)
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>od_trip_count</th>
      <th>od_trip_share</th>
      <th>valid_distance_sum</th>
      <th>valid_distance_n</th>
      <th>valid_duration_sum</th>
      <th>valid_duration_n</th>
      <th>valid_speed_sum</th>
      <th>valid_speed_n</th>
      <th>avg_od_distance</th>
      <th>avg_duration_minutes</th>
      <th>avg_speed_mph</th>
    </tr>
    <tr>
      <th>PULocationID</th>
      <th>DOLocationID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">1</th>
      <th>1</th>
      <td>849</td>
      <td>8.885627e-05</td>
      <td>431.37</td>
      <td>137</td>
      <td>594.700000</td>
      <td>848</td>
      <td>2287.880122</td>
      <td>115</td>
      <td>3.148686</td>
      <td>0.701297</td>
      <td>19.894610</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2</td>
      <td>2.093198e-07</td>
      <td>37.12</td>
      <td>2</td>
      <td>80.050000</td>
      <td>2</td>
      <td>55.069041</td>
      <td>2</td>
      <td>18.560000</td>
      <td>40.025000</td>
      <td>27.534521</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1</td>
      <td>1.046599e-07</td>
      <td>16.10</td>
      <td>1</td>
      <td>80.933333</td>
      <td>1</td>
      <td>11.935750</td>
      <td>1</td>
      <td>16.100000</td>
      <td>80.933333</td>
      <td>11.935750</td>
    </tr>
    <tr>
      <th>48</th>
      <td>4</td>
      <td>4.186397e-07</td>
      <td>53.90</td>
      <td>3</td>
      <td>230.733333</td>
      <td>4</td>
      <td>74.845193</td>
      <td>3</td>
      <td>17.966667</td>
      <td>57.683333</td>
      <td>24.948398</td>
    </tr>
    <tr>
      <th>50</th>
      <td>1</td>
      <td>1.046599e-07</td>
      <td>17.50</td>
      <td>1</td>
      <td>44.166667</td>
      <td>1</td>
      <td>23.773585</td>
      <td>1</td>
      <td>17.500000</td>
      <td>44.166667</td>
      <td>23.773585</td>
    </tr>
  </tbody>
</table>
</div>



```python
#(نسخه اصلاحی) ویژگی‌های مبدأ و مقصد
# Rebuild clean pickup and dropoff spatial features
# Original zone_df and destination_features remain unchanged.

pickup_parts = []
dropoff_parts = []

columns_to_read = [
    "PULocationID",
    "DOLocationID",
    "trip_distance",
    "duration_minutes",
    "fare_amount",
    "speed_mph"
]

for filename in MONTHLY_FILES:

    df = pd.read_parquet(
        PROCESSED_DIR / filename,
        columns=columns_to_read
    )

    # Independent validity masks
    valid_distance = (
        df["trip_distance"].gt(MIN_VALID_DISTANCE)
        & df["trip_distance"].le(MAX_VALID_DISTANCE)
    )

    valid_duration = (
        df["duration_minutes"].gt(MIN_VALID_DURATION)
        & df["duration_minutes"].le(MAX_VALID_DURATION)
    )

    valid_speed = (
        valid_distance
        & valid_duration
        & df["speed_mph"].gt(MIN_VALID_SPEED)
        & df["speed_mph"].le(MAX_VALID_SPEED)
    )

    # Mask invalid values for feature aggregation
    df["valid_distance_value"] = (
        df["trip_distance"].where(valid_distance)
    )

    df["valid_duration_value"] = (
        df["duration_minutes"].where(valid_duration)
    )

    df["valid_speed_value"] = (
        df["speed_mph"].where(valid_speed)
    )

    # Aggregate pickup-zone features
    pickup_month = (
        df.groupby("PULocationID")
        .agg(
            pickup_trips=("PULocationID", "size"),

            distance_sum=("valid_distance_value", "sum"),
            distance_n=("valid_distance_value", "count"),

            duration_sum=("valid_duration_value", "sum"),
            duration_n=("valid_duration_value", "count"),

            fare_sum=("fare_amount", "sum"),
            fare_n=("fare_amount", "count"),

            speed_sum=("valid_speed_value", "sum"),
            speed_n=("valid_speed_value", "count")
        )
    )

    pickup_parts.append(pickup_month)

    # Aggregate dropoff-zone features
    dropoff_month = (
        df.groupby("DOLocationID")
        .agg(
            dropoff_trips=("DOLocationID", "size"),

            distance_sum=("valid_distance_value", "sum"),
            distance_n=("valid_distance_value", "count"),

            duration_sum=("valid_duration_value", "sum"),
            duration_n=("valid_duration_value", "count"),

            fare_sum=("fare_amount", "sum"),
            fare_n=("fare_amount", "count"),

            speed_sum=("valid_speed_value", "sum"),
            speed_n=("valid_speed_value", "count")
        )
    )

    dropoff_parts.append(dropoff_month)

    del df


# Combine monthly pickup aggregates
zone_features_clean = (
    pd.concat(pickup_parts)
    .groupby(level=0)
    .sum()
)

zone_features_clean.index.name = "PULocationID"

# Combine monthly dropoff aggregates
destination_features_clean = (
    pd.concat(dropoff_parts)
    .groupby(level=0)
    .sum()
)

destination_features_clean.index.name = "DOLocationID"

print("Pickup features:", zone_features_clean.shape)
print("Dropoff features:", destination_features_clean.shape)
```

    Pickup features: (262, 9)
    Dropoff features: (261, 9)
    


```python
#محاسبه میانگین‌های اصلاح‌شده
# Calculate clean pickup averages

zone_features_clean["avg_trip_distance"] = (
    zone_features_clean["distance_sum"]
    / zone_features_clean["distance_n"].replace(0, np.nan)
)

zone_features_clean["avg_duration_minutes"] = (
    zone_features_clean["duration_sum"]
    / zone_features_clean["duration_n"].replace(0, np.nan)
)

zone_features_clean["avg_fare_amount"] = (
    zone_features_clean["fare_sum"]
    / zone_features_clean["fare_n"].replace(0, np.nan)
)

zone_features_clean["avg_speed_mph"] = (
    zone_features_clean["speed_sum"]
    / zone_features_clean["speed_n"].replace(0, np.nan)
)

zone_features_clean["is_unknown_zone"] = (
    zone_features_clean.index.isin([264, 265])
)


# Calculate clean dropoff averages

destination_features_clean["avg_trip_distance"] = (
    destination_features_clean["distance_sum"]
    / destination_features_clean["distance_n"].replace(0, np.nan)
)

destination_features_clean["avg_duration_minutes"] = (
    destination_features_clean["duration_sum"]
    / destination_features_clean["duration_n"].replace(0, np.nan)
)

destination_features_clean["avg_fare_amount"] = (
    destination_features_clean["fare_sum"]
    / destination_features_clean["fare_n"].replace(0, np.nan)
)

destination_features_clean["avg_speed_mph"] = (
    destination_features_clean["speed_sum"]
    / destination_features_clean["speed_n"].replace(0, np.nan)
)

destination_features_clean["is_unknown_zone"] = (
    destination_features_clean.index.isin([264, 265])
)

print("Clean pickup features created.")
print("Clean dropoff features created.")
```

    Clean pickup features created.
    Clean dropoff features created.
    


```python
#اعتبارسنجی اولیه
print("=" * 60)
print("PICKUP FEATURES - CLEAN")
print("=" * 60)

print("Shape:", zone_features_clean.shape)
print("Total pickup trips:",
      zone_features_clean["pickup_trips"].sum())

print("\nMissing values:")
display(zone_features_clean.isna().sum())

print("\nClean pickup averages:")
display(
    zone_features_clean[
        [
            "avg_trip_distance",
            "avg_duration_minutes",
            "avg_speed_mph"
        ]
    ].describe()
)


print("\n" + "=" * 60)
print("DROPOFF FEATURES - CLEAN")
print("=" * 60)

print("Shape:", destination_features_clean.shape)
print("Total dropoff trips:",
      destination_features_clean["dropoff_trips"].sum())

print("\nMissing values:")
display(destination_features_clean.isna().sum())

print("\nClean dropoff averages:")
display(
    destination_features_clean[
        [
            "avg_trip_distance",
            "avg_duration_minutes",
            "avg_speed_mph"
        ]
    ].describe()
)
```

    ============================================================
    PICKUP FEATURES - CLEAN
    ============================================================
    Shape: (262, 14)
    Total pickup trips: 9554756
    
    Missing values:
    


    pickup_trips            0
    distance_sum            0
    distance_n              0
    duration_sum            0
    duration_n              0
    fare_sum                0
    fare_n                  0
    speed_sum               0
    speed_n                 0
    avg_trip_distance       1
    avg_duration_minutes    0
    avg_fare_amount         0
    avg_speed_mph           1
    is_unknown_zone         0
    dtype: int64


    
    Clean pickup averages:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>avg_trip_distance</th>
      <th>avg_duration_minutes</th>
      <th>avg_speed_mph</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>261.000000</td>
      <td>262.000000</td>
      <td>261.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6.821635</td>
      <td>26.667834</td>
      <td>15.145575</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4.009842</td>
      <td>11.310653</td>
      <td>6.281963</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.410000</td>
      <td>0.166667</td>
      <td>7.788144</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4.421406</td>
      <td>16.347125</td>
      <td>11.125713</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>6.427698</td>
      <td>26.295306</td>
      <td>13.280730</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8.552299</td>
      <td>35.716903</td>
      <td>16.790872</td>
    </tr>
    <tr>
      <th>max</th>
      <td>39.455000</td>
      <td>61.042554</td>
      <td>58.695652</td>
    </tr>
  </tbody>
</table>
</div>


    
    ============================================================
    DROPOFF FEATURES - CLEAN
    ============================================================
    Shape: (261, 14)
    Total dropoff trips: 9554756
    
    Missing values:
    


    dropoff_trips           0
    distance_sum            0
    distance_n              0
    duration_sum            0
    duration_n              0
    fare_sum                0
    fare_n                  0
    speed_sum               0
    speed_n                 0
    avg_trip_distance       0
    avg_duration_minutes    0
    avg_fare_amount         0
    avg_speed_mph           0
    is_unknown_zone         0
    dtype: int64


    
    Clean dropoff averages:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>avg_trip_distance</th>
      <th>avg_duration_minutes</th>
      <th>avg_speed_mph</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>261.000000</td>
      <td>261.000000</td>
      <td>261.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>9.192803</td>
      <td>27.590682</td>
      <td>19.440815</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5.463669</td>
      <td>9.828094</td>
      <td>6.578907</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.767123</td>
      <td>5.124270</td>
      <td>8.569588</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.780000</td>
      <td>20.048572</td>
      <td>14.820560</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>8.466073</td>
      <td>27.714870</td>
      <td>19.532025</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>11.986515</td>
      <td>34.832486</td>
      <td>22.904704</td>
    </tr>
    <tr>
      <th>max</th>
      <td>31.048235</td>
      <td>55.131982</td>
      <td>38.278348</td>
    </tr>
  </tbody>
</table>
</div>



```python
#اعتبارسنجی ویژگی‌های مسیرهای مبدأ–مقصد 
print("Original OD pairs:", len(od_features))
print("Final OD pairs:", len(od_features_final))

print("\nTotal OD trips:")
print("Original:", od_features["od_trip_count"].sum())
print("Final:", od_features_final["od_trip_count"].sum())

print("\nNumber of OD pairs without valid observations:")

for col in [
    "avg_od_distance",
    "avg_duration_minutes",
    "avg_speed_mph"
]:
    print(
        col,
        od_features_final[col].isna().sum()
    )

print("\nFinal OD feature statistics:")

display(
    od_features_final[
        [
            "avg_od_distance",
            "avg_duration_minutes",
            "avg_speed_mph"
        ]
    ].describe()
)

print("\nFinal columns:")
print(od_features_final.columns.tolist())
```

    Original OD pairs: 37255
    Final OD pairs: 37255
    
    Total OD trips:
    Original: 9554756
    Final: 9554756
    
    Number of OD pairs without valid observations:
    avg_od_distance 769
    avg_duration_minutes 54
    avg_speed_mph 843
    
    Final OD feature statistics:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>avg_od_distance</th>
      <th>avg_duration_minutes</th>
      <th>avg_speed_mph</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>36486.000000</td>
      <td>37201.000000</td>
      <td>36412.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>9.639003</td>
      <td>38.401601</td>
      <td>16.744940</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5.657659</td>
      <td>22.142245</td>
      <td>7.635415</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.010000</td>
      <td>0.033333</td>
      <td>0.070175</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.256038</td>
      <td>22.883333</td>
      <td>11.242729</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>9.011013</td>
      <td>33.516667</td>
      <td>15.187913</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>13.254750</td>
      <td>49.398611</td>
      <td>20.739678</td>
    </tr>
    <tr>
      <th>max</th>
      <td>72.540000</td>
      <td>177.266667</td>
      <td>96.000000</td>
    </tr>
  </tbody>
</table>
</div>


    
    Final columns:
    ['od_trip_count', 'od_trip_share', 'valid_distance_sum', 'valid_distance_n', 'valid_duration_sum', 'valid_duration_n', 'valid_speed_sum', 'valid_speed_n', 'avg_od_distance', 'avg_duration_minutes', 'avg_speed_mph']
    


```python
# اعتبارسنجی ویژگی‌های مکانی و تقاضای ساعتی مناطق

print("=" * 65)
print("1. ZONE FEATURES")
print("=" * 65)

print("Number of zones:", zone_df["PULocationID"].nunique())
print("Duplicate zone IDs:", zone_df["PULocationID"].duplicated().sum())

print("\nUnknown zones:")
print(zone_df["is_unknown_zone"].value_counts(dropna=False))

print("\nTotal pickup trips in zone_features:")
print(zone_df["pickup_trips"].sum())


print("\n" + "=" * 65)
print("2. HOURLY ZONE DEMAND")
print("=" * 65)

print("Number of rows:", len(hourly_zone_df))
print("Number of unique timestamps:",
      hourly_zone_df["pickup_hour"].nunique())

print("Number of unique zones:",
      hourly_zone_df["PULocationID"].nunique())

print("Duplicate (hour, zone) combinations:",
      hourly_zone_df.duplicated(
          subset=["pickup_hour", "PULocationID"]
      ).sum())

print("\nTime range:")
print("First:", hourly_zone_df["pickup_hour"].min())
print("Last:", hourly_zone_df["pickup_hour"].max())

print("\nTotal trips in hourly_zone_demand:")
print(hourly_zone_df["trip_count"].sum())


print("\n" + "=" * 65)
print("3. COMPARISON WITH HOURLY TOTAL DEMAND")
print("=" * 65)

hourly_total_df = pd.read_parquet(
    PROCESSED_DIR / "hourly_trip_counts.parquet"
)

print("Total trips in hourly_trip_counts:",
      hourly_total_df["trip_count"].sum())

print("Difference between the two totals:",
      hourly_total_df["trip_count"].sum()
      - hourly_zone_df["trip_count"].sum())


print("\n" + "=" * 65)
print("4. HOURLY ZONE COVERAGE")
print("=" * 65)

zones_per_hour = (
    hourly_zone_df.groupby("pickup_hour")["PULocationID"]
    .nunique()
)

print("Minimum zones per hour:", zones_per_hour.min())
print("Maximum zones per hour:", zones_per_hour.max())
print("Average zones per hour:", zones_per_hour.mean())
```

    =================================================================
    1. ZONE FEATURES
    =================================================================
    Number of zones: 262
    Duplicate zone IDs: 0
    
    Unknown zones:
    is_unknown_zone
    False    260
    True       2
    Name: count, dtype: int64
    
    Total pickup trips in zone_features:
    9554756
    
    =================================================================
    2. HOURLY ZONE DEMAND
    =================================================================
    Number of rows: 240917
    Number of unique timestamps: 2183
    Number of unique zones: 262
    Duplicate (hour, zone) combinations: 0
    
    Time range:
    First: 2024-01-01 00:00:00-05:00
    Last: 2024-03-31 23:00:00-04:00
    
    Total trips in hourly_zone_demand:
    9554756
    
    =================================================================
    3. COMPARISON WITH HOURLY TOTAL DEMAND
    =================================================================
    Total trips in hourly_trip_counts: 9554756
    Difference between the two totals: 0
    
    =================================================================
    4. HOURLY ZONE COVERAGE
    =================================================================
    Minimum zones per hour: 43
    Maximum zones per hour: 175
    Average zones per hour: 110.36051305542831
    


```python
# Final validation of clean spatial feature tables

print("=" * 70)
print("FINAL SPATIAL FEATURE VALIDATION")
print("=" * 70)

# Compare total trip counts across datasets
total_summary = pd.DataFrame([
    {
        "dataset": "Hourly trip counts",
        "total_trips": hourly_total_df["trip_count"].sum()
    },
    {
        "dataset": "Hourly zone demand",
        "total_trips": hourly_zone_df["trip_count"].sum()
    },
    {
        "dataset": "Clean pickup features",
        "total_trips": zone_features_clean["pickup_trips"].sum()
    },
    {
        "dataset": "Clean dropoff features",
        "total_trips": destination_features_clean["dropoff_trips"].sum()
    },
    {
        "dataset": "Final OD features",
        "total_trips": od_features_final["od_trip_count"].sum()
    }
])

display(total_summary)

print("\nTotal consistency:")
print(
    "All totals equal:",
    total_summary["total_trips"].nunique() == 1
)

# Check unique identifiers
print("\nUnique identifier checks:")

print(
    "Pickup zone IDs unique:",
    zone_features_clean.index.is_unique
)

print(
    "Dropoff zone IDs unique:",
    destination_features_clean.index.is_unique
)

print(
    "OD pairs unique:",
    od_features_final.index.is_unique
)

# Check missing values in clean averages
print("\nMissing values in clean averages:")

for name, df in [
    ("Pickup", zone_features_clean),
    ("Dropoff", destination_features_clean),
    ("OD", od_features_final)
]:
    print(f"\n{name}:")
    display(
        df[
            [
                col for col in [
                    "avg_trip_distance",
                    "avg_duration_minutes",
                    "avg_speed_mph"
                ]
                if col in df.columns
            ]
        ].isna().sum()
    )

# Check invalid average values
print("\nInvalid average values:")

for name, df in [
    ("Pickup", zone_features_clean),
    ("Dropoff", destination_features_clean),
    ("OD", od_features_final)
]:
    print(f"\n{name}:")

    for col, lower, upper in [
        ("avg_trip_distance", 0, 100),
        ("avg_duration_minutes", 0, 180),
        ("avg_speed_mph", 0, 100)
    ]:
        if col in df.columns:
            invalid = (
                df[col].notna()
                & (
                    df[col].le(lower)
                    | df[col].gt(upper)
                )
            ).sum()

            print(f"{col}: {invalid}")
```

    ======================================================================
    FINAL SPATIAL FEATURE VALIDATION
    ======================================================================
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dataset</th>
      <th>total_trips</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Hourly trip counts</td>
      <td>9554756</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Hourly zone demand</td>
      <td>9554756</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Clean pickup features</td>
      <td>9554756</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Clean dropoff features</td>
      <td>9554756</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Final OD features</td>
      <td>9554756</td>
    </tr>
  </tbody>
</table>
</div>


    
    Total consistency:
    All totals equal: True
    
    Unique identifier checks:
    Pickup zone IDs unique: True
    Dropoff zone IDs unique: True
    OD pairs unique: True
    
    Missing values in clean averages:
    
    Pickup:
    


    avg_trip_distance       1
    avg_duration_minutes    0
    avg_speed_mph           1
    dtype: int64


    
    Dropoff:
    


    avg_trip_distance       0
    avg_duration_minutes    0
    avg_speed_mph           0
    dtype: int64


    
    OD:
    


    avg_duration_minutes     54
    avg_speed_mph           843
    dtype: int64


    
    Invalid average values:
    
    Pickup:
    avg_trip_distance: 0
    avg_duration_minutes: 0
    avg_speed_mph: 0
    
    Dropoff:
    avg_trip_distance: 0
    avg_duration_minutes: 0
    avg_speed_mph: 0
    
    OD:
    avg_duration_minutes: 0
    avg_speed_mph: 0
    


```python
# Build integrated zone-level feature table

# Select pickup features
pickup_selected = zone_features_clean[
    [
        "pickup_trips",
        "distance_n",
        "duration_n",
        "speed_n",
        "avg_trip_distance",
        "avg_duration_minutes",
        "avg_fare_amount",
        "avg_speed_mph"
    ]
].add_prefix("pickup_")

pickup_selected.index.name = "LocationID"


# Select dropoff features
dropoff_selected = destination_features_clean[
    [
        "dropoff_trips",
        "distance_n",
        "duration_n",
        "speed_n",
        "avg_trip_distance",
        "avg_duration_minutes",
        "avg_fare_amount",
        "avg_speed_mph"
    ]
].add_prefix("dropoff_")

dropoff_selected.index.name = "LocationID"


# Combine pickup and dropoff features
zone_analysis_df = pickup_selected.join(
    dropoff_selected,
    how="outer"
)

# Fill missing trip counts and observation counts with zero
count_columns = [
    col for col in zone_analysis_df.columns
    if col.endswith((
        "pickup_trips",
        "dropoff_trips",
        "distance_n",
        "duration_n",
        "speed_n"
    ))
]

zone_analysis_df[count_columns] = (
    zone_analysis_df[count_columns].fillna(0)
)

# Identify unknown TLC zones
zone_analysis_df["is_unknown_zone"] = (
    zone_analysis_df.index.isin([264, 265])
)

zone_analysis_df = zone_analysis_df.reset_index()

print("Zone analysis dataset created.")
print("Shape:", zone_analysis_df.shape)

print("\nColumns:")
print(zone_analysis_df.columns.tolist())

print("\nFirst 10 rows:")
display(zone_analysis_df.head(10))

print("\nMissing values:")
display(zone_analysis_df.isna().sum())
```

    Zone analysis dataset created.
    Shape: (262, 18)
    
    Columns:
    ['LocationID', 'pickup_pickup_trips', 'pickup_distance_n', 'pickup_duration_n', 'pickup_speed_n', 'pickup_avg_trip_distance', 'pickup_avg_duration_minutes', 'pickup_avg_fare_amount', 'pickup_avg_speed_mph', 'dropoff_dropoff_trips', 'dropoff_distance_n', 'dropoff_duration_n', 'dropoff_speed_n', 'dropoff_avg_trip_distance', 'dropoff_avg_duration_minutes', 'dropoff_avg_fare_amount', 'dropoff_avg_speed_mph', 'is_unknown_zone']
    
    First 10 rows:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LocationID</th>
      <th>pickup_pickup_trips</th>
      <th>pickup_distance_n</th>
      <th>pickup_duration_n</th>
      <th>pickup_speed_n</th>
      <th>pickup_avg_trip_distance</th>
      <th>pickup_avg_duration_minutes</th>
      <th>pickup_avg_fare_amount</th>
      <th>pickup_avg_speed_mph</th>
      <th>dropoff_dropoff_trips</th>
      <th>dropoff_distance_n</th>
      <th>dropoff_duration_n</th>
      <th>dropoff_speed_n</th>
      <th>dropoff_avg_trip_distance</th>
      <th>dropoff_avg_duration_minutes</th>
      <th>dropoff_avg_fare_amount</th>
      <th>dropoff_avg_speed_mph</th>
      <th>is_unknown_zone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>954</td>
      <td>194</td>
      <td>953</td>
      <td>171</td>
      <td>6.291392</td>
      <td>2.623382</td>
      <td>83.524591</td>
      <td>22.287090</td>
      <td>22950.0</td>
      <td>22137.0</td>
      <td>22932.0</td>
      <td>22098.0</td>
      <td>18.165595</td>
      <td>37.358979</td>
      <td>90.299463</td>
      <td>29.968444</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>14</td>
      <td>13</td>
      <td>14</td>
      <td>13</td>
      <td>10.858462</td>
      <td>27.240476</td>
      <td>38.853571</td>
      <td>25.514643</td>
      <td>17.0</td>
      <td>16.0</td>
      <td>17.0</td>
      <td>16.0</td>
      <td>10.305000</td>
      <td>31.914706</td>
      <td>34.911765</td>
      <td>21.701508</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>350</td>
      <td>282</td>
      <td>349</td>
      <td>279</td>
      <td>9.439291</td>
      <td>39.942216</td>
      <td>36.439114</td>
      <td>13.778782</td>
      <td>763.0</td>
      <td>696.0</td>
      <td>759.0</td>
      <td>691.0</td>
      <td>13.397759</td>
      <td>34.943325</td>
      <td>46.783368</td>
      <td>24.339169</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>15879</td>
      <td>14761</td>
      <td>15861</td>
      <td>14744</td>
      <td>2.853294</td>
      <td>14.363545</td>
      <td>16.816984</td>
      <td>11.437372</td>
      <td>39888.0</td>
      <td>38678.0</td>
      <td>39857.0</td>
      <td>38646.0</td>
      <td>2.854780</td>
      <td>14.356796</td>
      <td>16.409995</td>
      <td>11.181948</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>0.166667</td>
      <td>100.000000</td>
      <td>NaN</td>
      <td>33.0</td>
      <td>32.0</td>
      <td>33.0</td>
      <td>32.0</td>
      <td>30.414687</td>
      <td>53.261111</td>
      <td>108.254545</td>
      <td>35.872295</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>69</td>
      <td>66</td>
      <td>68</td>
      <td>65</td>
      <td>5.952879</td>
      <td>12.577941</td>
      <td>20.669710</td>
      <td>34.185438</td>
      <td>200.0</td>
      <td>196.0</td>
      <td>197.0</td>
      <td>193.0</td>
      <td>13.272041</td>
      <td>27.365736</td>
      <td>48.168050</td>
      <td>33.539751</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>5823</td>
      <td>5028</td>
      <td>5811</td>
      <td>5014</td>
      <td>3.994344</td>
      <td>17.355117</td>
      <td>20.474699</td>
      <td>12.779749</td>
      <td>23412.0</td>
      <td>22668.0</td>
      <td>23387.0</td>
      <td>22634.0</td>
      <td>5.464754</td>
      <td>21.802658</td>
      <td>25.948205</td>
      <td>14.820560</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>23</td>
      <td>22</td>
      <td>23</td>
      <td>22</td>
      <td>8.118636</td>
      <td>23.240580</td>
      <td>50.896087</td>
      <td>21.424167</td>
      <td>118.0</td>
      <td>115.0</td>
      <td>118.0</td>
      <td>115.0</td>
      <td>6.897043</td>
      <td>21.379944</td>
      <td>28.838475</td>
      <td>23.516539</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>170</td>
      <td>148</td>
      <td>170</td>
      <td>147</td>
      <td>6.928649</td>
      <td>27.431569</td>
      <td>30.052176</td>
      <td>15.848142</td>
      <td>854.0</td>
      <td>827.0</td>
      <td>854.0</td>
      <td>827.0</td>
      <td>10.116360</td>
      <td>26.584719</td>
      <td>39.523478</td>
      <td>24.409763</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>3290</td>
      <td>2992</td>
      <td>3280</td>
      <td>2980</td>
      <td>11.987794</td>
      <td>34.662048</td>
      <td>53.211112</td>
      <td>21.659658</td>
      <td>8177.0</td>
      <td>7883.0</td>
      <td>8166.0</td>
      <td>7866.0</td>
      <td>5.164724</td>
      <td>16.371990</td>
      <td>22.499980</td>
      <td>22.347565</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>


    
    Missing values:
    


    LocationID                      0
    pickup_pickup_trips             0
    pickup_distance_n               0
    pickup_duration_n               0
    pickup_speed_n                  0
    pickup_avg_trip_distance        1
    pickup_avg_duration_minutes     0
    pickup_avg_fare_amount          0
    pickup_avg_speed_mph            1
    dropoff_dropoff_trips           0
    dropoff_distance_n              0
    dropoff_duration_n              0
    dropoff_speed_n                 0
    dropoff_avg_trip_distance       1
    dropoff_avg_duration_minutes    1
    dropoff_avg_fare_amount         1
    dropoff_avg_speed_mph           1
    is_unknown_zone                 0
    dtype: int64



```python
# Build hourly demand profile for each zone

# Create hourly zone demand matrix
hourly_zone_matrix = hourly_zone_df.pivot_table(
    index="pickup_hour",
    columns="PULocationID",
    values="trip_count",
    aggfunc="sum"
)

# Include all observed hours, including zero-demand hours
all_hours = pd.DatetimeIndex(
    hourly_df["pickup_hour"].drop_duplicates().sort_values()
)

hourly_zone_matrix = (
    hourly_zone_matrix
    .reindex(index=all_hours, fill_value=0)
    .fillna(0)
)

print("Hourly zone matrix shape:", hourly_zone_matrix.shape)


# Calculate average demand for each hour of the day
zone_hour_profile = (
    hourly_zone_matrix
    .groupby(hourly_zone_matrix.index.hour)
    .mean()
    .T
)

# Rename hourly profile columns
zone_hour_profile.columns = [
    f"mean_demand_hour_{int(hour):02d}"
    for hour in zone_hour_profile.columns
]

zone_hour_profile.index.name = "LocationID"


# Merge hourly profiles with zone-level features
zone_analysis_df = zone_analysis_df.merge(
    zone_hour_profile.reset_index(),
    on="LocationID",
    how="left",
    validate="one_to_one"
)

print("\nFinal zone analysis dataset:")
print("Shape:", zone_analysis_df.shape)

print("\nHourly profile columns:")
print([
    col for col in zone_analysis_df.columns
    if col.startswith("mean_demand_hour_")
])

print("\nFirst 10 rows:")
display(zone_analysis_df.head(10))
```

    Hourly zone matrix shape: (2183, 262)
    
    Final zone analysis dataset:
    Shape: (262, 42)
    
    Hourly profile columns:
    ['mean_demand_hour_00', 'mean_demand_hour_01', 'mean_demand_hour_02', 'mean_demand_hour_03', 'mean_demand_hour_04', 'mean_demand_hour_05', 'mean_demand_hour_06', 'mean_demand_hour_07', 'mean_demand_hour_08', 'mean_demand_hour_09', 'mean_demand_hour_10', 'mean_demand_hour_11', 'mean_demand_hour_12', 'mean_demand_hour_13', 'mean_demand_hour_14', 'mean_demand_hour_15', 'mean_demand_hour_16', 'mean_demand_hour_17', 'mean_demand_hour_18', 'mean_demand_hour_19', 'mean_demand_hour_20', 'mean_demand_hour_21', 'mean_demand_hour_22', 'mean_demand_hour_23']
    
    First 10 rows:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LocationID</th>
      <th>pickup_pickup_trips</th>
      <th>pickup_distance_n</th>
      <th>pickup_duration_n</th>
      <th>pickup_speed_n</th>
      <th>pickup_avg_trip_distance</th>
      <th>pickup_avg_duration_minutes</th>
      <th>pickup_avg_fare_amount</th>
      <th>pickup_avg_speed_mph</th>
      <th>dropoff_dropoff_trips</th>
      <th>...</th>
      <th>mean_demand_hour_14</th>
      <th>mean_demand_hour_15</th>
      <th>mean_demand_hour_16</th>
      <th>mean_demand_hour_17</th>
      <th>mean_demand_hour_18</th>
      <th>mean_demand_hour_19</th>
      <th>mean_demand_hour_20</th>
      <th>mean_demand_hour_21</th>
      <th>mean_demand_hour_22</th>
      <th>mean_demand_hour_23</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>954</td>
      <td>194</td>
      <td>953</td>
      <td>171</td>
      <td>6.291392</td>
      <td>2.623382</td>
      <td>83.524591</td>
      <td>22.287090</td>
      <td>22950.0</td>
      <td>...</td>
      <td>1.076923</td>
      <td>1.208791</td>
      <td>1.164835</td>
      <td>1.000000</td>
      <td>0.835165</td>
      <td>0.758242</td>
      <td>0.230769</td>
      <td>0.065934</td>
      <td>0.076923</td>
      <td>0.032967</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>14</td>
      <td>13</td>
      <td>14</td>
      <td>13</td>
      <td>10.858462</td>
      <td>27.240476</td>
      <td>38.853571</td>
      <td>25.514643</td>
      <td>17.0</td>
      <td>...</td>
      <td>0.021978</td>
      <td>0.000000</td>
      <td>0.032967</td>
      <td>0.010989</td>
      <td>0.010989</td>
      <td>0.010989</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010989</td>
      <td>0.021978</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>350</td>
      <td>282</td>
      <td>349</td>
      <td>279</td>
      <td>9.439291</td>
      <td>39.942216</td>
      <td>36.439114</td>
      <td>13.778782</td>
      <td>763.0</td>
      <td>...</td>
      <td>0.153846</td>
      <td>0.241758</td>
      <td>0.186813</td>
      <td>0.076923</td>
      <td>0.153846</td>
      <td>0.109890</td>
      <td>0.098901</td>
      <td>0.043956</td>
      <td>0.043956</td>
      <td>0.065934</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>15879</td>
      <td>14761</td>
      <td>15861</td>
      <td>14744</td>
      <td>2.853294</td>
      <td>14.363545</td>
      <td>16.816984</td>
      <td>11.437372</td>
      <td>39888.0</td>
      <td>...</td>
      <td>3.648352</td>
      <td>3.450549</td>
      <td>3.296703</td>
      <td>4.131868</td>
      <td>5.615385</td>
      <td>6.692308</td>
      <td>6.109890</td>
      <td>7.032967</td>
      <td>8.582418</td>
      <td>13.527473</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>0.166667</td>
      <td>100.000000</td>
      <td>NaN</td>
      <td>33.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>69</td>
      <td>66</td>
      <td>68</td>
      <td>65</td>
      <td>5.952879</td>
      <td>12.577941</td>
      <td>20.669710</td>
      <td>34.185438</td>
      <td>200.0</td>
      <td>...</td>
      <td>0.032967</td>
      <td>0.076923</td>
      <td>0.021978</td>
      <td>0.010989</td>
      <td>0.000000</td>
      <td>0.010989</td>
      <td>0.000000</td>
      <td>0.043956</td>
      <td>0.010989</td>
      <td>0.021978</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>5823</td>
      <td>5028</td>
      <td>5811</td>
      <td>5014</td>
      <td>3.994344</td>
      <td>17.355117</td>
      <td>20.474699</td>
      <td>12.779749</td>
      <td>23412.0</td>
      <td>...</td>
      <td>3.406593</td>
      <td>3.868132</td>
      <td>3.549451</td>
      <td>2.681319</td>
      <td>2.780220</td>
      <td>3.373626</td>
      <td>2.648352</td>
      <td>2.516484</td>
      <td>1.692308</td>
      <td>2.450549</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>23</td>
      <td>22</td>
      <td>23</td>
      <td>22</td>
      <td>8.118636</td>
      <td>23.240580</td>
      <td>50.896087</td>
      <td>21.424167</td>
      <td>118.0</td>
      <td>...</td>
      <td>0.010989</td>
      <td>0.054945</td>
      <td>0.010989</td>
      <td>0.000000</td>
      <td>0.010989</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.021978</td>
      <td>0.000000</td>
      <td>0.021978</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>170</td>
      <td>148</td>
      <td>170</td>
      <td>147</td>
      <td>6.928649</td>
      <td>27.431569</td>
      <td>30.052176</td>
      <td>15.848142</td>
      <td>854.0</td>
      <td>...</td>
      <td>0.120879</td>
      <td>0.021978</td>
      <td>0.076923</td>
      <td>0.043956</td>
      <td>0.087912</td>
      <td>0.043956</td>
      <td>0.021978</td>
      <td>0.043956</td>
      <td>0.010989</td>
      <td>0.032967</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>3290</td>
      <td>2992</td>
      <td>3280</td>
      <td>2980</td>
      <td>11.987794</td>
      <td>34.662048</td>
      <td>53.211112</td>
      <td>21.659658</td>
      <td>8177.0</td>
      <td>...</td>
      <td>2.384615</td>
      <td>2.054945</td>
      <td>3.703297</td>
      <td>2.087912</td>
      <td>1.560440</td>
      <td>1.769231</td>
      <td>1.516484</td>
      <td>1.868132</td>
      <td>1.769231</td>
      <td>1.395604</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 42 columns</p>
</div>


## آماده‌سازی برای خوشه‌بندی


```python
# Final validation of hourly zone demand profiles

hourly_total = hourly_zone_matrix.to_numpy().sum()

print("Hourly matrix shape:", hourly_zone_matrix.shape)
print("Hourly matrix total:", int(hourly_total))
print("Expected total trips:", int(hourly_df["trip_count"].sum()))

print(
    "Total demand consistent:",
    hourly_total == hourly_df["trip_count"].sum()
)

print("\nMissing values in hourly matrix:")
print(hourly_zone_matrix.isna().sum().sum())

print("\nNumber of hourly profile features:")
print(
    len([
        col for col in zone_analysis_df.columns
        if col.startswith("mean_demand_hour_")
    ])
)

print("\nUnknown zones:")
display(
    zone_analysis_df.loc[
        zone_analysis_df["is_unknown_zone"],
        ["LocationID", "pickup_pickup_trips",
         "dropoff_dropoff_trips"]
    ]
)
```

    Hourly matrix shape: (2183, 262)
    Hourly matrix total: 9554756
    Expected total trips: 9554756
    Total demand consistent: True
    
    Missing values in hourly matrix:
    0
    
    Number of hourly profile features:
    24
    
    Unknown zones:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LocationID</th>
      <th>pickup_pickup_trips</th>
      <th>dropoff_dropoff_trips</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>260</th>
      <td>264</td>
      <td>31588</td>
      <td>48499.0</td>
    </tr>
    <tr>
      <th>261</th>
      <td>265</td>
      <td>5172</td>
      <td>36377.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Prepare final zone-level clustering features

zone_model_df = zone_analysis_df.copy()

# Exclude unknown TLC zones from the main clustering analysis
zone_model_df = zone_model_df.loc[
    ~zone_model_df["is_unknown_zone"]
].copy()

# Log-transform total pickup and dropoff counts
zone_model_df["log_pickup_trips"] = np.log1p(
    zone_model_df["pickup_pickup_trips"]
)

zone_model_df["log_dropoff_trips"] = np.log1p(
    zone_model_df["dropoff_dropoff_trips"]
)

# Build relative hourly demand profiles
hourly_profile_columns = [
    f"mean_demand_hour_{hour:02d}"
    for hour in range(24)
]

hourly_totals = zone_model_df[
    hourly_profile_columns
].sum(axis=1)

for col in hourly_profile_columns:
    share_col = col.replace(
        "mean_demand",
        "demand_share"
    )

    zone_model_df[share_col] = np.divide(
        zone_model_df[col],
        hourly_totals,
        out=np.zeros(len(zone_model_df), dtype=float),
        where=hourly_totals.to_numpy() > 0
    )

demand_share_columns = [
    col.replace("mean_demand", "demand_share")
    for col in hourly_profile_columns
]

# Select features for clustering
feature_columns = [
    "log_pickup_trips",
    "log_dropoff_trips",

    "pickup_avg_trip_distance",
    "pickup_avg_duration_minutes",
    "pickup_avg_fare_amount",
    "pickup_avg_speed_mph",

    "dropoff_avg_trip_distance",
    "dropoff_avg_duration_minutes",
    "dropoff_avg_fare_amount",
    "dropoff_avg_speed_mph",

    *demand_share_columns
]

X_zone_raw = zone_model_df[
    feature_columns
].copy()

X_zone_raw.index = zone_model_df["LocationID"]

print("Zone-level clustering dataset")
print("Number of zones:", len(X_zone_raw))
print("Number of features:", len(feature_columns))

print("\nFeature names:")
print(feature_columns)

print("\nMissing values:")
display(X_zone_raw.isna().sum())

print("\nFirst 5 rows:")
display(X_zone_raw.head())
```

    Zone-level clustering dataset
    Number of zones: 260
    Number of features: 34
    
    Feature names:
    ['log_pickup_trips', 'log_dropoff_trips', 'pickup_avg_trip_distance', 'pickup_avg_duration_minutes', 'pickup_avg_fare_amount', 'pickup_avg_speed_mph', 'dropoff_avg_trip_distance', 'dropoff_avg_duration_minutes', 'dropoff_avg_fare_amount', 'dropoff_avg_speed_mph', 'demand_share_hour_00', 'demand_share_hour_01', 'demand_share_hour_02', 'demand_share_hour_03', 'demand_share_hour_04', 'demand_share_hour_05', 'demand_share_hour_06', 'demand_share_hour_07', 'demand_share_hour_08', 'demand_share_hour_09', 'demand_share_hour_10', 'demand_share_hour_11', 'demand_share_hour_12', 'demand_share_hour_13', 'demand_share_hour_14', 'demand_share_hour_15', 'demand_share_hour_16', 'demand_share_hour_17', 'demand_share_hour_18', 'demand_share_hour_19', 'demand_share_hour_20', 'demand_share_hour_21', 'demand_share_hour_22', 'demand_share_hour_23']
    
    Missing values:
    


    log_pickup_trips                0
    log_dropoff_trips               0
    pickup_avg_trip_distance        1
    pickup_avg_duration_minutes     0
    pickup_avg_fare_amount          0
    pickup_avg_speed_mph            1
    dropoff_avg_trip_distance       1
    dropoff_avg_duration_minutes    1
    dropoff_avg_fare_amount         1
    dropoff_avg_speed_mph           1
    demand_share_hour_00            0
    demand_share_hour_01            0
    demand_share_hour_02            0
    demand_share_hour_03            0
    demand_share_hour_04            0
    demand_share_hour_05            0
    demand_share_hour_06            0
    demand_share_hour_07            0
    demand_share_hour_08            0
    demand_share_hour_09            0
    demand_share_hour_10            0
    demand_share_hour_11            0
    demand_share_hour_12            0
    demand_share_hour_13            0
    demand_share_hour_14            0
    demand_share_hour_15            0
    demand_share_hour_16            0
    demand_share_hour_17            0
    demand_share_hour_18            0
    demand_share_hour_19            0
    demand_share_hour_20            0
    demand_share_hour_21            0
    demand_share_hour_22            0
    demand_share_hour_23            0
    dtype: int64


    
    First 5 rows:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>log_pickup_trips</th>
      <th>log_dropoff_trips</th>
      <th>pickup_avg_trip_distance</th>
      <th>pickup_avg_duration_minutes</th>
      <th>pickup_avg_fare_amount</th>
      <th>pickup_avg_speed_mph</th>
      <th>dropoff_avg_trip_distance</th>
      <th>dropoff_avg_duration_minutes</th>
      <th>dropoff_avg_fare_amount</th>
      <th>dropoff_avg_speed_mph</th>
      <th>...</th>
      <th>demand_share_hour_14</th>
      <th>demand_share_hour_15</th>
      <th>demand_share_hour_16</th>
      <th>demand_share_hour_17</th>
      <th>demand_share_hour_18</th>
      <th>demand_share_hour_19</th>
      <th>demand_share_hour_20</th>
      <th>demand_share_hour_21</th>
      <th>demand_share_hour_22</th>
      <th>demand_share_hour_23</th>
    </tr>
    <tr>
      <th>LocationID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>6.861711</td>
      <td>10.041117</td>
      <td>6.291392</td>
      <td>2.623382</td>
      <td>83.524591</td>
      <td>22.287090</td>
      <td>18.165595</td>
      <td>37.358979</td>
      <td>90.299463</td>
      <td>29.968444</td>
      <td>...</td>
      <td>0.102719</td>
      <td>0.115297</td>
      <td>0.111105</td>
      <td>0.095382</td>
      <td>0.079660</td>
      <td>0.072323</td>
      <td>0.022011</td>
      <td>0.006289</td>
      <td>0.007337</td>
      <td>0.003144</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.708050</td>
      <td>2.890372</td>
      <td>10.858462</td>
      <td>27.240476</td>
      <td>38.853571</td>
      <td>25.514643</td>
      <td>10.305000</td>
      <td>31.914706</td>
      <td>34.911765</td>
      <td>21.701508</td>
      <td>...</td>
      <td>0.142857</td>
      <td>0.000000</td>
      <td>0.214286</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.071429</td>
      <td>0.142857</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.860786</td>
      <td>6.638568</td>
      <td>9.439291</td>
      <td>39.942216</td>
      <td>36.439114</td>
      <td>13.778782</td>
      <td>13.397759</td>
      <td>34.943325</td>
      <td>46.783368</td>
      <td>24.339169</td>
      <td>...</td>
      <td>0.039999</td>
      <td>0.062855</td>
      <td>0.048570</td>
      <td>0.019999</td>
      <td>0.039999</td>
      <td>0.028571</td>
      <td>0.025713</td>
      <td>0.011428</td>
      <td>0.011428</td>
      <td>0.017142</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9.672816</td>
      <td>10.593856</td>
      <td>2.853294</td>
      <td>14.363545</td>
      <td>16.816984</td>
      <td>11.437372</td>
      <td>2.854780</td>
      <td>14.356796</td>
      <td>16.409995</td>
      <td>11.181948</td>
      <td>...</td>
      <td>0.020883</td>
      <td>0.019751</td>
      <td>0.018870</td>
      <td>0.023650</td>
      <td>0.032142</td>
      <td>0.038306</td>
      <td>0.034973</td>
      <td>0.040256</td>
      <td>0.049125</td>
      <td>0.077430</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.693147</td>
      <td>3.526361</td>
      <td>NaN</td>
      <td>0.166667</td>
      <td>100.000000</td>
      <td>NaN</td>
      <td>30.414687</td>
      <td>53.261111</td>
      <td>108.254545</td>
      <td>35.872295</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 34 columns</p>
</div>



```python
# Build preprocessing pipeline for zone clustering

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

zone_preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Fit and transform selected features
X_zone_scaled_array = zone_preprocessor.fit_transform(
    X_zone_raw
)

# Convert result back to DataFrame
X_zone_scaled = pd.DataFrame(
    X_zone_scaled_array,
    index=X_zone_raw.index,
    columns=X_zone_raw.columns
)

print("Preprocessing completed.")

print("Original shape:", X_zone_raw.shape)
print("Processed shape:", X_zone_scaled.shape)

print("\nMissing values after preprocessing:")
print(X_zone_scaled.isna().sum().sum())

print("\nFeature means after scaling:")
display(X_zone_scaled.mean().round(4))

print("\nFeature standard deviations after scaling:")
display(X_zone_scaled.std().round(4))

print("\nFirst 5 processed rows:")
display(X_zone_scaled.head())
```

    Preprocessing completed.
    Original shape: (260, 34)
    Processed shape: (260, 34)
    
    Missing values after preprocessing:
    0
    
    Feature means after scaling:
    


    log_pickup_trips                0.0
    log_dropoff_trips              -0.0
    pickup_avg_trip_distance       -0.0
    pickup_avg_duration_minutes    -0.0
    pickup_avg_fare_amount          0.0
    pickup_avg_speed_mph           -0.0
    dropoff_avg_trip_distance      -0.0
    dropoff_avg_duration_minutes    0.0
    dropoff_avg_fare_amount        -0.0
    dropoff_avg_speed_mph           0.0
    demand_share_hour_00            0.0
    demand_share_hour_01           -0.0
    demand_share_hour_02           -0.0
    demand_share_hour_03            0.0
    demand_share_hour_04            0.0
    demand_share_hour_05           -0.0
    demand_share_hour_06            0.0
    demand_share_hour_07           -0.0
    demand_share_hour_08           -0.0
    demand_share_hour_09           -0.0
    demand_share_hour_10           -0.0
    demand_share_hour_11           -0.0
    demand_share_hour_12           -0.0
    demand_share_hour_13           -0.0
    demand_share_hour_14           -0.0
    demand_share_hour_15           -0.0
    demand_share_hour_16           -0.0
    demand_share_hour_17            0.0
    demand_share_hour_18            0.0
    demand_share_hour_19            0.0
    demand_share_hour_20            0.0
    demand_share_hour_21            0.0
    demand_share_hour_22           -0.0
    demand_share_hour_23            0.0
    dtype: float64


    
    Feature standard deviations after scaling:
    


    log_pickup_trips                1.0019
    log_dropoff_trips               1.0019
    pickup_avg_trip_distance        1.0019
    pickup_avg_duration_minutes     1.0019
    pickup_avg_fare_amount          1.0019
    pickup_avg_speed_mph            1.0019
    dropoff_avg_trip_distance       1.0019
    dropoff_avg_duration_minutes    1.0019
    dropoff_avg_fare_amount         1.0019
    dropoff_avg_speed_mph           1.0019
    demand_share_hour_00            1.0019
    demand_share_hour_01            1.0019
    demand_share_hour_02            1.0019
    demand_share_hour_03            1.0019
    demand_share_hour_04            1.0019
    demand_share_hour_05            1.0019
    demand_share_hour_06            1.0019
    demand_share_hour_07            1.0019
    demand_share_hour_08            1.0019
    demand_share_hour_09            1.0019
    demand_share_hour_10            1.0019
    demand_share_hour_11            1.0019
    demand_share_hour_12            1.0019
    demand_share_hour_13            1.0019
    demand_share_hour_14            1.0019
    demand_share_hour_15            1.0019
    demand_share_hour_16            1.0019
    demand_share_hour_17            1.0019
    demand_share_hour_18            1.0019
    demand_share_hour_19            1.0019
    demand_share_hour_20            1.0019
    demand_share_hour_21            1.0019
    demand_share_hour_22            1.0019
    demand_share_hour_23            1.0019
    dtype: float64


    
    First 5 processed rows:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>log_pickup_trips</th>
      <th>log_dropoff_trips</th>
      <th>pickup_avg_trip_distance</th>
      <th>pickup_avg_duration_minutes</th>
      <th>pickup_avg_fare_amount</th>
      <th>pickup_avg_speed_mph</th>
      <th>dropoff_avg_trip_distance</th>
      <th>dropoff_avg_duration_minutes</th>
      <th>dropoff_avg_fare_amount</th>
      <th>dropoff_avg_speed_mph</th>
      <th>...</th>
      <th>demand_share_hour_14</th>
      <th>demand_share_hour_15</th>
      <th>demand_share_hour_16</th>
      <th>demand_share_hour_17</th>
      <th>demand_share_hour_18</th>
      <th>demand_share_hour_19</th>
      <th>demand_share_hour_20</th>
      <th>demand_share_hour_21</th>
      <th>demand_share_hour_22</th>
      <th>demand_share_hour_23</th>
    </tr>
    <tr>
      <th>LocationID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>-0.136119</td>
      <td>0.745188</td>
      <td>-0.133751</td>
      <td>-2.143655</td>
      <td>3.691001</td>
      <td>1.146804</td>
      <td>1.659397</td>
      <td>0.996506</td>
      <td>2.990674</td>
      <td>1.613717</td>
      <td>...</td>
      <td>1.330423</td>
      <td>0.941688</td>
      <td>2.028579</td>
      <td>1.430347</td>
      <td>1.448273</td>
      <td>1.024593</td>
      <td>-0.339324</td>
      <td>-0.971715</td>
      <td>-0.774026</td>
      <td>-1.071797</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.577204</td>
      <td>-2.233169</td>
      <td>1.007179</td>
      <td>0.041408</td>
      <td>0.583663</td>
      <td>1.662938</td>
      <td>0.208283</td>
      <td>0.439726</td>
      <td>-0.103685</td>
      <td>0.347417</td>
      <td>...</td>
      <td>2.367154</td>
      <td>-0.825571</td>
      <td>5.312785</td>
      <td>0.707079</td>
      <td>1.122734</td>
      <td>0.995933</td>
      <td>-0.818332</td>
      <td>-1.206938</td>
      <td>1.071112</td>
      <td>3.665772</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.483383</td>
      <td>-0.672008</td>
      <td>0.652647</td>
      <td>1.168840</td>
      <td>0.415712</td>
      <td>-0.213802</td>
      <td>0.779225</td>
      <td>0.749459</td>
      <td>0.559549</td>
      <td>0.751444</td>
      <td>...</td>
      <td>-0.289609</td>
      <td>0.137863</td>
      <td>0.038127</td>
      <td>-0.845795</td>
      <td>-0.120272</td>
      <td>-0.377628</td>
      <td>-0.258757</td>
      <td>-0.779494</td>
      <td>-0.656247</td>
      <td>-0.597139</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.839175</td>
      <td>0.975409</td>
      <td>-0.992645</td>
      <td>-1.101574</td>
      <td>-0.949213</td>
      <td>-0.588229</td>
      <td>-1.167073</td>
      <td>-1.355902</td>
      <td>-1.137328</td>
      <td>-1.263934</td>
      <td>...</td>
      <td>-0.783359</td>
      <td>-0.522836</td>
      <td>-0.907204</td>
      <td>-0.735552</td>
      <td>-0.430995</td>
      <td>-0.065608</td>
      <td>-0.057263</td>
      <td>0.298743</td>
      <td>0.429013</td>
      <td>1.447184</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-2.276261</td>
      <td>-1.968273</td>
      <td>-0.099699</td>
      <td>-2.361718</td>
      <td>4.837038</td>
      <td>-0.293448</td>
      <td>3.920654</td>
      <td>2.622799</td>
      <td>3.993776</td>
      <td>2.518049</td>
      <td>...</td>
      <td>-1.322749</td>
      <td>-0.825571</td>
      <td>-1.507829</td>
      <td>-1.449664</td>
      <td>-1.702166</td>
      <td>-1.293286</td>
      <td>-0.818332</td>
      <td>-1.206938</td>
      <td>-0.985255</td>
      <td>-1.178424</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 34 columns</p>
</div>


## PCA, کاهش ابعاد
<div dir="rtl" style="text-align: right;">
در PCA فعلی، دو نوع اطلاعات را با هم استفاده کرده‌ایم:
 اطلاعات مربوط به ویژگی‌های هر محدوده جغرافیایی:
- تعداد سفرهایی که از آن محدوده آغاز شده‌اند.
- تعداد سفرهایی که به آن محدوده ختم شده‌اند.
- میانگین مسافت، مدت، کرایه و سرعت سفرها.
   اطلاعات مربوط به الگوی زمانی تقاضا در هر محدوده:
- میانگین تعداد سفرها در ساعت ۰۰.
- میانگین تعداد سفرها در ساعت ۰۱.
- همین‌طور تا ساعت ۲۳.
پس ما ۳۴ ویژگی داریم که ۱۰ ویژگی اول اطلاعات مربوط به حجم و مشخصات سفرهای هر محدوده را نشان می‌دهند و ۲۴ ویژگی دیگر الگوی تقاضای ساعتی آن محدوده را توصیف می‌کنند.

 و  PCA با استفاده از ۳۴ ویژگی استانداردشده شامل مشخصات سفرها و الگوی تقاضای ساعتی، برای کاهش ابعاد داده‌ها و آماده‌سازی آن‌ها جهت خوشه‌بندی محدوده‌های جغرافیایی به کار گرفته شده است
</div>


```python

# PCA: Dimensionality Reduction for Zone Clustering

from sklearn.decomposition import PCA

# 1. Apply PCA to standardized zone features
zone_pca = PCA()

zone_pca_scores_array = zone_pca.fit_transform(
    X_zone_scaled
)

# 2. Create a DataFrame of PCA scores
pc_columns = [
    f"PC{i+1}"
    for i in range(zone_pca_scores_array.shape[1])
]

zone_pca_scores = pd.DataFrame(
    zone_pca_scores_array,
    index=X_zone_scaled.index,
    columns=pc_columns
)

# 3. Calculate explained and cumulative variance
explained_variance = pd.DataFrame({
    "explained_variance_ratio":
        zone_pca.explained_variance_ratio_
})

explained_variance["cumulative_variance"] = (
    explained_variance["explained_variance_ratio"].cumsum()
)

# 4. Display results
print("Original feature shape:", X_zone_scaled.shape)
print("PCA score shape:", zone_pca_scores.shape)

print("\nExplained variance by principal component:")
display(explained_variance.round(4))

print("\nFirst 5 PCA scores:")
display(zone_pca_scores.head())

```

    Original feature shape: (260, 34)
    PCA score shape: (260, 34)
    
    Explained variance by principal component:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>explained_variance_ratio</th>
      <th>cumulative_variance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.2632</td>
      <td>0.2632</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.1455</td>
      <td>0.4087</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0612</td>
      <td>0.4699</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0510</td>
      <td>0.5209</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0464</td>
      <td>0.5673</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0419</td>
      <td>0.6092</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0379</td>
      <td>0.6470</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0373</td>
      <td>0.6844</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0341</td>
      <td>0.7184</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.0315</td>
      <td>0.7499</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.0303</td>
      <td>0.7802</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.0262</td>
      <td>0.8064</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.0242</td>
      <td>0.8306</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.0213</td>
      <td>0.8519</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.0193</td>
      <td>0.8712</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.0178</td>
      <td>0.8890</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.0163</td>
      <td>0.9053</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.0138</td>
      <td>0.9191</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.0124</td>
      <td>0.9315</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.0119</td>
      <td>0.9435</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.0103</td>
      <td>0.9537</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.0090</td>
      <td>0.9627</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.0082</td>
      <td>0.9709</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.0068</td>
      <td>0.9777</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.0057</td>
      <td>0.9835</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.0049</td>
      <td>0.9884</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.0043</td>
      <td>0.9927</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.0029</td>
      <td>0.9956</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.0020</td>
      <td>0.9976</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.0016</td>
      <td>0.9992</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0.0004</td>
      <td>0.9996</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0.0003</td>
      <td>0.9999</td>
    </tr>
    <tr>
      <th>32</th>
      <td>0.0001</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>33</th>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
  </tbody>
</table>
</div>


    
    First 5 PCA scores:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PC1</th>
      <th>PC2</th>
      <th>PC3</th>
      <th>PC4</th>
      <th>PC5</th>
      <th>PC6</th>
      <th>PC7</th>
      <th>PC8</th>
      <th>PC9</th>
      <th>PC10</th>
      <th>...</th>
      <th>PC25</th>
      <th>PC26</th>
      <th>PC27</th>
      <th>PC28</th>
      <th>PC29</th>
      <th>PC30</th>
      <th>PC31</th>
      <th>PC32</th>
      <th>PC33</th>
      <th>PC34</th>
    </tr>
    <tr>
      <th>LocationID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1.310385</td>
      <td>-2.772500</td>
      <td>-3.875496</td>
      <td>0.911549</td>
      <td>-0.180726</td>
      <td>1.808703</td>
      <td>0.436630</td>
      <td>0.666978</td>
      <td>-0.042077</td>
      <td>0.297974</td>
      <td>...</td>
      <td>-1.316440</td>
      <td>-1.263598</td>
      <td>-2.017370</td>
      <td>0.460248</td>
      <td>1.696572</td>
      <td>0.108902</td>
      <td>0.023063</td>
      <td>-0.173743</td>
      <td>-0.254558</td>
      <td>1.220322e-15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.005918</td>
      <td>-4.383454</td>
      <td>-1.914243</td>
      <td>0.407425</td>
      <td>-1.773247</td>
      <td>2.281881</td>
      <td>-4.247505</td>
      <td>2.267863</td>
      <td>-2.231057</td>
      <td>0.904945</td>
      <td>...</td>
      <td>-0.838538</td>
      <td>0.960680</td>
      <td>-0.736746</td>
      <td>-0.059418</td>
      <td>-0.754053</td>
      <td>-0.056009</td>
      <td>0.305071</td>
      <td>0.089949</td>
      <td>0.021774</td>
      <td>-5.988809e-16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.656468</td>
      <td>0.943005</td>
      <td>0.399509</td>
      <td>0.038670</td>
      <td>-0.446352</td>
      <td>0.709136</td>
      <td>0.204861</td>
      <td>0.277944</td>
      <td>-0.309782</td>
      <td>0.197152</td>
      <td>...</td>
      <td>0.023764</td>
      <td>0.476658</td>
      <td>0.237819</td>
      <td>0.233508</td>
      <td>0.139948</td>
      <td>-0.381887</td>
      <td>0.264260</td>
      <td>-0.042943</td>
      <td>0.045145</td>
      <td>2.648109e-16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-3.413510</td>
      <td>-2.565654</td>
      <td>4.394655</td>
      <td>1.771870</td>
      <td>-0.544594</td>
      <td>-1.409790</td>
      <td>0.587315</td>
      <td>-1.394459</td>
      <td>0.201948</td>
      <td>2.058993</td>
      <td>...</td>
      <td>0.392596</td>
      <td>-0.207022</td>
      <td>0.180129</td>
      <td>0.125300</td>
      <td>-0.243126</td>
      <td>0.079673</td>
      <td>-0.020877</td>
      <td>-0.069641</td>
      <td>0.008393</td>
      <td>1.239058e-15</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6.876921</td>
      <td>-7.183830</td>
      <td>0.767552</td>
      <td>1.427543</td>
      <td>3.048522</td>
      <td>-3.935260</td>
      <td>7.508908</td>
      <td>5.266093</td>
      <td>4.733078</td>
      <td>-2.604149</td>
      <td>...</td>
      <td>0.198307</td>
      <td>0.400534</td>
      <td>0.420010</td>
      <td>0.207160</td>
      <td>0.085132</td>
      <td>0.356965</td>
      <td>0.029588</td>
      <td>-0.076222</td>
      <td>0.090448</td>
      <td>9.268416e-17</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 34 columns</p>
</div>



```python
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Output directory
FIGURE_DIR = Path(r"D:\کارشناسی\تحلیل داده\figures\phase2")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Number of principal components
n_components = len(explained_variance)

component_numbers = np.arange(1, n_components + 1)


# 1. Scree Plot
plt.figure(figsize=(10, 5))

plt.plot(
    component_numbers,
    explained_variance["explained_variance_ratio"],
    marker="o"
)

plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("PCA Scree Plot")
plt.xticks(component_numbers)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "pca_scree_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# 2. Cumulative Explained Variance
plt.figure(figsize=(10, 5))

plt.plot(
    component_numbers,
    explained_variance["cumulative_variance"],
    marker="o"
)

plt.axhline(
    y=0.80,
    linestyle="--",
    label="80% variance"
)

plt.axhline(
    y=0.90,
    linestyle="--",
    label="90% variance"
)

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance")
plt.xticks(component_numbers)
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "pca_cumulative_explained_variance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()
```


    
![png](output_47_0.png)
    



    
![png](output_47_1.png)
    



```python
# Prepare datasets for clustering comparison

# 1. Original standardized features (34 features)
X_cluster_original = X_zone_scaled.copy()

# 2. PCA with 12 components (~80.64% variance)
X_cluster_pca_12 = zone_pca_scores.iloc[:, :12].copy()

# 3. PCA with 17 components (~90.53% variance)
X_cluster_pca_17 = zone_pca_scores.iloc[:, :17].copy()
```


```python
# Compare dimensions of clustering datasets

print("Original standardized features:")
print(X_cluster_original.shape)

print("\nPCA - 12 components:")
print(X_cluster_pca_12.shape)

print("\nPCA - 17 components:")
print(X_cluster_pca_17.shape)
```

    Original standardized features:
    (260, 34)
    
    PCA - 12 components:
    (260, 12)
    
    PCA - 17 components:
    (260, 17)
    


```python
# Verify explained variance retained by PCA datasets

print("Explained variance comparison:\n")

for n in [12, 17]:
    variance = explained_variance.loc[
        n - 1, "cumulative_variance"
    ]

    print(f"PCA with {n} components: {variance:.2%}")
```

    Explained variance comparison:
    
    PCA with 12 components: 80.64%
    PCA with 17 components: 90.53%
    


```python
# Validate clustering datasets

datasets = {
    "Original": X_cluster_original,
    "PCA_12": X_cluster_pca_12,
    "PCA_17": X_cluster_pca_17
}

for name, data in datasets.items():

    print(f"\n{name}")
    print("-" * 30)

    print("Shape:", data.shape)
    print("Missing values:", data.isna().sum().sum())
    print("Infinite values:", np.isinf(data.to_numpy()).sum())
    print("Number of zones:", data.shape[0])
```

    
    Original
    ------------------------------
    Shape: (260, 34)
    Missing values: 0
    Infinite values: 0
    Number of zones: 260
    
    PCA_12
    ------------------------------
    Shape: (260, 12)
    Missing values: 0
    Infinite values: 0
    Number of zones: 260
    
    PCA_17
    ------------------------------
    Shape: (260, 17)
    Missing values: 0
    Infinite values: 0
    Number of zones: 260
    

## (خوشه بندی (روی محدوده جغرافیایی


```python
# Clustering: Import Required Libraries

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
```


```python
# Prepare datasets for clustering comparison

X_cluster_original = X_zone_scaled.copy()

X_cluster_pca_12 = zone_pca_scores.iloc[:, :12].copy()

X_cluster_pca_17 = zone_pca_scores.iloc[:, :17].copy()
```


```python
print("Original:", X_cluster_original.shape)
print("PCA 12:", X_cluster_pca_12.shape)
print("PCA 17:", X_cluster_pca_17.shape)
```

    Original: (260, 34)
    PCA 12: (260, 12)
    PCA 17: (260, 17)
    


```python

# Evaluate the Optimal Number of Clusters

import matplotlib.pyplot as plt

# Three datasets for comparison
clustering_datasets = {
    "Original (34 Features)": X_cluster_original,
    "PCA (12 Components)": X_cluster_pca_12,
    "PCA (17 Components)": X_cluster_pca_17
}

# Test different numbers of clusters
k_values = range(2, 11)

results = []

for dataset_name, X in clustering_datasets.items():

    for k in k_values:

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20
        )

        labels = kmeans.fit_predict(X)

        results.append({
            "Dataset": dataset_name,
            "K": k,
            "Inertia": kmeans.inertia_,
            "Silhouette Score": silhouette_score(X, labels)
        })

# Create results table
clustering_evaluation = pd.DataFrame(results)

display(clustering_evaluation.round(4))

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dataset</th>
      <th>K</th>
      <th>Inertia</th>
      <th>Silhouette Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Original (34 Features)</td>
      <td>2</td>
      <td>7209.0024</td>
      <td>0.2172</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Original (34 Features)</td>
      <td>3</td>
      <td>6387.8140</td>
      <td>0.2525</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Original (34 Features)</td>
      <td>4</td>
      <td>5859.4768</td>
      <td>0.2212</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Original (34 Features)</td>
      <td>5</td>
      <td>5478.1674</td>
      <td>0.2173</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Original (34 Features)</td>
      <td>6</td>
      <td>5213.8262</td>
      <td>0.2191</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Original (34 Features)</td>
      <td>7</td>
      <td>4879.4945</td>
      <td>0.2226</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Original (34 Features)</td>
      <td>8</td>
      <td>4637.1297</td>
      <td>0.2127</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Original (34 Features)</td>
      <td>9</td>
      <td>4428.5639</td>
      <td>0.2222</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Original (34 Features)</td>
      <td>10</td>
      <td>4208.1455</td>
      <td>0.2254</td>
    </tr>
    <tr>
      <th>9</th>
      <td>PCA (12 Components)</td>
      <td>2</td>
      <td>5500.8409</td>
      <td>0.2906</td>
    </tr>
    <tr>
      <th>10</th>
      <td>PCA (12 Components)</td>
      <td>3</td>
      <td>4694.9944</td>
      <td>0.3185</td>
    </tr>
    <tr>
      <th>11</th>
      <td>PCA (12 Components)</td>
      <td>4</td>
      <td>4189.4439</td>
      <td>0.2852</td>
    </tr>
    <tr>
      <th>12</th>
      <td>PCA (12 Components)</td>
      <td>5</td>
      <td>3805.0519</td>
      <td>0.2844</td>
    </tr>
    <tr>
      <th>13</th>
      <td>PCA (12 Components)</td>
      <td>6</td>
      <td>3515.3644</td>
      <td>0.2768</td>
    </tr>
    <tr>
      <th>14</th>
      <td>PCA (12 Components)</td>
      <td>7</td>
      <td>3251.4827</td>
      <td>0.2852</td>
    </tr>
    <tr>
      <th>15</th>
      <td>PCA (12 Components)</td>
      <td>8</td>
      <td>3029.5155</td>
      <td>0.2946</td>
    </tr>
    <tr>
      <th>16</th>
      <td>PCA (12 Components)</td>
      <td>9</td>
      <td>2751.5270</td>
      <td>0.2951</td>
    </tr>
    <tr>
      <th>17</th>
      <td>PCA (12 Components)</td>
      <td>10</td>
      <td>2539.2672</td>
      <td>0.2968</td>
    </tr>
    <tr>
      <th>18</th>
      <td>PCA (17 Components)</td>
      <td>2</td>
      <td>6373.3719</td>
      <td>0.2493</td>
    </tr>
    <tr>
      <th>19</th>
      <td>PCA (17 Components)</td>
      <td>3</td>
      <td>5559.8996</td>
      <td>0.2830</td>
    </tr>
    <tr>
      <th>20</th>
      <td>PCA (17 Components)</td>
      <td>4</td>
      <td>5039.4774</td>
      <td>0.2491</td>
    </tr>
    <tr>
      <th>21</th>
      <td>PCA (17 Components)</td>
      <td>5</td>
      <td>4646.4148</td>
      <td>0.2499</td>
    </tr>
    <tr>
      <th>22</th>
      <td>PCA (17 Components)</td>
      <td>6</td>
      <td>4393.3332</td>
      <td>0.2504</td>
    </tr>
    <tr>
      <th>23</th>
      <td>PCA (17 Components)</td>
      <td>7</td>
      <td>4064.4594</td>
      <td>0.2528</td>
    </tr>
    <tr>
      <th>24</th>
      <td>PCA (17 Components)</td>
      <td>8</td>
      <td>3830.5533</td>
      <td>0.2734</td>
    </tr>
    <tr>
      <th>25</th>
      <td>PCA (17 Components)</td>
      <td>9</td>
      <td>3566.7107</td>
      <td>0.2612</td>
    </tr>
    <tr>
      <th>26</th>
      <td>PCA (17 Components)</td>
      <td>10</td>
      <td>3371.4776</td>
      <td>0.2601</td>
    </tr>
  </tbody>
</table>
</div>



```python
import os
import matplotlib.pyplot as plt

# Output directory
output_dir = r"D:\کارشناسی\تحلیل داده\figures\phase2"

# Create directory if it does not exist
os.makedirs(output_dir, exist_ok=True)


# ==========================================
# 1. Plot Elbow Results
# ==========================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (dataset_name, _) in zip(axes, clustering_datasets.items()):

    subset = clustering_evaluation[
        clustering_evaluation["Dataset"] == dataset_name
    ]

    ax.plot(
        subset["K"],
        subset["Inertia"],
        marker="o"
    )

    ax.set_title(dataset_name)
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Inertia")
    ax.set_xticks(list(k_values))
    ax.grid(True, alpha=0.3)

plt.suptitle("Elbow Method - K-Means")
plt.tight_layout()

# Save figure
plt.savefig(
    os.path.join(output_dir, "elbow_method_kmeans.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ==========================================
# 2. Plot Silhouette Results
# ==========================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (dataset_name, _) in zip(axes, clustering_datasets.items()):

    subset = clustering_evaluation[
        clustering_evaluation["Dataset"] == dataset_name
    ]

    ax.plot(
        subset["K"],
        subset["Silhouette Score"],
        marker="o"
    )

    ax.set_title(dataset_name)
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Silhouette Score")
    ax.set_xticks(list(k_values))
    ax.grid(True, alpha=0.3)

plt.suptitle("Silhouette Analysis - K-Means")
plt.tight_layout()

# Save figure
plt.savefig(
    os.path.join(output_dir, "silhouette_analysis_kmeans.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()
```


    
![png](output_57_0.png)
    



    
![png](output_57_1.png)
    



```python
# Final K-Means Clustering with K=3

final_k = 3

cluster_labels = {}

for dataset_name, X in clustering_datasets.items():

    kmeans = KMeans(
        n_clusters=final_k,
        random_state=42,
        n_init=20
    )

    labels = kmeans.fit_predict(X)

    cluster_labels[dataset_name] = labels

    print(f"\n{dataset_name}")
    print("-" * 40)

    cluster_counts = pd.Series(labels).value_counts().sort_index()

    print("Cluster sizes:")
    print(cluster_counts)

    print("\nCluster percentages:")
    print((cluster_counts / len(labels) * 100).round(2))
```

    
    Original (34 Features)
    ----------------------------------------
    Cluster sizes:
    0     17
    1    112
    2    131
    Name: count, dtype: int64
    
    Cluster percentages:
    0     6.54
    1    43.08
    2    50.38
    Name: count, dtype: float64
    
    PCA (12 Components)
    ----------------------------------------
    Cluster sizes:
    0     17
    1    135
    2    108
    Name: count, dtype: int64
    
    Cluster percentages:
    0     6.54
    1    51.92
    2    41.54
    Name: count, dtype: float64
    
    PCA (17 Components)
    ----------------------------------------
    Cluster sizes:
    0    112
    1     17
    2    131
    Name: count, dtype: int64
    
    Cluster percentages:
    0    43.08
    1     6.54
    2    50.38
    Name: count, dtype: float64
    


```python
# Map cluster labels to zone IDs

cluster_assignments = {}

for dataset_name, X in clustering_datasets.items():
    cluster_assignments[dataset_name] = pd.DataFrame({
        "LocationID": X.index,
        "Cluster": cluster_labels[dataset_name]
    })

    print(f"\n{dataset_name}")
    display(cluster_assignments[dataset_name].head())
```

    
    Original (34 Features)
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LocationID</th>
      <th>Cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


    
    PCA (12 Components)
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LocationID</th>
      <th>Cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


    
    PCA (17 Components)
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LocationID</th>
      <th>Cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Cluster profiling for all three datasets

# Prepare zone-level features
zone_profile_base = zone_analysis_df.copy()

if "LocationID" in zone_profile_base.columns:
    zone_profile_base = zone_profile_base.set_index("LocationID")

# Exclude unknown zones
zone_profile_base = zone_profile_base.loc[
    ~zone_profile_base.index.isin([264, 265])
]

cluster_profiles = {}

for dataset_name, assignments in cluster_assignments.items():

    profile_df = zone_profile_base.join(
        assignments.set_index("LocationID"),
        how="inner"
    )

    numeric_cols = profile_df.select_dtypes(
        include=np.number
    ).columns.drop("Cluster", errors="ignore")

    cluster_profiles[dataset_name] = (
        profile_df.groupby("Cluster")[numeric_cols]
        .mean()
        .round(3)
    )

    print(f"\n{dataset_name}")
    display(cluster_profiles[dataset_name])
```

    
    Original (34 Features)
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pickup_pickup_trips</th>
      <th>pickup_distance_n</th>
      <th>pickup_duration_n</th>
      <th>pickup_speed_n</th>
      <th>pickup_avg_trip_distance</th>
      <th>pickup_avg_duration_minutes</th>
      <th>pickup_avg_fare_amount</th>
      <th>pickup_avg_speed_mph</th>
      <th>dropoff_dropoff_trips</th>
      <th>dropoff_distance_n</th>
      <th>...</th>
      <th>mean_demand_hour_14</th>
      <th>mean_demand_hour_15</th>
      <th>mean_demand_hour_16</th>
      <th>mean_demand_hour_17</th>
      <th>mean_demand_hour_18</th>
      <th>mean_demand_hour_19</th>
      <th>mean_demand_hour_20</th>
      <th>mean_demand_hour_21</th>
      <th>mean_demand_hour_22</th>
      <th>mean_demand_hour_23</th>
    </tr>
    <tr>
      <th>Cluster</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9.176</td>
      <td>6.824</td>
      <td>9.176</td>
      <td>6.706</td>
      <td>14.199</td>
      <td>19.920</td>
      <td>52.206</td>
      <td>29.483</td>
      <td>96.059</td>
      <td>93.471</td>
      <td>...</td>
      <td>0.005</td>
      <td>0.005</td>
      <td>0.005</td>
      <td>0.008</td>
      <td>0.002</td>
      <td>0.000</td>
      <td>0.006</td>
      <td>0.004</td>
      <td>0.008</td>
      <td>0.009</td>
    </tr>
    <tr>
      <th>1</th>
      <td>84086.696</td>
      <td>82312.964</td>
      <td>84013.875</td>
      <td>82239.152</td>
      <td>4.371</td>
      <td>18.048</td>
      <td>23.223</td>
      <td>13.648</td>
      <td>81722.866</td>
      <td>79991.161</td>
      <td>...</td>
      <td>54.512</td>
      <td>56.321</td>
      <td>57.519</td>
      <td>63.576</td>
      <td>67.255</td>
      <td>59.775</td>
      <td>53.743</td>
      <td>53.772</td>
      <td>48.814</td>
      <td>37.452</td>
    </tr>
    <tr>
      <th>2</th>
      <td>764.351</td>
      <td>669.061</td>
      <td>762.473</td>
      <td>666.634</td>
      <td>8.029</td>
      <td>35.124</td>
      <td>33.831</td>
      <td>14.630</td>
      <td>2406.763</td>
      <td>2305.374</td>
      <td>...</td>
      <td>0.406</td>
      <td>0.362</td>
      <td>0.315</td>
      <td>0.292</td>
      <td>0.262</td>
      <td>0.233</td>
      <td>0.189</td>
      <td>0.152</td>
      <td>0.154</td>
      <td>0.176</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 40 columns</p>
</div>


    
    PCA (12 Components)
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pickup_pickup_trips</th>
      <th>pickup_distance_n</th>
      <th>pickup_duration_n</th>
      <th>pickup_speed_n</th>
      <th>pickup_avg_trip_distance</th>
      <th>pickup_avg_duration_minutes</th>
      <th>pickup_avg_fare_amount</th>
      <th>pickup_avg_speed_mph</th>
      <th>dropoff_dropoff_trips</th>
      <th>dropoff_distance_n</th>
      <th>...</th>
      <th>mean_demand_hour_14</th>
      <th>mean_demand_hour_15</th>
      <th>mean_demand_hour_16</th>
      <th>mean_demand_hour_17</th>
      <th>mean_demand_hour_18</th>
      <th>mean_demand_hour_19</th>
      <th>mean_demand_hour_20</th>
      <th>mean_demand_hour_21</th>
      <th>mean_demand_hour_22</th>
      <th>mean_demand_hour_23</th>
    </tr>
    <tr>
      <th>Cluster</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9.176</td>
      <td>6.824</td>
      <td>9.176</td>
      <td>6.706</td>
      <td>14.199</td>
      <td>19.920</td>
      <td>52.206</td>
      <td>29.483</td>
      <td>96.059</td>
      <td>93.471</td>
      <td>...</td>
      <td>0.005</td>
      <td>0.005</td>
      <td>0.005</td>
      <td>0.008</td>
      <td>0.002</td>
      <td>0.000</td>
      <td>0.006</td>
      <td>0.004</td>
      <td>0.008</td>
      <td>0.009</td>
    </tr>
    <tr>
      <th>1</th>
      <td>781.407</td>
      <td>684.970</td>
      <td>779.467</td>
      <td>682.363</td>
      <td>8.063</td>
      <td>34.911</td>
      <td>34.062</td>
      <td>14.822</td>
      <td>2437.756</td>
      <td>2335.311</td>
      <td>...</td>
      <td>0.413</td>
      <td>0.373</td>
      <td>0.332</td>
      <td>0.307</td>
      <td>0.273</td>
      <td>0.241</td>
      <td>0.199</td>
      <td>0.165</td>
      <td>0.168</td>
      <td>0.193</td>
    </tr>
    <tr>
      <th>2</th>
      <td>87151.389</td>
      <td>85316.926</td>
      <td>87076.019</td>
      <td>85240.694</td>
      <td>4.193</td>
      <td>17.682</td>
      <td>22.541</td>
      <td>13.372</td>
      <td>84621.759</td>
      <td>82830.991</td>
      <td>...</td>
      <td>56.507</td>
      <td>58.379</td>
      <td>59.615</td>
      <td>65.901</td>
      <td>69.722</td>
      <td>61.970</td>
      <td>55.714</td>
      <td>55.741</td>
      <td>50.599</td>
      <td>38.812</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 40 columns</p>
</div>


    
    PCA (17 Components)
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pickup_pickup_trips</th>
      <th>pickup_distance_n</th>
      <th>pickup_duration_n</th>
      <th>pickup_speed_n</th>
      <th>pickup_avg_trip_distance</th>
      <th>pickup_avg_duration_minutes</th>
      <th>pickup_avg_fare_amount</th>
      <th>pickup_avg_speed_mph</th>
      <th>dropoff_dropoff_trips</th>
      <th>dropoff_distance_n</th>
      <th>...</th>
      <th>mean_demand_hour_14</th>
      <th>mean_demand_hour_15</th>
      <th>mean_demand_hour_16</th>
      <th>mean_demand_hour_17</th>
      <th>mean_demand_hour_18</th>
      <th>mean_demand_hour_19</th>
      <th>mean_demand_hour_20</th>
      <th>mean_demand_hour_21</th>
      <th>mean_demand_hour_22</th>
      <th>mean_demand_hour_23</th>
    </tr>
    <tr>
      <th>Cluster</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>84086.696</td>
      <td>82312.964</td>
      <td>84013.875</td>
      <td>82239.152</td>
      <td>4.371</td>
      <td>18.048</td>
      <td>23.223</td>
      <td>13.648</td>
      <td>81722.866</td>
      <td>79991.161</td>
      <td>...</td>
      <td>54.512</td>
      <td>56.321</td>
      <td>57.519</td>
      <td>63.576</td>
      <td>67.255</td>
      <td>59.775</td>
      <td>53.743</td>
      <td>53.772</td>
      <td>48.814</td>
      <td>37.452</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9.176</td>
      <td>6.824</td>
      <td>9.176</td>
      <td>6.706</td>
      <td>14.199</td>
      <td>19.920</td>
      <td>52.206</td>
      <td>29.483</td>
      <td>96.059</td>
      <td>93.471</td>
      <td>...</td>
      <td>0.005</td>
      <td>0.005</td>
      <td>0.005</td>
      <td>0.008</td>
      <td>0.002</td>
      <td>0.000</td>
      <td>0.006</td>
      <td>0.004</td>
      <td>0.008</td>
      <td>0.009</td>
    </tr>
    <tr>
      <th>2</th>
      <td>764.351</td>
      <td>669.061</td>
      <td>762.473</td>
      <td>666.634</td>
      <td>8.029</td>
      <td>35.124</td>
      <td>33.831</td>
      <td>14.630</td>
      <td>2406.763</td>
      <td>2305.374</td>
      <td>...</td>
      <td>0.406</td>
      <td>0.362</td>
      <td>0.315</td>
      <td>0.292</td>
      <td>0.262</td>
      <td>0.233</td>
      <td>0.189</td>
      <td>0.152</td>
      <td>0.154</td>
      <td>0.176</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 40 columns</p>
</div>



```python
# مقایسه دقیق سه روش

from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# 1. Compare clustering quality
quality_results = []

for dataset_name, X in clustering_datasets.items():
    labels = cluster_labels[dataset_name]

    quality_results.append({
        "Dataset": dataset_name,
        "Silhouette": silhouette_score(X, labels),
        "Davies-Bouldin": davies_bouldin_score(X, labels),
        "Calinski-Harabasz": calinski_harabasz_score(X, labels)
    })

quality_comparison = pd.DataFrame(quality_results)
display(quality_comparison.round(4))


# 2. Compare cluster memberships
dataset_names = list(clustering_datasets.keys())

ari_results = []

for i in range(len(dataset_names)):
    for j in range(i + 1, len(dataset_names)):

        name_1 = dataset_names[i]
        name_2 = dataset_names[j]

        ari = adjusted_rand_score(
            cluster_labels[name_1],
            cluster_labels[name_2]
        )

        ari_results.append({
            "Comparison": f"{name_1} vs {name_2}",
            "Adjusted Rand Index": ari
        })

ari_comparison = pd.DataFrame(ari_results)
display(ari_comparison.round(4))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dataset</th>
      <th>Silhouette</th>
      <th>Davies-Bouldin</th>
      <th>Calinski-Harabasz</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Original (34 Features)</td>
      <td>0.2525</td>
      <td>1.9201</td>
      <td>49.3292</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PCA (12 Components)</td>
      <td>0.3185</td>
      <td>1.6536</td>
      <td>66.6037</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PCA (17 Components)</td>
      <td>0.2830</td>
      <td>1.7908</td>
      <td>56.4574</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Comparison</th>
      <th>Adjusted Rand Index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Original (34 Features) vs PCA (12 Components)</td>
      <td>0.9425</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Original (34 Features) vs PCA (17 Components)</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PCA (12 Components) vs PCA (17 Components)</td>
      <td>0.9425</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Cluster size comparison

for dataset_name, assignments in cluster_assignments.items():

    print(f"\n{dataset_name}")

    summary = (
        assignments["Cluster"]
        .value_counts()
        .sort_index()
        .rename("Zone_Count")
        .to_frame()
    )

    summary["Percentage"] = (
        summary["Zone_Count"] / len(assignments) * 100
    ).round(2)

    display(summary)
```

    
    Original (34 Features)
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Zone_Count</th>
      <th>Percentage</th>
    </tr>
    <tr>
      <th>Cluster</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17</td>
      <td>6.54</td>
    </tr>
    <tr>
      <th>1</th>
      <td>112</td>
      <td>43.08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>131</td>
      <td>50.38</td>
    </tr>
  </tbody>
</table>
</div>


    
    PCA (12 Components)
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Zone_Count</th>
      <th>Percentage</th>
    </tr>
    <tr>
      <th>Cluster</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17</td>
      <td>6.54</td>
    </tr>
    <tr>
      <th>1</th>
      <td>135</td>
      <td>51.92</td>
    </tr>
    <tr>
      <th>2</th>
      <td>108</td>
      <td>41.54</td>
    </tr>
  </tbody>
</table>
</div>


    
    PCA (17 Components)
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Zone_Count</th>
      <th>Percentage</th>
    </tr>
    <tr>
      <th>Cluster</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>112</td>
      <td>43.08</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17</td>
      <td>6.54</td>
    </tr>
    <tr>
      <th>2</th>
      <td>131</td>
      <td>50.38</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Identify the smallest cluster in each clustering method

small_cluster_zones = {}

for dataset_name, assignments in cluster_assignments.items():

    smallest_cluster = (
        assignments["Cluster"]
        .value_counts()
        .idxmin()
    )

    zones = sorted(
        assignments.loc[
            assignments["Cluster"] == smallest_cluster,
            "LocationID"
        ].tolist()
    )

    small_cluster_zones[dataset_name] = set(zones)

    print(f"\n{dataset_name}")
    print(f"Smallest cluster: {smallest_cluster}")
    print(f"Number of zones: {len(zones)}")
    print("LocationIDs:", zones)


# Compare the smallest clusters
print("\nComparison of smallest clusters:")

for name_1, zones_1 in small_cluster_zones.items():
    for name_2, zones_2 in small_cluster_zones.items():
        if name_1 < name_2:
            print(
                f"{name_1} vs {name_2}: "
                f"{len(zones_1.intersection(zones_2))} common zones"
            )
```

    
    Original (34 Features)
    Smallest cluster: 0
    Number of zones: 17
    LocationIDs: [5, 23, 27, 44, 84, 99, 109, 115, 118, 156, 172, 199, 204, 206, 214, 245, 251]
    
    PCA (12 Components)
    Smallest cluster: 0
    Number of zones: 17
    LocationIDs: [5, 23, 27, 44, 84, 99, 109, 115, 118, 156, 172, 199, 204, 206, 214, 245, 251]
    
    PCA (17 Components)
    Smallest cluster: 1
    Number of zones: 17
    LocationIDs: [5, 23, 27, 44, 84, 99, 109, 115, 118, 156, 172, 199, 204, 206, 214, 245, 251]
    
    Comparison of smallest clusters:
    Original (34 Features) vs PCA (12 Components): 17 common zones
    Original (34 Features) vs PCA (17 Components): 17 common zones
    PCA (12 Components) vs PCA (17 Components): 17 common zones
    


```python
# Compare the 17 small-cluster zones with all other zones

small_zones = small_cluster_zones["Original (34 Features)"]

comparison_df = zone_profile_base.copy()

comparison_df["Zone_Group"] = np.where(
    comparison_df.index.isin(small_zones),
    "Small Cluster (17 zones)",
    "Other Zones (243 zones)"
)

key_features = [
    "pickup_pickup_trips",
    "dropoff_dropoff_trips",
    "pickup_avg_trip_distance",
    "pickup_avg_duration_minutes",
    "pickup_avg_fare_amount",
    "pickup_avg_speed_mph",
    "dropoff_avg_trip_distance",
    "dropoff_avg_duration_minutes",
    "dropoff_avg_fare_amount",
    "dropoff_avg_speed_mph"
]

available_features = [
    col for col in key_features
    if col in comparison_df.columns
]

group_comparison = comparison_df.groupby("Zone_Group")[
    available_features
].agg(["mean", "median"]).round(3)

display(group_comparison)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">pickup_pickup_trips</th>
      <th colspan="2" halign="left">dropoff_dropoff_trips</th>
      <th colspan="2" halign="left">pickup_avg_trip_distance</th>
      <th colspan="2" halign="left">pickup_avg_duration_minutes</th>
      <th colspan="2" halign="left">pickup_avg_fare_amount</th>
      <th colspan="2" halign="left">pickup_avg_speed_mph</th>
      <th colspan="2" halign="left">dropoff_avg_trip_distance</th>
      <th colspan="2" halign="left">dropoff_avg_duration_minutes</th>
      <th colspan="2" halign="left">dropoff_avg_fare_amount</th>
      <th colspan="2" halign="left">dropoff_avg_speed_mph</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th>median</th>
    </tr>
    <tr>
      <th>Zone_Group</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Other Zones (243 zones)</th>
      <td>39168.066</td>
      <td>1107.0</td>
      <td>38963.979</td>
      <td>3017.0</td>
      <td>6.343</td>
      <td>6.365</td>
      <td>27.253</td>
      <td>27.141</td>
      <td>28.942</td>
      <td>29.531</td>
      <td>14.177</td>
      <td>13.071</td>
      <td>8.233</td>
      <td>8.219</td>
      <td>26.428</td>
      <td>27.215</td>
      <td>33.671</td>
      <td>34.398</td>
      <td>18.611</td>
      <td>18.744</td>
    </tr>
    <tr>
      <th>Small Cluster (17 zones)</th>
      <td>9.176</td>
      <td>10.0</td>
      <td>96.059</td>
      <td>100.0</td>
      <td>14.199</td>
      <td>12.627</td>
      <td>19.920</td>
      <td>20.945</td>
      <td>52.206</td>
      <td>41.167</td>
      <td>29.483</td>
      <td>26.651</td>
      <td>23.553</td>
      <td>22.112</td>
      <td>45.643</td>
      <td>44.998</td>
      <td>83.922</td>
      <td>80.227</td>
      <td>31.914</td>
      <td>31.360</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Compare hourly demand profiles

hourly_features = [
    col for col in comparison_df.columns
    if col.startswith("mean_demand_hour_")
]

hourly_comparison = (
    comparison_df.groupby("Zone_Group")[hourly_features]
    .mean()
    .T
    .round(4)
)

display(hourly_comparison)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Zone_Group</th>
      <th>Other Zones (243 zones)</th>
      <th>Small Cluster (17 zones)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mean_demand_hour_00</th>
      <td>12.1371</td>
      <td>0.0078</td>
    </tr>
    <tr>
      <th>mean_demand_hour_01</th>
      <td>8.1580</td>
      <td>0.0013</td>
    </tr>
    <tr>
      <th>mean_demand_hour_02</th>
      <td>5.4274</td>
      <td>0.0026</td>
    </tr>
    <tr>
      <th>mean_demand_hour_03</th>
      <td>3.7017</td>
      <td>0.0058</td>
    </tr>
    <tr>
      <th>mean_demand_hour_04</th>
      <td>2.6093</td>
      <td>0.0058</td>
    </tr>
    <tr>
      <th>mean_demand_hour_05</th>
      <td>2.6915</td>
      <td>0.0039</td>
    </tr>
    <tr>
      <th>mean_demand_hour_06</th>
      <td>5.9488</td>
      <td>0.0019</td>
    </tr>
    <tr>
      <th>mean_demand_hour_07</th>
      <td>11.9072</td>
      <td>0.0013</td>
    </tr>
    <tr>
      <th>mean_demand_hour_08</th>
      <td>16.7340</td>
      <td>0.0039</td>
    </tr>
    <tr>
      <th>mean_demand_hour_09</th>
      <td>18.2604</td>
      <td>0.0019</td>
    </tr>
    <tr>
      <th>mean_demand_hour_10</th>
      <td>19.3294</td>
      <td>0.0026</td>
    </tr>
    <tr>
      <th>mean_demand_hour_11</th>
      <td>20.8555</td>
      <td>0.0026</td>
    </tr>
    <tr>
      <th>mean_demand_hour_12</th>
      <td>22.9262</td>
      <td>0.0052</td>
    </tr>
    <tr>
      <th>mean_demand_hour_13</th>
      <td>23.6626</td>
      <td>0.0032</td>
    </tr>
    <tr>
      <th>mean_demand_hour_14</th>
      <td>25.3436</td>
      <td>0.0045</td>
    </tr>
    <tr>
      <th>mean_demand_hour_15</th>
      <td>26.1537</td>
      <td>0.0045</td>
    </tr>
    <tr>
      <th>mean_demand_hour_16</th>
      <td>26.6802</td>
      <td>0.0045</td>
    </tr>
    <tr>
      <th>mean_demand_hour_17</th>
      <td>29.4600</td>
      <td>0.0078</td>
    </tr>
    <tr>
      <th>mean_demand_hour_18</th>
      <td>31.1392</td>
      <td>0.0019</td>
    </tr>
    <tr>
      <th>mean_demand_hour_19</th>
      <td>27.6759</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>mean_demand_hour_20</th>
      <td>24.8723</td>
      <td>0.0065</td>
    </tr>
    <tr>
      <th>mean_demand_hour_21</th>
      <td>24.8655</td>
      <td>0.0039</td>
    </tr>
    <tr>
      <th>mean_demand_hour_22</th>
      <td>22.5817</td>
      <td>0.0084</td>
    </tr>
    <tr>
      <th>mean_demand_hour_23</th>
      <td>17.3567</td>
      <td>0.0090</td>
    </tr>
  </tbody>
</table>
</div>



```python
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES_DIR = Path(r"D:\کارشناسی\تحلیل داده\figures\phase2")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

hourly_comparison.plot(figsize=(10, 5), marker="o")

plt.title("Hourly Demand: Small Cluster vs Other Zones")
plt.xlabel("Hour of Day")
plt.ylabel("Mean Hourly Demand")
plt.xticks(range(24))
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "cluster_hourly_demand_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
```


    
![png](output_66_0.png)
    


## ادامه تحلیل ویژگی‌های زمانی و سری زمانی


```python
# Temporal Feature Relationship Analysis

temporal_cols = [
    col for col in temporal_features.columns
    if "lag" in col.lower() or "rolling" in col.lower()
]

temporal_corr = (
    temporal_features[["trip_count"] + temporal_cols]
    .corr()["trip_count"]
    .drop("trip_count")
    .sort_values(key=abs, ascending=False)
    .to_frame("correlation_with_trip_count")
)

display(temporal_corr.round(4))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>correlation_with_trip_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>lag_168</th>
      <td>0.9465</td>
    </tr>
    <tr>
      <th>lag_1</th>
      <td>0.9342</td>
    </tr>
    <tr>
      <th>lag_24</th>
      <td>0.8525</td>
    </tr>
    <tr>
      <th>rolling_mean_3</th>
      <td>0.8089</td>
    </tr>
    <tr>
      <th>lag_2</th>
      <td>0.7885</td>
    </tr>
    <tr>
      <th>lag_3</th>
      <td>0.6099</td>
    </tr>
    <tr>
      <th>rolling_std_24</th>
      <td>0.2515</td>
    </tr>
    <tr>
      <th>rolling_mean_24</th>
      <td>0.1837</td>
    </tr>
    <tr>
      <th>rolling_std_3</th>
      <td>-0.1511</td>
    </tr>
    <tr>
      <th>rolling_mean_168</th>
      <td>0.1343</td>
    </tr>
    <tr>
      <th>rolling_std_168</th>
      <td>0.1297</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Correlation Between Engineered Temporal Features

temporal_feature_corr = temporal_features[temporal_cols].corr()

display(temporal_feature_corr.round(3))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lag_1</th>
      <th>lag_2</th>
      <th>lag_3</th>
      <th>lag_24</th>
      <th>lag_168</th>
      <th>rolling_mean_3</th>
      <th>rolling_std_3</th>
      <th>rolling_mean_24</th>
      <th>rolling_std_24</th>
      <th>rolling_mean_168</th>
      <th>rolling_std_168</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>lag_1</th>
      <td>1.000</td>
      <td>0.934</td>
      <td>0.789</td>
      <td>0.790</td>
      <td>0.896</td>
      <td>0.944</td>
      <td>-0.075</td>
      <td>0.206</td>
      <td>0.267</td>
      <td>0.139</td>
      <td>0.135</td>
    </tr>
    <tr>
      <th>lag_2</th>
      <td>0.934</td>
      <td>1.000</td>
      <td>0.934</td>
      <td>0.665</td>
      <td>0.764</td>
      <td>0.995</td>
      <td>-0.010</td>
      <td>0.229</td>
      <td>0.273</td>
      <td>0.142</td>
      <td>0.140</td>
    </tr>
    <tr>
      <th>lag_3</th>
      <td>0.789</td>
      <td>0.934</td>
      <td>1.000</td>
      <td>0.507</td>
      <td>0.592</td>
      <td>0.944</td>
      <td>0.049</td>
      <td>0.248</td>
      <td>0.270</td>
      <td>0.143</td>
      <td>0.143</td>
    </tr>
    <tr>
      <th>lag_24</th>
      <td>0.790</td>
      <td>0.665</td>
      <td>0.507</td>
      <td>1.000</td>
      <td>0.858</td>
      <td>0.680</td>
      <td>-0.214</td>
      <td>0.209</td>
      <td>0.162</td>
      <td>0.156</td>
      <td>0.150</td>
    </tr>
    <tr>
      <th>lag_168</th>
      <td>0.896</td>
      <td>0.764</td>
      <td>0.592</td>
      <td>0.858</td>
      <td>1.000</td>
      <td>0.782</td>
      <td>-0.184</td>
      <td>0.166</td>
      <td>0.221</td>
      <td>0.160</td>
      <td>0.144</td>
    </tr>
    <tr>
      <th>rolling_mean_3</th>
      <td>0.944</td>
      <td>0.995</td>
      <td>0.944</td>
      <td>0.680</td>
      <td>0.782</td>
      <td>1.000</td>
      <td>-0.012</td>
      <td>0.237</td>
      <td>0.281</td>
      <td>0.147</td>
      <td>0.145</td>
    </tr>
    <tr>
      <th>rolling_std_3</th>
      <td>-0.075</td>
      <td>-0.010</td>
      <td>0.049</td>
      <td>-0.214</td>
      <td>-0.184</td>
      <td>-0.012</td>
      <td>1.000</td>
      <td>0.177</td>
      <td>0.183</td>
      <td>0.084</td>
      <td>0.092</td>
    </tr>
    <tr>
      <th>rolling_mean_24</th>
      <td>0.206</td>
      <td>0.229</td>
      <td>0.248</td>
      <td>0.209</td>
      <td>0.166</td>
      <td>0.237</td>
      <td>0.177</td>
      <td>1.000</td>
      <td>0.777</td>
      <td>0.549</td>
      <td>0.553</td>
    </tr>
    <tr>
      <th>rolling_std_24</th>
      <td>0.267</td>
      <td>0.273</td>
      <td>0.270</td>
      <td>0.162</td>
      <td>0.221</td>
      <td>0.281</td>
      <td>0.183</td>
      <td>0.777</td>
      <td>1.000</td>
      <td>0.480</td>
      <td>0.501</td>
    </tr>
    <tr>
      <th>rolling_mean_168</th>
      <td>0.139</td>
      <td>0.142</td>
      <td>0.143</td>
      <td>0.156</td>
      <td>0.160</td>
      <td>0.147</td>
      <td>0.084</td>
      <td>0.549</td>
      <td>0.480</td>
      <td>1.000</td>
      <td>0.968</td>
    </tr>
    <tr>
      <th>rolling_std_168</th>
      <td>0.135</td>
      <td>0.140</td>
      <td>0.143</td>
      <td>0.150</td>
      <td>0.144</td>
      <td>0.145</td>
      <td>0.092</td>
      <td>0.553</td>
      <td>0.501</td>
      <td>0.968</td>
      <td>1.000</td>
    </tr>
  </tbody>
</table>
</div>


## تحلیل الگوهای مسیرهای OD ادامه 


```python
print("OD features:")
print("Shape:", od_features.shape)
print("Index:", od_features.index.names)
print("Columns:", od_features.columns.tolist())
display(od_features.head())

print("\nClean OD features:")
print("Shape:", od_features_clean.shape)
print("Index:", od_features_clean.index.names)
print("Columns:", od_features_clean.columns.tolist())
display(od_features_clean.head())
```

    OD features:
    Shape: (37255, 5)
    Index: ['PULocationID', 'DOLocationID']
    Columns: ['od_trip_count', 'distance_sum', 'distance_n', 'avg_od_distance', 'od_trip_share']
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>od_trip_count</th>
      <th>distance_sum</th>
      <th>distance_n</th>
      <th>avg_od_distance</th>
      <th>od_trip_share</th>
    </tr>
    <tr>
      <th>PULocationID</th>
      <th>DOLocationID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">1</th>
      <th>1</th>
      <td>849</td>
      <td>431.37</td>
      <td>849</td>
      <td>0.508092</td>
      <td>8.885627e-05</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2</td>
      <td>37.12</td>
      <td>2</td>
      <td>18.560000</td>
      <td>2.093198e-07</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1</td>
      <td>16.10</td>
      <td>1</td>
      <td>16.100000</td>
      <td>1.046599e-07</td>
    </tr>
    <tr>
      <th>48</th>
      <td>4</td>
      <td>53.90</td>
      <td>4</td>
      <td>13.475000</td>
      <td>4.186397e-07</td>
    </tr>
    <tr>
      <th>50</th>
      <td>1</td>
      <td>17.50</td>
      <td>1</td>
      <td>17.500000</td>
      <td>1.046599e-07</td>
    </tr>
  </tbody>
</table>
</div>


    
    Clean OD features:
    Shape: (37255, 5)
    Index: ['PULocationID', 'DOLocationID']
    Columns: ['od_trip_count', 'od_trip_share', 'valid_distance_sum', 'valid_distance_n', 'avg_od_distance']
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>od_trip_count</th>
      <th>od_trip_share</th>
      <th>valid_distance_sum</th>
      <th>valid_distance_n</th>
      <th>avg_od_distance</th>
    </tr>
    <tr>
      <th>PULocationID</th>
      <th>DOLocationID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">1</th>
      <th>1</th>
      <td>849</td>
      <td>8.885627e-05</td>
      <td>431.37</td>
      <td>137.0</td>
      <td>3.148686</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2</td>
      <td>2.093198e-07</td>
      <td>37.12</td>
      <td>2.0</td>
      <td>18.560000</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1</td>
      <td>1.046599e-07</td>
      <td>16.10</td>
      <td>1.0</td>
      <td>16.100000</td>
    </tr>
    <tr>
      <th>48</th>
      <td>4</td>
      <td>4.186397e-07</td>
      <td>53.90</td>
      <td>3.0</td>
      <td>17.966667</td>
    </tr>
    <tr>
      <th>50</th>
      <td>1</td>
      <td>1.046599e-07</td>
      <td>17.50</td>
      <td>1.0</td>
      <td>17.500000</td>
    </tr>
  </tbody>
</table>
</div>



```python
# OD Route Popularity and Demand Concentration

od_analysis = (
    od_features_final
    .reset_index()
    .sort_values("od_trip_count", ascending=False)
    .reset_index(drop=True)
)

# Top 20 routes
top_20_od = od_analysis.head(20).copy()
top_20_od["share_pct"] = top_20_od["od_trip_share"] * 100

display(
    top_20_od[
        [
            "PULocationID",
            "DOLocationID",
            "od_trip_count",
            "share_pct",
            "avg_od_distance",
            "avg_duration_minutes",
            "avg_speed_mph"
        ]
    ].round(3)
)

# Cumulative demand concentration
total_trips = od_analysis["od_trip_count"].sum()
od_analysis["cumulative_share_pct"] = (
    od_analysis["od_trip_count"].cumsum() / total_trips * 100
)

concentration = pd.DataFrame({
    "Top N routes": [10, 20, 50, 100, 500, 1000],
    "Cumulative demand share (%)": [
        od_analysis.loc[n - 1, "cumulative_share_pct"]
        for n in [10, 20, 50, 100, 500, 1000]
    ]
})

display(concentration.round(2))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PULocationID</th>
      <th>DOLocationID</th>
      <th>od_trip_count</th>
      <th>share_pct</th>
      <th>avg_od_distance</th>
      <th>avg_duration_minutes</th>
      <th>avg_speed_mph</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>237</td>
      <td>236</td>
      <td>64384</td>
      <td>0.674</td>
      <td>1.059</td>
      <td>7.156</td>
      <td>9.907</td>
    </tr>
    <tr>
      <th>1</th>
      <td>236</td>
      <td>237</td>
      <td>57111</td>
      <td>0.598</td>
      <td>1.044</td>
      <td>7.965</td>
      <td>8.632</td>
    </tr>
    <tr>
      <th>2</th>
      <td>236</td>
      <td>236</td>
      <td>47509</td>
      <td>0.497</td>
      <td>0.642</td>
      <td>4.619</td>
      <td>8.907</td>
    </tr>
    <tr>
      <th>3</th>
      <td>237</td>
      <td>237</td>
      <td>44270</td>
      <td>0.463</td>
      <td>0.648</td>
      <td>5.462</td>
      <td>7.829</td>
    </tr>
    <tr>
      <th>4</th>
      <td>161</td>
      <td>237</td>
      <td>31118</td>
      <td>0.326</td>
      <td>1.072</td>
      <td>9.074</td>
      <td>7.958</td>
    </tr>
    <tr>
      <th>5</th>
      <td>142</td>
      <td>239</td>
      <td>27085</td>
      <td>0.283</td>
      <td>0.997</td>
      <td>6.176</td>
      <td>10.313</td>
    </tr>
    <tr>
      <th>6</th>
      <td>237</td>
      <td>161</td>
      <td>26836</td>
      <td>0.281</td>
      <td>1.063</td>
      <td>9.261</td>
      <td>7.727</td>
    </tr>
    <tr>
      <th>7</th>
      <td>239</td>
      <td>142</td>
      <td>26828</td>
      <td>0.281</td>
      <td>0.884</td>
      <td>6.024</td>
      <td>9.575</td>
    </tr>
    <tr>
      <th>8</th>
      <td>161</td>
      <td>236</td>
      <td>26504</td>
      <td>0.277</td>
      <td>1.946</td>
      <td>13.076</td>
      <td>10.007</td>
    </tr>
    <tr>
      <th>9</th>
      <td>239</td>
      <td>238</td>
      <td>25824</td>
      <td>0.270</td>
      <td>0.839</td>
      <td>5.112</td>
      <td>10.695</td>
    </tr>
    <tr>
      <th>10</th>
      <td>237</td>
      <td>162</td>
      <td>24058</td>
      <td>0.252</td>
      <td>0.987</td>
      <td>8.367</td>
      <td>8.162</td>
    </tr>
    <tr>
      <th>11</th>
      <td>141</td>
      <td>236</td>
      <td>23182</td>
      <td>0.243</td>
      <td>1.084</td>
      <td>7.623</td>
      <td>9.366</td>
    </tr>
    <tr>
      <th>12</th>
      <td>264</td>
      <td>264</td>
      <td>21940</td>
      <td>0.230</td>
      <td>3.122</td>
      <td>13.787</td>
      <td>11.471</td>
    </tr>
    <tr>
      <th>13</th>
      <td>236</td>
      <td>161</td>
      <td>21554</td>
      <td>0.226</td>
      <td>1.930</td>
      <td>14.017</td>
      <td>9.362</td>
    </tr>
    <tr>
      <th>14</th>
      <td>238</td>
      <td>239</td>
      <td>21248</td>
      <td>0.222</td>
      <td>0.792</td>
      <td>5.089</td>
      <td>10.061</td>
    </tr>
    <tr>
      <th>15</th>
      <td>186</td>
      <td>230</td>
      <td>21139</td>
      <td>0.221</td>
      <td>1.067</td>
      <td>13.439</td>
      <td>5.899</td>
    </tr>
    <tr>
      <th>16</th>
      <td>132</td>
      <td>132</td>
      <td>21059</td>
      <td>0.220</td>
      <td>2.530</td>
      <td>4.454</td>
      <td>19.339</td>
    </tr>
    <tr>
      <th>17</th>
      <td>236</td>
      <td>141</td>
      <td>20861</td>
      <td>0.218</td>
      <td>1.116</td>
      <td>8.608</td>
      <td>8.444</td>
    </tr>
    <tr>
      <th>18</th>
      <td>142</td>
      <td>238</td>
      <td>20320</td>
      <td>0.213</td>
      <td>1.507</td>
      <td>8.132</td>
      <td>11.917</td>
    </tr>
    <tr>
      <th>19</th>
      <td>236</td>
      <td>239</td>
      <td>20238</td>
      <td>0.212</td>
      <td>1.449</td>
      <td>8.739</td>
      <td>10.630</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Top N routes</th>
      <th>Cumulative demand share (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>3.95</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20</td>
      <td>6.21</td>
    </tr>
    <tr>
      <th>2</th>
      <td>50</td>
      <td>11.81</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100</td>
      <td>19.17</td>
    </tr>
    <tr>
      <th>4</th>
      <td>500</td>
      <td>51.21</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1000</td>
      <td>70.53</td>
    </tr>
  </tbody>
</table>
</div>



```python
# OD Distance Analysis

od_distance = od_features_final.copy()

total_od_trips = od_distance["od_trip_count"].sum()
valid_distance_trips = od_distance["valid_distance_n"].sum()

print("Total trips:", total_od_trips)
print("Trips with valid distance:", int(valid_distance_trips))
print(
    "Valid distance coverage (%):",
    round(valid_distance_trips / total_od_trips * 100, 2)
)

# Route-level distance statistics
stable_routes = od_distance.loc[
    od_distance["valid_distance_n"] >= 30,
    "avg_od_distance"
].dropna()

print("\nRoutes with at least 30 valid distance observations:",
      len(stable_routes))

display(
    stable_routes.describe(
        percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    ).to_frame("avg_od_distance")
)

# Trip-weighted average OD distance
weighted_avg_distance = (
    od_distance["valid_distance_sum"].sum()
    / valid_distance_trips
)

print("\nTrip-weighted average distance (miles):",
      round(weighted_avg_distance, 3))
```

    Total trips: 9554756
    Trips with valid distance: 9338740
    Valid distance coverage (%): 97.74
    
    Routes with at least 30 valid distance observations: 7307
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>avg_od_distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7307.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.735291</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4.407141</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.483592</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.512253</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.652481</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.586783</td>
    </tr>
    <tr>
      <th>90%</th>
      <td>11.166013</td>
    </tr>
    <tr>
      <th>95%</th>
      <td>14.823250</td>
    </tr>
    <tr>
      <th>99%</th>
      <td>20.882703</td>
    </tr>
    <tr>
      <th>max</th>
      <td>37.228529</td>
    </tr>
  </tbody>
</table>
</div>


    
    Trip-weighted average distance (miles): 3.296
    

## ﻧﺎﻣﺰﺩﻫﺎﯼ ﻧﺎﻫﻨﺠﺎﺭﯼ


```python

# Anomaly Analysis – Hourly Demand
# Robust comparison by day of week and hour

demand_anomaly_df = hourly_df.copy()

if "pickup_hour" in demand_anomaly_df.columns:
    demand_anomaly_df["pickup_hour"] = pd.to_datetime(
        demand_anomaly_df["pickup_hour"]
    )
    demand_anomaly_df = demand_anomaly_df.set_index("pickup_hour")
else:
    demand_anomaly_df.index = pd.to_datetime(demand_anomaly_df.index)

demand_anomaly_df = demand_anomaly_df.sort_index()

# Compare each observation with the same weekday-hour pattern
demand_anomaly_df["weekday"] = demand_anomaly_df.index.dayofweek
demand_anomaly_df["hour"] = demand_anomaly_df.index.hour
demand_anomaly_df["weekday_hour"] = (
    demand_anomaly_df["weekday"] * 24
    + demand_anomaly_df["hour"]
)

grouped = demand_anomaly_df.groupby("weekday_hour")["trip_count"]

demand_anomaly_df["expected_demand"] = (
    demand_anomaly_df["weekday_hour"].map(grouped.median())
)

demand_anomaly_df["demand_mad"] = (
    demand_anomaly_df["weekday_hour"].map(
        grouped.apply(
            lambda x: (x - x.median()).abs().median()
        )
    )
)

# Robust z-score
demand_anomaly_df["robust_z_score"] = np.where(
    demand_anomaly_df["demand_mad"] > 0,
    0.6745 * (
        demand_anomaly_df["trip_count"]
        - demand_anomaly_df["expected_demand"]
    ) / demand_anomaly_df["demand_mad"],
    np.nan
)

# Flag candidates, not confirmed errors
demand_anomaly_df["anomaly_candidate"] = (
    demand_anomaly_df["robust_z_score"].abs() >= 3.5
)

demand_anomaly_df["anomaly_type"] = np.select(
    [
        demand_anomaly_df["robust_z_score"] >= 3.5,
        demand_anomaly_df["robust_z_score"] <= -3.5
    ],
    [
        "Unusually high demand",
        "Unusually low demand"
    ],
    default="Normal"
)

candidates = demand_anomaly_df[
    demand_anomaly_df["anomaly_candidate"]
].copy()

display(
    candidates[
        [
            "trip_count",
            "expected_demand",
            "robust_z_score",
            "anomaly_type"
        ]
    ].sort_values(
        "robust_z_score",
        key=lambda x: x.abs(),
        ascending=False
    ).head(30)
)

print("Total demand anomaly candidates:", len(candidates))
print("\nCandidate types:")
display(candidates["anomaly_type"].value_counts())

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trip_count</th>
      <th>expected_demand</th>
      <th>robust_z_score</th>
      <th>anomaly_type</th>
    </tr>
    <tr>
      <th>pickup_hour</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2024-01-01 03:00:00-05:00</th>
      <td>4936</td>
      <td>259.0</td>
      <td>67.119926</td>
      <td>Unusually high demand</td>
    </tr>
    <tr>
      <th>2024-01-01 01:00:00-05:00</th>
      <td>7355</td>
      <td>610.0</td>
      <td>49.451114</td>
      <td>Unusually high demand</td>
    </tr>
    <tr>
      <th>2024-01-01 02:00:00-05:00</th>
      <td>6220</td>
      <td>338.0</td>
      <td>47.800108</td>
      <td>Unusually high demand</td>
    </tr>
    <tr>
      <th>2024-01-01 04:00:00-05:00</th>
      <td>3188</td>
      <td>341.0</td>
      <td>22.329087</td>
      <td>Unusually high demand</td>
    </tr>
    <tr>
      <th>2024-01-01 08:00:00-05:00</th>
      <td>1204</td>
      <td>4791.0</td>
      <td>-12.036973</td>
      <td>Unusually low demand</td>
    </tr>
    <tr>
      <th>2024-01-01 00:00:00-05:00</th>
      <td>6596</td>
      <td>1339.0</td>
      <td>11.979211</td>
      <td>Unusually high demand</td>
    </tr>
    <tr>
      <th>2024-01-01 05:00:00-05:00</th>
      <td>1522</td>
      <td>652.0</td>
      <td>10.117500</td>
      <td>Unusually high demand</td>
    </tr>
    <tr>
      <th>2024-01-01 09:00:00-05:00</th>
      <td>1430</td>
      <td>4889.0</td>
      <td>-9.113654</td>
      <td>Unusually low demand</td>
    </tr>
    <tr>
      <th>2024-02-19 08:00:00-05:00</th>
      <td>2358</td>
      <td>4791.0</td>
      <td>-8.164470</td>
      <td>Unusually low demand</td>
    </tr>
    <tr>
      <th>2024-01-15 08:00:00-05:00</th>
      <td>2440</td>
      <td>4791.0</td>
      <td>-7.889301</td>
      <td>Unusually low demand</td>
    </tr>
    <tr>
      <th>2024-01-07 11:00:00-05:00</th>
      <td>4087</td>
      <td>4817.0</td>
      <td>-7.575154</td>
      <td>Unusually low demand</td>
    </tr>
    <tr>
      <th>2024-01-06 02:00:00-05:00</th>
      <td>1847</td>
      <td>3386.0</td>
      <td>-7.522141</td>
      <td>Unusually low demand</td>
    </tr>
    <tr>
      <th>2024-02-13 11:00:00-05:00</th>
      <td>3161</td>
      <td>5312.0</td>
      <td>-7.147042</td>
      <td>Unusually low demand</td>
    </tr>
    <tr>
      <th>2024-02-13 14:00:00-05:00</th>
      <td>4105</td>
      <td>6167.0</td>
      <td>-7.132405</td>
      <td>Unusually low demand</td>
    </tr>
    <tr>
      <th>2024-02-13 16:00:00-05:00</th>
      <td>4284</td>
      <td>6275.0</td>
      <td>-6.886818</td>
      <td>Unusually low demand</td>
    </tr>
    <tr>
      <th>2024-01-01 10:00:00-05:00</th>
      <td>2263</td>
      <td>4581.0</td>
      <td>-6.827472</td>
      <td>Unusually low demand</td>
    </tr>
    <tr>
      <th>2024-02-27 18:00:00-05:00</th>
      <td>11445</td>
      <td>7805.0</td>
      <td>6.726521</td>
      <td>Unusually high demand</td>
    </tr>
    <tr>
      <th>2024-03-10 03:00:00-04:00</th>
      <td>4559</td>
      <td>2494.0</td>
      <td>6.696358</td>
      <td>Unusually high demand</td>
    </tr>
    <tr>
      <th>2024-02-13 15:00:00-05:00</th>
      <td>4062</td>
      <td>6374.0</td>
      <td>-6.339203</td>
      <td>Unusually low demand</td>
    </tr>
    <tr>
      <th>2024-02-13 13:00:00-05:00</th>
      <td>3737</td>
      <td>5575.0</td>
      <td>-6.137282</td>
      <td>Unusually low demand</td>
    </tr>
    <tr>
      <th>2024-03-10 15:00:00-04:00</th>
      <td>6994</td>
      <td>5647.0</td>
      <td>6.057010</td>
      <td>Unusually high demand</td>
    </tr>
    <tr>
      <th>2024-01-15 07:00:00-05:00</th>
      <td>1361</td>
      <td>3496.0</td>
      <td>-5.737281</td>
      <td>Unusually low demand</td>
    </tr>
    <tr>
      <th>2024-02-11 23:00:00-05:00</th>
      <td>6021</td>
      <td>2231.0</td>
      <td>5.731738</td>
      <td>Unusually high demand</td>
    </tr>
    <tr>
      <th>2024-01-01 07:00:00-05:00</th>
      <td>1378</td>
      <td>3496.0</td>
      <td>-5.691598</td>
      <td>Unusually low demand</td>
    </tr>
    <tr>
      <th>2024-02-13 12:00:00-05:00</th>
      <td>3449</td>
      <td>5636.0</td>
      <td>-5.545607</td>
      <td>Unusually low demand</td>
    </tr>
    <tr>
      <th>2024-01-07 10:00:00-05:00</th>
      <td>3169</td>
      <td>4147.0</td>
      <td>-5.543370</td>
      <td>Unusually low demand</td>
    </tr>
    <tr>
      <th>2024-03-15 14:00:00-04:00</th>
      <td>7220</td>
      <td>6358.0</td>
      <td>5.383509</td>
      <td>Unusually high demand</td>
    </tr>
    <tr>
      <th>2024-02-19 07:00:00-05:00</th>
      <td>1499</td>
      <td>3496.0</td>
      <td>-5.366440</td>
      <td>Unusually low demand</td>
    </tr>
    <tr>
      <th>2024-03-10 04:00:00-04:00</th>
      <td>2764</td>
      <td>1424.0</td>
      <td>5.254826</td>
      <td>Unusually high demand</td>
    </tr>
    <tr>
      <th>2024-03-17 07:00:00-04:00</th>
      <td>1697</td>
      <td>1162.0</td>
      <td>5.229819</td>
      <td>Unusually high demand</td>
    </tr>
  </tbody>
</table>
</div>


    Total demand anomaly candidates: 67
    
    Candidate types:
    


    anomaly_type
    Unusually low demand     41
    Unusually high demand    26
    Name: count, dtype: int64



```python

# Anomaly Analysis – Step 2: Daily context of flagged hours

daily_totals = demand_anomaly_df["trip_count"].groupby(
    demand_anomaly_df.index.date
).sum()

daily_summary = pd.DataFrame({
    "daily_trip_count": daily_totals
})

daily_summary.index = pd.to_datetime(daily_summary.index)
daily_summary.index.name = "date"

# Count high and low anomaly candidates per day
candidate_daily = candidates.copy()
candidate_daily["date"] = pd.to_datetime(
    candidate_daily.index.date
)

anomaly_counts = candidate_daily.groupby("date").agg(
    total_candidates=("anomaly_candidate", "size"),
    high_candidates=("anomaly_type",
                     lambda x: (x == "Unusually high demand").sum()),
    low_candidates=("anomaly_type",
                    lambda x: (x == "Unusually low demand").sum())
)

daily_summary = daily_summary.join(anomaly_counts)
daily_summary[["total_candidates", "high_candidates", "low_candidates"]] = (
    daily_summary[["total_candidates", "high_candidates", "low_candidates"]]
    .fillna(0).astype(int)
)

# Compare each flagged date with its previous and following day
flagged_dates = daily_summary[
    daily_summary["total_candidates"] > 0
].index

context_rows = []

for date in flagged_dates:
    row = {
        "date": date,
        "weekday": date.day_name(),
        "daily_trip_count": daily_summary.loc[date, "daily_trip_count"],
        "anomaly_hours": daily_summary.loc[date, "total_candidates"],
        "high_candidates": daily_summary.loc[date, "high_candidates"],
        "low_candidates": daily_summary.loc[date, "low_candidates"],
        "previous_day_trips": daily_summary["daily_trip_count"].get(
            date - pd.Timedelta(days=1), np.nan
        ),
        "next_day_trips": daily_summary["daily_trip_count"].get(
            date + pd.Timedelta(days=1), np.nan
        )
    }
    context_rows.append(row)

anomaly_date_summary = pd.DataFrame(context_rows)

display(
    anomaly_date_summary.sort_values(
        ["anomaly_hours", "date"],
        ascending=[False, True]
    )
)

print("Number of dates containing anomaly candidates:",
      len(anomaly_date_summary))

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>weekday</th>
      <th>daily_trip_count</th>
      <th>anomaly_hours</th>
      <th>high_candidates</th>
      <th>low_candidates</th>
      <th>previous_day_trips</th>
      <th>next_day_trips</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-01-01</td>
      <td>Monday</td>
      <td>81013</td>
      <td>16</td>
      <td>6</td>
      <td>10</td>
      <td>NaN</td>
      <td>75519</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2024-02-13</td>
      <td>Tuesday</td>
      <td>69067</td>
      <td>9</td>
      <td>0</td>
      <td>9</td>
      <td>92090.0</td>
      <td>121771</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-01-07</td>
      <td>Sunday</td>
      <td>67543</td>
      <td>8</td>
      <td>0</td>
      <td>8</td>
      <td>97117.0</td>
      <td>80034</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-01-15</td>
      <td>Monday</td>
      <td>77033</td>
      <td>4</td>
      <td>0</td>
      <td>4</td>
      <td>94420.0</td>
      <td>93057</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2024-02-19</td>
      <td>Monday</td>
      <td>76073</td>
      <td>4</td>
      <td>0</td>
      <td>4</td>
      <td>93788.0</td>
      <td>97086</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2024-03-09</td>
      <td>Saturday</td>
      <td>140493</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>119534.0</td>
      <td>104486</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2024-03-10</td>
      <td>Sunday</td>
      <td>104486</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>140493.0</td>
      <td>98319</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2024-03-17</td>
      <td>Sunday</td>
      <td>103415</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>131432.0</td>
      <td>94988</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-01-06</td>
      <td>Saturday</td>
      <td>97117</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>103178.0</td>
      <td>67543</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2024-01-19</td>
      <td>Friday</td>
      <td>95951</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>110358.0</td>
      <td>108768</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2024-02-27</td>
      <td>Tuesday</td>
      <td>118614</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>86321.0</td>
      <td>112463</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2024-03-15</td>
      <td>Friday</td>
      <td>128813</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>137060.0</td>
      <td>131432</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-01-02</td>
      <td>Tuesday</td>
      <td>75519</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>81013.0</td>
      <td>82427</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2024-01-20</td>
      <td>Saturday</td>
      <td>108768</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>95951.0</td>
      <td>84502</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2024-02-11</td>
      <td>Sunday</td>
      <td>96566</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>111655.0</td>
      <td>92090</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2024-02-14</td>
      <td>Wednesday</td>
      <td>121771</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>69067.0</td>
      <td>122956</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2024-03-06</td>
      <td>Wednesday</td>
      <td>132881</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>112562.0</td>
      <td>122119</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2024-03-16</td>
      <td>Saturday</td>
      <td>131432</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>128813.0</td>
      <td>103415</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2024-03-23</td>
      <td>Saturday</td>
      <td>121966</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>116847.0</td>
      <td>101913</td>
    </tr>
  </tbody>
</table>
</div>


    Number of dates containing anomaly candidates: 19
    


```python
# ممیزی هدفمند کرایه

# Anomaly Analysis – Fare Audit
# Reuse existing monthly processed files

fare_data = []

for file in MONTHLY_FILES:
    temp = pd.read_parquet(
        PROCESSED_DIR / file,
        columns=["fare_amount", "total_amount"]
    )
    temp["month_file"] = file
    fare_data.append(temp)

fare_df = pd.concat(fare_data, ignore_index=True)

# Overall distribution
fare_summary = fare_df[["fare_amount", "total_amount"]].describe(
    percentiles=[0.01, 0.05, 0.50, 0.95, 0.99, 0.999]
).T

print("Overall fare distribution:")
display(fare_summary)

# Count suspicious values
for col in ["fare_amount", "total_amount"]:
    values = fare_df[col]

    print(f"\n{col}")
    print("Missing:", values.isna().sum())
    print("Zero:", (values == 0).sum())
    print("Negative:", (values < 0).sum())
    print("Above 99th percentile:",
          (values > values.quantile(0.99)).sum())
    print("Above 99.9th percentile:",
          (values > values.quantile(0.999)).sum())
    print("Maximum:", values.max())

# Monthly comparison
monthly_fare_summary = fare_df.groupby("month_file").agg(
    records=("fare_amount", "size"),
    fare_missing=("fare_amount", lambda x: x.isna().sum()),
    fare_zero=("fare_amount", lambda x: (x == 0).sum()),
    fare_negative=("fare_amount", lambda x: (x < 0).sum()),
    fare_median=("fare_amount", "median"),
    fare_p99=("fare_amount", lambda x: x.quantile(0.99)),
    fare_max=("fare_amount", "max"),
    total_amount_negative=("total_amount", lambda x: (x < 0).sum()),
    total_amount_zero=("total_amount", lambda x: (x == 0).sum())
)

display(monthly_fare_summary)

```

    Overall fare distribution:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>1%</th>
      <th>5%</th>
      <th>50%</th>
      <th>95%</th>
      <th>99%</th>
      <th>99.9%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>fare_amount</th>
      <td>9554756.0</td>
      <td>18.325119</td>
      <td>18.544973</td>
      <td>-999.0</td>
      <td>-5.1</td>
      <td>5.1</td>
      <td>12.8</td>
      <td>59.70</td>
      <td>75.00</td>
      <td>140.00000</td>
      <td>9792.0</td>
    </tr>
    <tr>
      <th>total_amount</th>
      <td>9554756.0</td>
      <td>26.865401</td>
      <td>23.050198</td>
      <td>-1000.0</td>
      <td>-9.8</td>
      <td>10.8</td>
      <td>20.2</td>
      <td>79.39</td>
      <td>103.34</td>
      <td>170.40245</td>
      <td>9792.0</td>
    </tr>
  </tbody>
</table>
</div>


    
    fare_amount
    Missing: 0
    Zero: 3029
    Negative: 136566
    Above 99th percentile: 94949
    Above 99.9th percentile: 9323
    Maximum: 9792.0
    
    total_amount
    Missing: 0
    Zero: 1292
    Negative: 115894
    Above 99th percentile: 94372
    Above 99.9th percentile: 9555
    Maximum: 9792.0
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>records</th>
      <th>fare_missing</th>
      <th>fare_zero</th>
      <th>fare_negative</th>
      <th>fare_median</th>
      <th>fare_p99</th>
      <th>fare_max</th>
      <th>total_amount_negative</th>
      <th>total_amount_zero</th>
    </tr>
    <tr>
      <th>month_file</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>trip_features_2024-01.parquet</th>
      <td>2964609</td>
      <td>0</td>
      <td>893</td>
      <td>37447</td>
      <td>12.8</td>
      <td>75.8</td>
      <td>5000.0</td>
      <td>35503</td>
      <td>416</td>
    </tr>
    <tr>
      <th>trip_features_2024-02.parquet</th>
      <td>3007523</td>
      <td>0</td>
      <td>844</td>
      <td>40655</td>
      <td>12.8</td>
      <td>73.4</td>
      <td>9792.0</td>
      <td>36080</td>
      <td>438</td>
    </tr>
    <tr>
      <th>trip_features_2024-03.parquet</th>
      <td>3582624</td>
      <td>0</td>
      <td>1292</td>
      <td>58464</td>
      <td>13.5</td>
      <td>75.0</td>
      <td>900.0</td>
      <td>44311</td>
      <td>438</td>
    </tr>
  </tbody>
</table>
</div>



```python

# Fare Anomaly Analysis – Inspect actual suspicious records

fare_samples = []

columns_to_read = [
    "pickup_time_local",
    "dropoff_time_local",
    "trip_distance",
    "fare_amount",
    "total_amount",
    "passenger_count",
    "duration_minutes",
    "speed_mph"
]

for file in MONTHLY_FILES:
    temp = pd.read_parquet(PROCESSED_DIR / file, columns=columns_to_read)

    temp["month_file"] = file

    groups = {
        "Negative fare": temp[temp["fare_amount"] < 0],
        "Negative total": temp[temp["total_amount"] < 0],
        "Zero fare": temp[temp["fare_amount"] == 0],
        "Extreme fare": temp[
            temp["fare_amount"] > fare_df["fare_amount"].quantile(0.999)
        ],
        "Extreme total": temp[
            temp["total_amount"] > fare_df["total_amount"].quantile(0.999)
        ]
    }

    for label, subset in groups.items():
        if len(subset) > 0:
            sample = (
                subset.sample(n=min(5, len(subset)), random_state=42)
                if label in ["Negative fare", "Negative total", "Zero fare"]
                else subset.nlargest(5, "fare_amount" if label == "Extreme fare"
                                     else "total_amount")
            )
            sample = sample.copy()
            sample["anomaly_type"] = label
            fare_samples.append(sample)

fare_samples_df = pd.concat(fare_samples, ignore_index=True)

display(
    fare_samples_df.sort_values(
        ["anomaly_type", "fare_amount"],
        ascending=[True, True]
    )
)

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pickup_time_local</th>
      <th>dropoff_time_local</th>
      <th>trip_distance</th>
      <th>fare_amount</th>
      <th>total_amount</th>
      <th>passenger_count</th>
      <th>duration_minutes</th>
      <th>speed_mph</th>
      <th>month_file</th>
      <th>anomaly_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>68</th>
      <td>2024-03-03 00:20:36-05:00</td>
      <td>2024-03-03 03:04:29-05:00</td>
      <td>159.74</td>
      <td>800.0</td>
      <td>807.75</td>
      <td>2.0</td>
      <td>163.883333</td>
      <td>58.483067</td>
      <td>trip_features_2024-03.parquet</td>
      <td>Extreme fare</td>
    </tr>
    <tr>
      <th>69</th>
      <td>2024-03-28 11:39:41-04:00</td>
      <td>2024-03-28 11:40:57-04:00</td>
      <td>0.00</td>
      <td>800.0</td>
      <td>801.00</td>
      <td>1.0</td>
      <td>1.266667</td>
      <td>0.000000</td>
      <td>trip_features_2024-03.parquet</td>
      <td>Extreme fare</td>
    </tr>
    <tr>
      <th>67</th>
      <td>2024-03-10 07:29:44-04:00</td>
      <td>2024-03-10 07:31:26-04:00</td>
      <td>0.10</td>
      <td>840.0</td>
      <td>841.00</td>
      <td>1.0</td>
      <td>1.700000</td>
      <td>3.529412</td>
      <td>trip_features_2024-03.parquet</td>
      <td>Extreme fare</td>
    </tr>
    <tr>
      <th>66</th>
      <td>2024-03-04 17:15:33-05:00</td>
      <td>2024-03-04 17:15:41-05:00</td>
      <td>0.00</td>
      <td>850.0</td>
      <td>851.00</td>
      <td>4.0</td>
      <td>0.133333</td>
      <td>0.000000</td>
      <td>trip_features_2024-03.parquet</td>
      <td>Extreme fare</td>
    </tr>
    <tr>
      <th>44</th>
      <td>2024-02-15 11:56:21-05:00</td>
      <td>2024-02-15 11:57:59-05:00</td>
      <td>0.00</td>
      <td>900.0</td>
      <td>901.00</td>
      <td>1.0</td>
      <td>1.633333</td>
      <td>0.000000</td>
      <td>trip_features_2024-02.parquet</td>
      <td>Extreme fare</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>60</th>
      <td>2024-03-12 20:22:53-04:00</td>
      <td>2024-03-12 20:23:29-04:00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
      <td>0.600000</td>
      <td>0.000000</td>
      <td>trip_features_2024-03.parquet</td>
      <td>Zero fare</td>
    </tr>
    <tr>
      <th>61</th>
      <td>2024-03-16 11:13:35-04:00</td>
      <td>2024-03-16 11:15:42-04:00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>2.116667</td>
      <td>0.000000</td>
      <td>trip_features_2024-03.parquet</td>
      <td>Zero fare</td>
    </tr>
    <tr>
      <th>62</th>
      <td>2024-03-19 21:12:41-04:00</td>
      <td>2024-03-19 21:12:59-04:00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
      <td>0.300000</td>
      <td>0.000000</td>
      <td>trip_features_2024-03.parquet</td>
      <td>Zero fare</td>
    </tr>
    <tr>
      <th>63</th>
      <td>2024-03-15 02:55:02-04:00</td>
      <td>2024-03-15 02:57:51-04:00</td>
      <td>0.50</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>2.0</td>
      <td>2.816667</td>
      <td>10.650888</td>
      <td>trip_features_2024-03.parquet</td>
      <td>Zero fare</td>
    </tr>
    <tr>
      <th>64</th>
      <td>2024-03-10 09:56:35-04:00</td>
      <td>2024-03-10 09:56:35-04:00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>trip_features_2024-03.parquet</td>
      <td>Zero fare</td>
    </tr>
  </tbody>
</table>
<p>75 rows × 10 columns</p>
</div>



```python

# Targeted Fare Consistency Audit – Full Dataset

audit_parts = []

for file in MONTHLY_FILES:
    temp = pd.read_parquet(
        PROCESSED_DIR / file,
        columns=[
            "pickup_time_local", "trip_distance",
            "fare_amount", "total_amount",
            "duration_minutes", "speed_mph"
        ]
    )
    temp["month_file"] = file
    audit_parts.append(temp)

fare_audit = pd.concat(audit_parts, ignore_index=True)

fare_threshold = fare_df["fare_amount"].quantile(0.999)

extreme = fare_audit["fare_amount"] > fare_threshold

checks = {
    "Negative fare, non-negative total":
        (fare_audit["fare_amount"] < 0) &
        (fare_audit["total_amount"] >= 0),

    "Non-negative fare, negative total":
        (fare_audit["fare_amount"] >= 0) &
        (fare_audit["total_amount"] < 0),

    "Extreme fare with zero distance":
        extreme & (fare_audit["trip_distance"] == 0),

    "Extreme fare with distance above 100 miles":
        extreme & (fare_audit["trip_distance"] > 100),

    "Extreme fare with duration <= 2 minutes":
        extreme & (fare_audit["duration_minutes"] <= 2),

    "Extreme fare with positive distance and duration":
        extreme & (fare_audit["trip_distance"] > 0) &
        (fare_audit["duration_minutes"] > 0)
}

print("Fare threshold (99.9th percentile):", fare_threshold)
print("\nConsistency checks:")
for label, mask in checks.items():
    print(f"{label}: {mask.sum():,}")

# Inspect the highest fare records
display(
    fare_audit.nlargest(20, "fare_amount")[
        [
            "pickup_time_local", "trip_distance",
            "fare_amount", "total_amount",
            "duration_minutes", "speed_mph",
            "month_file"
        ]
    ]
)

```

    Fare threshold (99.9th percentile): 140.0
    
    Consistency checks:
    Negative fare, non-negative total: 21,013
    Non-negative fare, negative total: 341
    Extreme fare with zero distance: 1,251
    Extreme fare with distance above 100 miles: 74
    Extreme fare with duration <= 2 minutes: 1,345
    Extreme fare with positive distance and duration: 8,072
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pickup_time_local</th>
      <th>trip_distance</th>
      <th>fare_amount</th>
      <th>total_amount</th>
      <th>duration_minutes</th>
      <th>speed_mph</th>
      <th>month_file</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5960537</th>
      <td>2024-02-29 07:56:45-05:00</td>
      <td>0.00</td>
      <td>9792.0</td>
      <td>9792.00</td>
      <td>5.166667</td>
      <td>0.000000</td>
      <td>trip_features_2024-02.parquet</td>
    </tr>
    <tr>
      <th>1714855</th>
      <td>2024-01-20 11:18:47-05:00</td>
      <td>0.00</td>
      <td>5000.0</td>
      <td>5000.00</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>1714856</th>
      <td>2024-01-20 11:19:33-05:00</td>
      <td>0.00</td>
      <td>5000.0</td>
      <td>5000.00</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>1714857</th>
      <td>2024-01-20 11:20:15-05:00</td>
      <td>0.00</td>
      <td>2500.0</td>
      <td>2500.00</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>1714859</th>
      <td>2024-01-20 11:27:48-05:00</td>
      <td>0.00</td>
      <td>2500.0</td>
      <td>2500.00</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>2084546</th>
      <td>2024-01-24 13:44:43-05:00</td>
      <td>0.00</td>
      <td>2500.0</td>
      <td>2500.00</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>1159580</th>
      <td>2024-01-14 10:08:11-05:00</td>
      <td>31.95</td>
      <td>2221.3</td>
      <td>2225.30</td>
      <td>3106.183333</td>
      <td>0.617156</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>76878</th>
      <td>2024-01-02 07:50:08-05:00</td>
      <td>233.25</td>
      <td>1616.5</td>
      <td>1617.50</td>
      <td>219.350000</td>
      <td>63.802143</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>1714858</th>
      <td>2024-01-20 11:22:27-05:00</td>
      <td>0.00</td>
      <td>1000.0</td>
      <td>1000.00</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>4603808</th>
      <td>2024-02-17 19:34:16-05:00</td>
      <td>0.00</td>
      <td>999.0</td>
      <td>1000.00</td>
      <td>1.166667</td>
      <td>0.000000</td>
      <td>trip_features_2024-02.parquet</td>
    </tr>
    <tr>
      <th>4670321</th>
      <td>2024-02-18 14:13:36-05:00</td>
      <td>1.07</td>
      <td>999.0</td>
      <td>1000.00</td>
      <td>2.700000</td>
      <td>23.777778</td>
      <td>trip_features_2024-02.parquet</td>
    </tr>
    <tr>
      <th>5441919</th>
      <td>2024-02-26 19:18:19-05:00</td>
      <td>36.20</td>
      <td>924.9</td>
      <td>931.40</td>
      <td>1214.800000</td>
      <td>1.787949</td>
      <td>trip_features_2024-02.parquet</td>
    </tr>
    <tr>
      <th>507301</th>
      <td>2024-01-06 21:01:38-05:00</td>
      <td>142.62</td>
      <td>912.3</td>
      <td>940.93</td>
      <td>160.100000</td>
      <td>53.449094</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>4352337</th>
      <td>2024-02-15 11:56:21-05:00</td>
      <td>0.00</td>
      <td>900.0</td>
      <td>901.00</td>
      <td>1.633333</td>
      <td>0.000000</td>
      <td>trip_features_2024-02.parquet</td>
    </tr>
    <tr>
      <th>4603810</th>
      <td>2024-02-17 19:47:10-05:00</td>
      <td>0.00</td>
      <td>900.0</td>
      <td>901.00</td>
      <td>0.133333</td>
      <td>0.000000</td>
      <td>trip_features_2024-02.parquet</td>
    </tr>
    <tr>
      <th>6141451</th>
      <td>2024-03-02 16:02:35-05:00</td>
      <td>181.50</td>
      <td>900.0</td>
      <td>951.26</td>
      <td>206.916667</td>
      <td>52.629883</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
    <tr>
      <th>1916191</th>
      <td>2024-01-22 16:40:43-05:00</td>
      <td>157.25</td>
      <td>899.0</td>
      <td>900.00</td>
      <td>150.616667</td>
      <td>62.642470</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>4794502</th>
      <td>2024-02-20 08:51:37-05:00</td>
      <td>0.00</td>
      <td>895.0</td>
      <td>898.50</td>
      <td>1.400000</td>
      <td>0.000000</td>
      <td>trip_features_2024-02.parquet</td>
    </tr>
    <tr>
      <th>4114272</th>
      <td>2024-02-12 17:48:54-05:00</td>
      <td>136.63</td>
      <td>877.3</td>
      <td>894.49</td>
      <td>153.216667</td>
      <td>53.504623</td>
      <td>trip_features_2024-02.parquet</td>
    </tr>
    <tr>
      <th>6335537</th>
      <td>2024-03-04 17:15:33-05:00</td>
      <td>0.00</td>
      <td>850.0</td>
      <td>851.00</td>
      <td>0.133333</td>
      <td>0.000000</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
  </tbody>
</table>
</div>



```python

# Joint Fare-Distance-Duration Anomaly Analysis

inspection_columns = [
    "pickup_time_local", "dropoff_time_local",
    "PULocationID", "DOLocationID",
    "trip_distance", "fare_amount", "total_amount",
    "duration_minutes", "speed_mph"
]

inspection_results = []

for file in MONTHLY_FILES:
    temp = pd.read_parquet(
        PROCESSED_DIR / file,
        columns=inspection_columns
    )

    temp["month_file"] = file

    high_fare = temp["fare_amount"] > 140

    valid_distance = (
        (temp["trip_distance"] > 0) &
        (temp["trip_distance"] <= 100)
    )

    valid_duration = (
        (temp["duration_minutes"] > 0) &
        (temp["duration_minutes"] <= 180)
    )

    temp["fare_category"] = np.select(
        [
            high_fare & ~valid_distance,
            high_fare & valid_distance & ~valid_duration,
            high_fare & valid_distance & valid_duration
        ],
        [
            "High fare + invalid distance",
            "High fare + invalid duration",
            "High fare + valid distance and duration"
        ],
        default="Other"
    )

    inspection_results.append(temp)

fare_inspection = pd.concat(inspection_results, ignore_index=True)

print("High-fare classification:")
display(
    fare_inspection[
        fare_inspection["fare_amount"] > 140
    ]["fare_category"].value_counts().to_frame("records")
)

print("\nNegative fare with non-negative total:")
display(
    fare_inspection[
        (fare_inspection["fare_amount"] < 0) &
        (fare_inspection["total_amount"] >= 0)
    ].sample(n=10, random_state=42)
)

print("\nNon-negative fare with negative total:")
display(
    fare_inspection[
        (fare_inspection["fare_amount"] >= 0) &
        (fare_inspection["total_amount"] < 0)
    ].sample(n=10, random_state=42)
)

print("\nHighest fare records:")
display(
    fare_inspection.nlargest(20, "fare_amount")[
        inspection_columns + ["month_file"]
    ]
)

```

    High-fare classification:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>records</th>
    </tr>
    <tr>
      <th>fare_category</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>High fare + valid distance and duration</th>
      <td>7937</td>
    </tr>
    <tr>
      <th>High fare + invalid distance</th>
      <td>1325</td>
    </tr>
    <tr>
      <th>High fare + invalid duration</th>
      <td>61</td>
    </tr>
  </tbody>
</table>
</div>


    
    Negative fare with non-negative total:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pickup_time_local</th>
      <th>dropoff_time_local</th>
      <th>PULocationID</th>
      <th>DOLocationID</th>
      <th>trip_distance</th>
      <th>fare_amount</th>
      <th>total_amount</th>
      <th>duration_minutes</th>
      <th>speed_mph</th>
      <th>month_file</th>
      <th>fare_category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9463852</th>
      <td>2024-03-26 22:40:33-04:00</td>
      <td>2024-03-26 22:57:56-04:00</td>
      <td>114</td>
      <td>48</td>
      <td>2.94</td>
      <td>-0.01</td>
      <td>3.99</td>
      <td>17.383333</td>
      <td>10.147651</td>
      <td>trip_features_2024-03.parquet</td>
      <td>Other</td>
    </tr>
    <tr>
      <th>9326585</th>
      <td>2024-03-16 18:51:30-04:00</td>
      <td>2024-03-16 19:08:14-04:00</td>
      <td>113</td>
      <td>246</td>
      <td>2.39</td>
      <td>-0.54</td>
      <td>3.46</td>
      <td>16.733333</td>
      <td>8.569721</td>
      <td>trip_features_2024-03.parquet</td>
      <td>Other</td>
    </tr>
    <tr>
      <th>5804380</th>
      <td>2024-02-04 21:43:07-05:00</td>
      <td>2024-02-04 22:00:05-05:00</td>
      <td>148</td>
      <td>164</td>
      <td>2.89</td>
      <td>-1.00</td>
      <td>3.00</td>
      <td>16.966667</td>
      <td>10.220039</td>
      <td>trip_features_2024-02.parquet</td>
      <td>Other</td>
    </tr>
    <tr>
      <th>5952585</th>
      <td>2024-02-28 08:40:28-05:00</td>
      <td>2024-02-28 08:59:21-05:00</td>
      <td>236</td>
      <td>162</td>
      <td>2.36</td>
      <td>-1.51</td>
      <td>2.49</td>
      <td>18.883333</td>
      <td>7.498676</td>
      <td>trip_features_2024-02.parquet</td>
      <td>Other</td>
    </tr>
    <tr>
      <th>9346275</th>
      <td>2024-03-17 13:52:34-04:00</td>
      <td>2024-03-17 14:06:20-04:00</td>
      <td>107</td>
      <td>186</td>
      <td>1.36</td>
      <td>-0.58</td>
      <td>3.42</td>
      <td>13.766667</td>
      <td>5.927361</td>
      <td>trip_features_2024-03.parquet</td>
      <td>Other</td>
    </tr>
    <tr>
      <th>9553871</th>
      <td>2024-03-31 21:36:52-04:00</td>
      <td>2024-03-31 21:45:35-04:00</td>
      <td>164</td>
      <td>233</td>
      <td>1.38</td>
      <td>-1.00</td>
      <td>3.00</td>
      <td>8.716667</td>
      <td>9.499044</td>
      <td>trip_features_2024-03.parquet</td>
      <td>Other</td>
    </tr>
    <tr>
      <th>5861481</th>
      <td>2024-02-15 21:19:50-05:00</td>
      <td>2024-02-15 21:28:09-05:00</td>
      <td>161</td>
      <td>161</td>
      <td>0.94</td>
      <td>-1.00</td>
      <td>3.00</td>
      <td>8.316667</td>
      <td>6.781563</td>
      <td>trip_features_2024-02.parquet</td>
      <td>Other</td>
    </tr>
    <tr>
      <th>9283108</th>
      <td>2024-03-13 21:27:34-04:00</td>
      <td>2024-03-13 21:33:46-04:00</td>
      <td>231</td>
      <td>158</td>
      <td>1.39</td>
      <td>-1.00</td>
      <td>3.00</td>
      <td>6.200000</td>
      <td>13.451613</td>
      <td>trip_features_2024-03.parquet</td>
      <td>Other</td>
    </tr>
    <tr>
      <th>9546501</th>
      <td>2024-03-31 11:52:01-04:00</td>
      <td>2024-03-31 12:19:00-04:00</td>
      <td>236</td>
      <td>246</td>
      <td>4.61</td>
      <td>-1.05</td>
      <td>2.95</td>
      <td>26.983333</td>
      <td>10.250772</td>
      <td>trip_features_2024-03.parquet</td>
      <td>Other</td>
    </tr>
    <tr>
      <th>9347811</th>
      <td>2024-03-17 16:42:37-04:00</td>
      <td>2024-03-17 16:51:43-04:00</td>
      <td>148</td>
      <td>144</td>
      <td>0.49</td>
      <td>-3.00</td>
      <td>1.00</td>
      <td>9.100000</td>
      <td>3.230769</td>
      <td>trip_features_2024-03.parquet</td>
      <td>Other</td>
    </tr>
  </tbody>
</table>
</div>


    
    Non-negative fare with negative total:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pickup_time_local</th>
      <th>dropoff_time_local</th>
      <th>PULocationID</th>
      <th>DOLocationID</th>
      <th>trip_distance</th>
      <th>fare_amount</th>
      <th>total_amount</th>
      <th>duration_minutes</th>
      <th>speed_mph</th>
      <th>month_file</th>
      <th>fare_category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8722143</th>
      <td>2024-03-27 20:12:02-04:00</td>
      <td>2024-03-27 20:12:15-04:00</td>
      <td>233</td>
      <td>233</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>-4.00</td>
      <td>0.216667</td>
      <td>0.000000</td>
      <td>trip_features_2024-03.parquet</td>
      <td>Other</td>
    </tr>
    <tr>
      <th>2697149</th>
      <td>2024-01-30 18:09:31-05:00</td>
      <td>2024-01-30 18:10:47-05:00</td>
      <td>161</td>
      <td>230</td>
      <td>0.02</td>
      <td>0.0</td>
      <td>-4.00</td>
      <td>1.266667</td>
      <td>0.947368</td>
      <td>trip_features_2024-01.parquet</td>
      <td>Other</td>
    </tr>
    <tr>
      <th>2655290</th>
      <td>2024-01-30 11:30:29-05:00</td>
      <td>2024-01-30 11:59:50-05:00</td>
      <td>138</td>
      <td>239</td>
      <td>9.34</td>
      <td>0.0</td>
      <td>-5.75</td>
      <td>29.350000</td>
      <td>19.093697</td>
      <td>trip_features_2024-01.parquet</td>
      <td>Other</td>
    </tr>
    <tr>
      <th>1128374</th>
      <td>2024-01-13 22:14:26-05:00</td>
      <td>2024-01-13 22:25:45-05:00</td>
      <td>163</td>
      <td>140</td>
      <td>1.20</td>
      <td>0.0</td>
      <td>-4.00</td>
      <td>11.316667</td>
      <td>6.362297</td>
      <td>trip_features_2024-01.parquet</td>
      <td>Other</td>
    </tr>
    <tr>
      <th>3166124</th>
      <td>2024-02-02 22:31:11-05:00</td>
      <td>2024-02-02 22:32:09-05:00</td>
      <td>263</td>
      <td>263</td>
      <td>0.01</td>
      <td>0.0</td>
      <td>-4.00</td>
      <td>0.966667</td>
      <td>0.620690</td>
      <td>trip_features_2024-02.parquet</td>
      <td>Other</td>
    </tr>
    <tr>
      <th>6608685</th>
      <td>2024-03-07 11:25:20-05:00</td>
      <td>2024-03-07 11:39:53-05:00</td>
      <td>264</td>
      <td>264</td>
      <td>4.77</td>
      <td>0.0</td>
      <td>-4.00</td>
      <td>14.550000</td>
      <td>19.670103</td>
      <td>trip_features_2024-03.parquet</td>
      <td>Other</td>
    </tr>
    <tr>
      <th>5155095</th>
      <td>2024-02-23 18:34:16-05:00</td>
      <td>2024-02-23 18:34:29-05:00</td>
      <td>107</td>
      <td>107</td>
      <td>0.03</td>
      <td>0.0</td>
      <td>-4.00</td>
      <td>0.216667</td>
      <td>8.307692</td>
      <td>trip_features_2024-02.parquet</td>
      <td>Other</td>
    </tr>
    <tr>
      <th>5012349</th>
      <td>2024-02-22 12:01:45-05:00</td>
      <td>2024-02-22 12:02:25-05:00</td>
      <td>70</td>
      <td>138</td>
      <td>0.13</td>
      <td>0.0</td>
      <td>-3.25</td>
      <td>0.666667</td>
      <td>11.700000</td>
      <td>trip_features_2024-02.parquet</td>
      <td>Other</td>
    </tr>
    <tr>
      <th>9064101</th>
      <td>2024-03-31 09:49:50-04:00</td>
      <td>2024-03-31 09:51:08-04:00</td>
      <td>162</td>
      <td>170</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>-4.00</td>
      <td>1.300000</td>
      <td>0.000000</td>
      <td>trip_features_2024-03.parquet</td>
      <td>Other</td>
    </tr>
    <tr>
      <th>612666</th>
      <td>2024-01-08 13:10:39-05:00</td>
      <td>2024-01-08 13:11:25-05:00</td>
      <td>237</td>
      <td>237</td>
      <td>0.01</td>
      <td>0.0</td>
      <td>-4.00</td>
      <td>0.766667</td>
      <td>0.782609</td>
      <td>trip_features_2024-01.parquet</td>
      <td>Other</td>
    </tr>
  </tbody>
</table>
</div>


    
    Highest fare records:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pickup_time_local</th>
      <th>dropoff_time_local</th>
      <th>PULocationID</th>
      <th>DOLocationID</th>
      <th>trip_distance</th>
      <th>fare_amount</th>
      <th>total_amount</th>
      <th>duration_minutes</th>
      <th>speed_mph</th>
      <th>month_file</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5960537</th>
      <td>2024-02-29 07:56:45-05:00</td>
      <td>2024-02-29 08:01:55-05:00</td>
      <td>262</td>
      <td>140</td>
      <td>0.00</td>
      <td>9792.0</td>
      <td>9792.00</td>
      <td>5.166667</td>
      <td>0.000000</td>
      <td>trip_features_2024-02.parquet</td>
    </tr>
    <tr>
      <th>1714855</th>
      <td>2024-01-20 11:18:47-05:00</td>
      <td>2024-01-20 11:18:47-05:00</td>
      <td>264</td>
      <td>264</td>
      <td>0.00</td>
      <td>5000.0</td>
      <td>5000.00</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>1714856</th>
      <td>2024-01-20 11:19:33-05:00</td>
      <td>2024-01-20 11:19:33-05:00</td>
      <td>264</td>
      <td>264</td>
      <td>0.00</td>
      <td>5000.0</td>
      <td>5000.00</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>1714857</th>
      <td>2024-01-20 11:20:15-05:00</td>
      <td>2024-01-20 11:20:15-05:00</td>
      <td>264</td>
      <td>264</td>
      <td>0.00</td>
      <td>2500.0</td>
      <td>2500.00</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>1714859</th>
      <td>2024-01-20 11:27:48-05:00</td>
      <td>2024-01-20 11:27:48-05:00</td>
      <td>264</td>
      <td>264</td>
      <td>0.00</td>
      <td>2500.0</td>
      <td>2500.00</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>2084546</th>
      <td>2024-01-24 13:44:43-05:00</td>
      <td>2024-01-24 13:44:43-05:00</td>
      <td>264</td>
      <td>264</td>
      <td>0.00</td>
      <td>2500.0</td>
      <td>2500.00</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>1159580</th>
      <td>2024-01-14 10:08:11-05:00</td>
      <td>2024-01-16 13:54:22-05:00</td>
      <td>220</td>
      <td>220</td>
      <td>31.95</td>
      <td>2221.3</td>
      <td>2225.30</td>
      <td>3106.183333</td>
      <td>0.617156</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>76878</th>
      <td>2024-01-02 07:50:08-05:00</td>
      <td>2024-01-02 11:29:29-05:00</td>
      <td>168</td>
      <td>265</td>
      <td>233.25</td>
      <td>1616.5</td>
      <td>1617.50</td>
      <td>219.350000</td>
      <td>63.802143</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>1714858</th>
      <td>2024-01-20 11:22:27-05:00</td>
      <td>2024-01-20 11:22:27-05:00</td>
      <td>264</td>
      <td>264</td>
      <td>0.00</td>
      <td>1000.0</td>
      <td>1000.00</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>4603808</th>
      <td>2024-02-17 19:34:16-05:00</td>
      <td>2024-02-17 19:35:26-05:00</td>
      <td>139</td>
      <td>139</td>
      <td>0.00</td>
      <td>999.0</td>
      <td>1000.00</td>
      <td>1.166667</td>
      <td>0.000000</td>
      <td>trip_features_2024-02.parquet</td>
    </tr>
    <tr>
      <th>4670321</th>
      <td>2024-02-18 14:13:36-05:00</td>
      <td>2024-02-18 14:16:18-05:00</td>
      <td>218</td>
      <td>218</td>
      <td>1.07</td>
      <td>999.0</td>
      <td>1000.00</td>
      <td>2.700000</td>
      <td>23.777778</td>
      <td>trip_features_2024-02.parquet</td>
    </tr>
    <tr>
      <th>5441919</th>
      <td>2024-02-26 19:18:19-05:00</td>
      <td>2024-02-27 15:33:07-05:00</td>
      <td>193</td>
      <td>193</td>
      <td>36.20</td>
      <td>924.9</td>
      <td>931.40</td>
      <td>1214.800000</td>
      <td>1.787949</td>
      <td>trip_features_2024-02.parquet</td>
    </tr>
    <tr>
      <th>507301</th>
      <td>2024-01-06 21:01:38-05:00</td>
      <td>2024-01-06 23:41:44-05:00</td>
      <td>132</td>
      <td>265</td>
      <td>142.62</td>
      <td>912.3</td>
      <td>940.93</td>
      <td>160.100000</td>
      <td>53.449094</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>4352337</th>
      <td>2024-02-15 11:56:21-05:00</td>
      <td>2024-02-15 11:57:59-05:00</td>
      <td>265</td>
      <td>265</td>
      <td>0.00</td>
      <td>900.0</td>
      <td>901.00</td>
      <td>1.633333</td>
      <td>0.000000</td>
      <td>trip_features_2024-02.parquet</td>
    </tr>
    <tr>
      <th>4603810</th>
      <td>2024-02-17 19:47:10-05:00</td>
      <td>2024-02-17 19:47:18-05:00</td>
      <td>139</td>
      <td>139</td>
      <td>0.00</td>
      <td>900.0</td>
      <td>901.00</td>
      <td>0.133333</td>
      <td>0.000000</td>
      <td>trip_features_2024-02.parquet</td>
    </tr>
    <tr>
      <th>6141451</th>
      <td>2024-03-02 16:02:35-05:00</td>
      <td>2024-03-02 19:29:30-05:00</td>
      <td>132</td>
      <td>265</td>
      <td>181.50</td>
      <td>900.0</td>
      <td>951.26</td>
      <td>206.916667</td>
      <td>52.629883</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
    <tr>
      <th>1916191</th>
      <td>2024-01-22 16:40:43-05:00</td>
      <td>2024-01-22 19:11:20-05:00</td>
      <td>265</td>
      <td>265</td>
      <td>157.25</td>
      <td>899.0</td>
      <td>900.00</td>
      <td>150.616667</td>
      <td>62.642470</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>4794502</th>
      <td>2024-02-20 08:51:37-05:00</td>
      <td>2024-02-20 08:53:01-05:00</td>
      <td>170</td>
      <td>170</td>
      <td>0.00</td>
      <td>895.0</td>
      <td>898.50</td>
      <td>1.400000</td>
      <td>0.000000</td>
      <td>trip_features_2024-02.parquet</td>
    </tr>
    <tr>
      <th>4114272</th>
      <td>2024-02-12 17:48:54-05:00</td>
      <td>2024-02-12 20:22:07-05:00</td>
      <td>138</td>
      <td>265</td>
      <td>136.63</td>
      <td>877.3</td>
      <td>894.49</td>
      <td>153.216667</td>
      <td>53.504623</td>
      <td>trip_features_2024-02.parquet</td>
    </tr>
    <tr>
      <th>6335537</th>
      <td>2024-03-04 17:15:33-05:00</td>
      <td>2024-03-04 17:15:41-05:00</td>
      <td>265</td>
      <td>265</td>
      <td>0.00</td>
      <td>850.0</td>
      <td>851.00</td>
      <td>0.133333</td>
      <td>0.000000</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
  </tbody>
</table>
</div>



```python

```
