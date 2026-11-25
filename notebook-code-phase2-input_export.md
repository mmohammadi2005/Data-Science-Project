# شناسایی مسیر پروژه و فایل‌های داده


```python
from pathlib import Path
import os

# 1. تعیین مسیر اصلی پروژه
PROJECT_ROOT = Path(r"D:\کارشناسی\تحلیل داده")

print("Project root directory:")
print(PROJECT_ROOT)

# 2. نمایش فایل‌ها و پوشه‌های موجود در ریشه پروژه
print("\nFiles and folders in project root:")
for item in sorted(PROJECT_ROOT.iterdir()):
    kind = "DIR " if item.is_dir() else "FILE"
    print(f"{kind} | {item.name}")

# 3. جست‌وجوی عمیق فایل‌های Parquet در کل پوشه‌های پروژه (از جمله data/raw)
print("\nSearching for monthly Parquet files in project:")

for month in ["2024-01", "2024-02", "2024-03"]:
    # جست‌وجو در تمام زیرپوشه‌های پروژه
    matches = list(PROJECT_ROOT.rglob(f"*{month}*.parquet"))
    print(f"\n{month}:")
    if matches:
        for path in matches:
            print(" ", path)
    else:
        print("  No matching file found.")

```

    Project root directory:
    D:\کارشناسی\تحلیل داده
    
    Files and folders in project root:
    FILE | da_p (1).pdf
    DIR  | data
    DIR  | figures
    FILE | notebook-code.zip
    FILE | notebook-code1.md
    DIR  | notebooks
    FILE | NYC_Taxi_Project.rar
    FILE | PROJECT_CARD.md
    DIR  | reports
    FILE | reports- phase1.pdf
    FILE | نقش و مأموریت تو.docx
    
    Searching for monthly Parquet files in project:
    
    2024-01:
      D:\کارشناسی\تحلیل داده\data\interim\pickup_timestamps_2024-01.parquet
      D:\کارشناسی\تحلیل داده\data\processed\trip_features_2024-01.parquet
      D:\کارشناسی\تحلیل داده\data\raw\yellow_tripdata_2024-01.parquet
    
    2024-02:
      D:\کارشناسی\تحلیل داده\data\interim\pickup_timestamps_2024-02.parquet
      D:\کارشناسی\تحلیل داده\data\processed\trip_features_2024-02.parquet
      D:\کارشناسی\تحلیل داده\data\raw\yellow_tripdata_2024-02.parquet
    
    2024-03:
      D:\کارشناسی\تحلیل داده\data\interim\pickup_timestamps_2024-03.parquet
      D:\کارشناسی\تحلیل داده\data\processed\trip_features_2024-03.parquet
      D:\کارشناسی\تحلیل داده\data\raw\yellow_tripdata_2024-03.parquet
    

# اعتبارسنجی فایل‌های ورودی


```python

from pathlib import Path
import pyarrow.parquet as pq

# مسیر اصلی پروژه
PROJECT_ROOT = Path(r"D:\کارشناسی\تحلیل داده")

# مسیر داده‌های خام
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# فایل‌های مورد انتظار
MONTHLY_FILES = {
    "2024-01": RAW_DIR / "yellow_tripdata_2024-01.parquet",
    "2024-02": RAW_DIR / "yellow_tripdata_2024-02.parquet",
    "2024-03": RAW_DIR / "yellow_tripdata_2024-03.parquet",
}

# بررسی وجود و خوانایی فایل‌ها
audit_results = []

for month, file_path in MONTHLY_FILES.items():
    print(f"\nChecking {month}...")
    
    if not file_path.exists():
        print("ERROR: File not found!")
        continue

    parquet_file = pq.ParquetFile(file_path)
    metadata = parquet_file.metadata

    audit_results.append({
        "month": month,
        "exists": file_path.exists(),
        "rows": metadata.num_rows,
        "columns": metadata.num_columns,
        "size_MB": round(file_path.stat().st_size / (1024 ** 2), 2)
    })

    print("File:", file_path.name)
    print("Rows:", metadata.num_rows)
    print("Columns:", metadata.num_columns)
    print("Size (MB):", round(file_path.stat().st_size / (1024 ** 2), 2))

# خلاصه اعتبارسنجی
print("\n--- FINAL INPUT AUDIT ---")

for result in audit_results:
    print(result)

print("\nSuccessfully inspected files:", len(audit_results))

```

    
    Checking 2024-01...
    File: yellow_tripdata_2024-01.parquet
    Rows: 2964624
    Columns: 19
    Size (MB): 47.65
    
    Checking 2024-02...
    File: yellow_tripdata_2024-02.parquet
    Rows: 3007526
    Columns: 19
    Size (MB): 48.02
    
    Checking 2024-03...
    File: yellow_tripdata_2024-03.parquet
    Rows: 3582628
    Columns: 19
    Size (MB): 57.3
    
    --- FINAL INPUT AUDIT ---
    {'month': '2024-01', 'exists': True, 'rows': 2964624, 'columns': 19, 'size_MB': 47.65}
    {'month': '2024-02', 'exists': True, 'rows': 3007526, 'columns': 19, 'size_MB': 48.02}
    {'month': '2024-03', 'exists': True, 'rows': 3582628, 'columns': 19, 'size_MB': 57.3}
    
    Successfully inspected files: 3
    

## تثبیت مسیرها و بررسی ستون‌های موردنیاز


```python

from pathlib import Path

# ==========================================
# Phase 2 - Project configuration
# ==========================================

PROJECT_ROOT = Path(r"D:\کارشناسی\تحلیل داده")

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
PHASE2_FIGURES_DIR = PROJECT_ROOT / "figures" / "phase2"
PHASE2_REPORT_DIR = PROJECT_ROOT / "reports" / "phase2"

TIMEZONE = "America/New_York"
START = "2024-01-01 00:00:00"
END_EXCLUSIVE = "2024-04-01 00:00:00"

MONTHLY_FILES = {
    "2024-01": RAW_DIR / "yellow_tripdata_2024-01.parquet",
    "2024-02": RAW_DIR / "yellow_tripdata_2024-02.parquet",
    "2024-03": RAW_DIR / "yellow_tripdata_2024-03.parquet",
}

# ساخت پوشه‌های موردنیاز فاز 2
PHASE2_DIRS = [
    PROCESSED_DIR,
    INTERIM_DIR,
    PHASE2_FIGURES_DIR,
    PHASE2_REPORT_DIR,
]

for folder in PHASE2_DIRS:
    folder.mkdir(parents=True, exist_ok=True)

# کنترل وجود فایل‌های خام
missing_files = [
    str(path)
    for path in MONTHLY_FILES.values()
    if not path.exists()
]

if missing_files:
    raise FileNotFoundError(
        "The following source files were not found:\n"
        + "\n".join(missing_files)
    )

print("Project root:", PROJECT_ROOT)
print("Timezone:", TIMEZONE)
print("Research window:", START, "to", END_EXCLUSIVE)
print("\nPhase 2 folders:")
for folder in PHASE2_DIRS:
    print("Created/verified:", folder)

print("\nAll three raw Parquet files are available.")

```

    Project root: D:\کارشناسی\تحلیل داده
    Timezone: America/New_York
    Research window: 2024-01-01 00:00:00 to 2024-04-01 00:00:00
    
    Phase 2 folders:
    Created/verified: D:\کارشناسی\تحلیل داده\data\processed
    Created/verified: D:\کارشناسی\تحلیل داده\data\interim
    Created/verified: D:\کارشناسی\تحلیل داده\figures\phase2
    Created/verified: D:\کارشناسی\تحلیل داده\reports\phase2
    
    All three raw Parquet files are available.
    

## بررسی ساختار ستون‌ها و نمونه داده‌ها


```python

import pandas as pd
import pyarrow.parquet as pq

# ستون‌هایی که در مراحل بعدی به آن‌ها نیاز داریم
REQUIRED_COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_distance",
    "fare_amount",
    "total_amount",
    "passenger_count",
]

schema_audit = []

for month, file_path in MONTHLY_FILES.items():

    parquet_file = pq.ParquetFile(file_path)
    available_columns = parquet_file.schema.names

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in available_columns
    ]

    schema_audit.append({
        "month": month,
        "rows": parquet_file.metadata.num_rows,
        "total_columns": len(available_columns),
        "required_columns_found": (
            len(REQUIRED_COLUMNS) - len(missing_columns)
        ),
        "missing_required_columns": missing_columns
    })

    print(f"\n--- {month} ---")

    if missing_columns:
        print("Missing columns:", missing_columns)
        raise ValueError(
            f"Required columns missing in {month}: {missing_columns}"
        )

    # فقط چند ردیف و ستون ضروری را برای بررسی می‌خوانیم
    sample = pd.read_parquet(
        file_path,
        columns=REQUIRED_COLUMNS
    ).head(5)

    print("Pickup dtype:", sample["tpep_pickup_datetime"].dtype)
    print("Dropoff dtype:", sample["tpep_dropoff_datetime"].dtype)
    display(sample)

schema_audit_df = pd.DataFrame(schema_audit)

print("\n--- SCHEMA AUDIT SUMMARY ---")
display(schema_audit_df)

assert schema_audit_df["required_columns_found"].eq(
    len(REQUIRED_COLUMNS)
).all()

print("All required columns are available.")

```

    
    --- 2024-01 ---
    Pickup dtype: datetime64[us]
    Dropoff dtype: datetime64[us]
    


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
      <th>tpep_pickup_datetime</th>
      <th>tpep_dropoff_datetime</th>
      <th>PULocationID</th>
      <th>DOLocationID</th>
      <th>trip_distance</th>
      <th>fare_amount</th>
      <th>total_amount</th>
      <th>passenger_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-01-01 00:57:55</td>
      <td>2024-01-01 01:17:43</td>
      <td>186</td>
      <td>79</td>
      <td>1.72</td>
      <td>17.7</td>
      <td>22.70</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-01-01 00:03:00</td>
      <td>2024-01-01 00:09:36</td>
      <td>140</td>
      <td>236</td>
      <td>1.80</td>
      <td>10.0</td>
      <td>18.75</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-01-01 00:17:06</td>
      <td>2024-01-01 00:35:01</td>
      <td>236</td>
      <td>79</td>
      <td>4.70</td>
      <td>23.3</td>
      <td>31.30</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-01-01 00:36:38</td>
      <td>2024-01-01 00:44:56</td>
      <td>79</td>
      <td>211</td>
      <td>1.40</td>
      <td>10.0</td>
      <td>17.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-01-01 00:46:51</td>
      <td>2024-01-01 00:52:57</td>
      <td>211</td>
      <td>148</td>
      <td>0.80</td>
      <td>7.9</td>
      <td>16.10</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>


    
    --- 2024-02 ---
    Pickup dtype: datetime64[us]
    Dropoff dtype: datetime64[us]
    


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
      <th>tpep_pickup_datetime</th>
      <th>tpep_dropoff_datetime</th>
      <th>PULocationID</th>
      <th>DOLocationID</th>
      <th>trip_distance</th>
      <th>fare_amount</th>
      <th>total_amount</th>
      <th>passenger_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-02-01 00:04:45</td>
      <td>2024-02-01 00:19:58</td>
      <td>68</td>
      <td>236</td>
      <td>4.39</td>
      <td>20.5</td>
      <td>26.78</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-02-01 00:56:31</td>
      <td>2024-02-01 01:10:53</td>
      <td>48</td>
      <td>243</td>
      <td>7.71</td>
      <td>31.0</td>
      <td>45.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-02-01 00:07:50</td>
      <td>2024-02-01 00:43:12</td>
      <td>132</td>
      <td>261</td>
      <td>28.69</td>
      <td>70.0</td>
      <td>82.69</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-02-01 00:01:49</td>
      <td>2024-02-01 00:10:47</td>
      <td>161</td>
      <td>163</td>
      <td>1.10</td>
      <td>9.3</td>
      <td>17.15</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-02-01 00:37:35</td>
      <td>2024-02-01 00:51:15</td>
      <td>246</td>
      <td>79</td>
      <td>2.60</td>
      <td>15.6</td>
      <td>20.60</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>


    
    --- 2024-03 ---
    Pickup dtype: datetime64[us]
    Dropoff dtype: datetime64[us]
    


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
      <th>tpep_pickup_datetime</th>
      <th>tpep_dropoff_datetime</th>
      <th>PULocationID</th>
      <th>DOLocationID</th>
      <th>trip_distance</th>
      <th>fare_amount</th>
      <th>total_amount</th>
      <th>passenger_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-03-01 00:18:51</td>
      <td>2024-03-01 00:23:45</td>
      <td>142</td>
      <td>239</td>
      <td>1.30</td>
      <td>8.6</td>
      <td>16.30</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-03-01 00:26:00</td>
      <td>2024-03-01 00:29:06</td>
      <td>238</td>
      <td>24</td>
      <td>1.10</td>
      <td>7.2</td>
      <td>15.20</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-03-01 00:09:22</td>
      <td>2024-03-01 00:15:24</td>
      <td>263</td>
      <td>75</td>
      <td>0.86</td>
      <td>7.9</td>
      <td>10.40</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-03-01 00:33:45</td>
      <td>2024-03-01 00:39:34</td>
      <td>164</td>
      <td>162</td>
      <td>0.82</td>
      <td>7.9</td>
      <td>14.19</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-03-01 00:05:43</td>
      <td>2024-03-01 00:26:22</td>
      <td>263</td>
      <td>7</td>
      <td>4.90</td>
      <td>25.4</td>
      <td>30.40</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


    
    --- SCHEMA AUDIT SUMMARY ---
    


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
      <th>month</th>
      <th>rows</th>
      <th>total_columns</th>
      <th>required_columns_found</th>
      <th>missing_required_columns</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-01</td>
      <td>2964624</td>
      <td>19</td>
      <td>8</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-02</td>
      <td>3007526</td>
      <td>19</td>
      <td>8</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-03</td>
      <td>3582628</td>
      <td>19</td>
      <td>8</td>
      <td>[]</td>
    </tr>
  </tbody>
</table>
</div>


    All required columns are available.
    

# استخراج مستقل ورودی سری زمانی فاز 

##  حذف تکرارهای دقیق و استخراج زمان‌های معتبر


```python

import pandas as pd

# ==========================================
# Phase 2 - Step 3.1
# Extract pickup timestamps with exact deduplication
# ==========================================

timestamp_audit_rows = []

for month, file_path in MONTHLY_FILES.items():

    print(f"\nProcessing {month}...")

    # خواندن تمام ستون‌ها برای تشخیص تکرار دقیق رکورد
    df = pd.read_parquet(file_path)

    raw_rows = len(df)

    # اثرانگشت هر رکورد بر اساس تمام ستون‌ها
    row_hash = pd.util.hash_pandas_object(
        df,
        index=False
    )

    # نگه‌داشتن اولین رخداد هر رکورد کاملاً یکسان
    keep_mask = ~row_hash.duplicated(keep="first")

    duplicate_rows = int((~keep_mask).sum())

    # استخراج زمان سوارشدن پس از حذف تکرار
    pickup_times = pd.to_datetime(
        df.loc[keep_mask, "tpep_pickup_datetime"],
        errors="coerce"
    )

    invalid_times = int(pickup_times.isna().sum())

    # ذخیره داده میانی هر ماه
    monthly_output = pd.DataFrame({
        "pickup_time": pickup_times.dropna()
    })

    output_path = (
        INTERIM_DIR /
        f"pickup_timestamps_{month}.parquet"
    )

    monthly_output.to_parquet(
        output_path,
        index=False
    )

    timestamp_audit_rows.append({
        "month": month,
        "raw_rows": raw_rows,
        "duplicate_rows_excluded": duplicate_rows,
        "rows_after_deduplication": int(keep_mask.sum()),
        "invalid_pickup_timestamps": invalid_times,
        "valid_pickup_timestamps": len(monthly_output)
    })

    print("Raw rows:", raw_rows)
    print("Duplicate rows excluded:", duplicate_rows)
    print("Invalid pickup timestamps:", invalid_times)
    print("Saved:", output_path)

    del df, row_hash, keep_mask, pickup_times, monthly_output

# گزارش ممیزی استخراج
timestamp_audit_df = pd.DataFrame(timestamp_audit_rows)

display(timestamp_audit_df)

# ذخیره گزارش ممیزی
timestamp_audit_df.to_csv(
    INTERIM_DIR / "timestamp_extraction_audit.csv",
    index=False
)

print("\nTimestamp extraction completed.")

```

    
    Processing 2024-01...
    Raw rows: 2964624
    Duplicate rows excluded: 0
    Invalid pickup timestamps: 0
    Saved: D:\کارشناسی\تحلیل داده\data\interim\pickup_timestamps_2024-01.parquet
    
    Processing 2024-02...
    Raw rows: 3007526
    Duplicate rows excluded: 1
    Invalid pickup timestamps: 0
    Saved: D:\کارشناسی\تحلیل داده\data\interim\pickup_timestamps_2024-02.parquet
    
    Processing 2024-03...
    Raw rows: 3582628
    Duplicate rows excluded: 0
    Invalid pickup timestamps: 0
    Saved: D:\کارشناسی\تحلیل داده\data\interim\pickup_timestamps_2024-03.parquet
    


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
      <th>month</th>
      <th>raw_rows</th>
      <th>duplicate_rows_excluded</th>
      <th>rows_after_deduplication</th>
      <th>invalid_pickup_timestamps</th>
      <th>valid_pickup_timestamps</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-01</td>
      <td>2964624</td>
      <td>0</td>
      <td>2964624</td>
      <td>0</td>
      <td>2964624</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-02</td>
      <td>3007526</td>
      <td>1</td>
      <td>3007525</td>
      <td>0</td>
      <td>3007525</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-03</td>
      <td>3582628</td>
      <td>0</td>
      <td>3582628</td>
      <td>0</td>
      <td>3582628</td>
    </tr>
  </tbody>
</table>
</div>


    
    Timestamp extraction completed.
    

## ساخت و ذخیره سری زمانی ساعتی


```python

# ==========================================
# Phase 2 - Step 3.2
# Reconstruct DST-aware hourly trip counts
# ==========================================

NY_TZ = "America/New_York"

# خواندن زمان‌های استخراج‌شده از سه ماه
timestamp_frames = []

for month in MONTHLY_FILES:

    input_path = (
        INTERIM_DIR /
        f"pickup_timestamps_{month}.parquet"
    )

    monthly_timestamps = pd.read_parquet(input_path)

    timestamp_frames.append(
        monthly_timestamps["pickup_time"]
    )

# ترکیب زمان‌های سه ماه
all_pickups = pd.concat(
    timestamp_frames,
    ignore_index=True
)

all_pickups = pd.to_datetime(
    all_pickups,
    errors="coerce"
)

# حذف زمان‌های نامعتبر
all_pickups = all_pickups.dropna()

# محدود کردن به بازه پژوهش
inside_window = (
    all_pickups.ge(pd.Timestamp(START)) &
    all_pickups.lt(pd.Timestamp(END_EXCLUSIVE))
)

valid_in_window = all_pickups.loc[inside_window].copy()

outside_window_count = int((~inside_window).sum())

# اعمال منطقه زمانی مطابق فاز 1
valid_in_window_ny = valid_in_window.dt.tz_localize(
    NY_TZ,
    ambiguous="NaT",
    nonexistent="NaT"
)

unresolved_local_times = int(
    valid_in_window_ny.isna().sum()
)

# شمارش سفرها در هر ساعت محلی
hourly_counts_observed = (
    valid_in_window_ny
    .dropna()
    .dt.floor("h")
    .value_counts()
    .sort_index()
)

# ساخت شاخص کامل ساعت‌های واقعی
start_local = pd.Timestamp(START).tz_localize(NY_TZ)
end_local = pd.Timestamp(END_EXCLUSIVE).tz_localize(NY_TZ)

full_hour_index = pd.date_range(
    start=start_local,
    end=end_local,
    freq="h",
    inclusive="left"
)

# سری زمانی نهایی
hourly_counts = (
    hourly_counts_observed
    .reindex(full_hour_index, fill_value=0)
    .rename("trip_count")
)

# کنترل نتایج
print("Total valid pickup timestamps:", len(all_pickups))
print("Records inside research window:", len(valid_in_window))
print("Records outside research window:", outside_window_count)
print("Unresolved local timestamps:", unresolved_local_times)

print("\nExpected hourly observations:", len(full_hour_index))
print("Actual hourly observations:", len(hourly_counts))
print("Hours with zero trips:", int((hourly_counts == 0).sum()))
print("Timezone:", hourly_counts.index.tz)

# کنترل تعداد ساعات مورد انتظار
assert len(hourly_counts) == 2183, (
    f"Unexpected hourly row count: {len(hourly_counts)}"
)

# ذخیره سری زمانی در قالب Parquet
hourly_output = hourly_counts.to_frame()

hourly_output.to_parquet(
    PROCESSED_DIR / "hourly_trip_counts.parquet"
)

print("\nSaved hourly time series:")
print(PROCESSED_DIR / "hourly_trip_counts.parquet")

display(hourly_output.head())
display(hourly_output.tail())

```

    Total valid pickup timestamps: 9554777
    Records inside research window: 9554756
    Records outside research window: 21
    Unresolved local timestamps: 0
    
    Expected hourly observations: 2183
    Actual hourly observations: 2183
    Hours with zero trips: 0
    Timezone: America/New_York
    
    Saved hourly time series:
    D:\کارشناسی\تحلیل داده\data\processed\hourly_trip_counts.parquet
    


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
      <th>2024-03-31 19:00:00-04:00</th>
      <td>5410</td>
    </tr>
    <tr>
      <th>2024-03-31 20:00:00-04:00</th>
      <td>5698</td>
    </tr>
    <tr>
      <th>2024-03-31 21:00:00-04:00</th>
      <td>5070</td>
    </tr>
    <tr>
      <th>2024-03-31 22:00:00-04:00</th>
      <td>3720</td>
    </tr>
    <tr>
      <th>2024-03-31 23:00:00-04:00</th>
      <td>2231</td>
    </tr>
  </tbody>
</table>
</div>


# آماده‌سازی داده‌های مکانی و ویژگی‌های سفر

## کنترل فایل سری زمانی ذخیره‌شده


```python

import pandas as pd

hourly_path = PROCESSED_DIR / "hourly_trip_counts.parquet"

hourly_check = pd.read_parquet(hourly_path)

print("File exists:", hourly_path.exists())
print("Rows:", len(hourly_check))
print("Columns:", hourly_check.columns.tolist())
print("Index type:", type(hourly_check.index))
print("Timezone:", hourly_check.index.tz)
print("Zero-count hours:", int((hourly_check["trip_count"] == 0).sum()))

# کنترل تغییر ساعت تابستانی
spring_forward_day = hourly_check.loc["2024-03-10"]

print("\nHours on March 10, 2024:", len(spring_forward_day))

assert len(hourly_check) == 2183
assert hourly_check.index.tz is not None
assert hourly_check.index.tz.zone == "America/New_York"
assert len(spring_forward_day) == 23

print("\nHourly time series validation: PASSED")

```

    File exists: True
    Rows: 2183
    Columns: ['trip_count']
    Index type: <class 'pandas.core.indexes.datetimes.DatetimeIndex'>
    Timezone: America/New_York
    Zero-count hours: 0
    
    Hours on March 10, 2024: 23
    
    Hourly time series validation: PASSED
    

## استخراج داده‌های سطح سفر


```python

import pandas as pd
import numpy as np

NY_TZ = "America/New_York"

TRIP_COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_distance",
    "fare_amount",
    "total_amount",
    "passenger_count",
]

trip_extraction_audit = []

for month, file_path in MONTHLY_FILES.items():

    print(f"\nProcessing trip features: {month}")

    # ==========================================
    # 1. Read raw data and remove exact duplicates
    # ==========================================

    df = pd.read_parquet(file_path)

    raw_rows = len(df)

    row_hash = pd.util.hash_pandas_object(
        df,
        index=False
    )

    keep_mask = ~row_hash.duplicated(keep="first")

    duplicate_rows = int((~keep_mask).sum())

    df = df.loc[keep_mask, TRIP_COLUMNS].copy()

    # ==========================================
    # 2. Convert raw timestamps
    # ==========================================

    df["tpep_pickup_datetime"] = pd.to_datetime(
        df["tpep_pickup_datetime"],
        errors="coerce"
    )

    df["tpep_dropoff_datetime"] = pd.to_datetime(
        df["tpep_dropoff_datetime"],
        errors="coerce"
    )

    # ==========================================
    # 3. Apply research window using pickup time
    # ==========================================

    in_window = (
        df["tpep_pickup_datetime"].ge(pd.Timestamp(START)) &
        df["tpep_pickup_datetime"].lt(pd.Timestamp(END_EXCLUSIVE))
    )

    df = df.loc[in_window].copy()

    # ==========================================
    # 4. Localize BOTH timestamps to New York time
    # ==========================================

    pickup_local = df["tpep_pickup_datetime"].dt.tz_localize(
        NY_TZ,
        ambiguous="NaT",
        nonexistent="NaT"
    )

    dropoff_local = df["tpep_dropoff_datetime"].dt.tz_localize(
        NY_TZ,
        ambiguous="NaT",
        nonexistent="NaT"
    )

    # Keep records with valid pickup timestamps
    resolved_pickup = pickup_local.notna()

    df = df.loc[resolved_pickup].copy()

    pickup_local = pickup_local.loc[resolved_pickup]
    dropoff_local = dropoff_local.loc[resolved_pickup]

    # ==========================================
    # 5. Convert numeric variables
    # ==========================================

    for col in [
        "trip_distance",
        "fare_amount",
        "total_amount",
        "passenger_count"
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # ==========================================
    # 6. Calculate duration using UTC timestamps
    # ==========================================

    pickup_utc = pickup_local.dt.tz_convert("UTC")
    dropoff_utc = dropoff_local.dt.tz_convert("UTC")

    duration = (
        dropoff_utc - pickup_utc
    ).dt.total_seconds() / 60

    df["duration_minutes"] = duration

    # ==========================================
    # 7. Calculate average speed
    # ==========================================

    valid_speed = (
        df["duration_minutes"].gt(0) &
        df["trip_distance"].ge(0) &
        df["trip_distance"].notna()
    )

    df["speed_mph"] = np.nan

    df.loc[valid_speed, "speed_mph"] = (
        df.loc[valid_speed, "trip_distance"] /
        (df.loc[valid_speed, "duration_minutes"] / 60)
    )

    # ==========================================
    # 8. Save local timestamps and hourly feature
    # ==========================================

    df["pickup_time_local"] = pickup_local
    df["dropoff_time_local"] = dropoff_local

    df["pickup_hour"] = pickup_local.dt.floor("h")

    # ==========================================
    # 9. Prepare output
    # ==========================================

    output_columns = [
        "pickup_time_local",
        "dropoff_time_local",
        "pickup_hour",
        "PULocationID",
        "DOLocationID",
        "trip_distance",
        "fare_amount",
        "total_amount",
        "passenger_count",
        "duration_minutes",
        "speed_mph",
    ]

    trip_output = df[output_columns].copy()

    output_path = (
        PROCESSED_DIR /
        f"trip_features_{month}.parquet"
    )

    trip_output.to_parquet(
        output_path,
        index=False
    )

    # ==========================================
    # 10. Extraction audit
    # ==========================================

    trip_extraction_audit.append({
        "month": month,
        "raw_rows": raw_rows,
        "duplicate_rows_removed": duplicate_rows,
        "rows_in_window_and_resolved": len(trip_output),
        "invalid_dropoff_timestamps": int(
            trip_output["dropoff_time_local"].isna().sum()
        ),
        "invalid_duration_rows": int(
            trip_output["duration_minutes"].isna().sum()
        ),
        "nonpositive_duration_rows": int(
            (trip_output["duration_minutes"] <= 0).sum()
        ),
        "saved_file": output_path.name
    })

    print("Saved:", output_path.name)
    print("Rows saved:", len(trip_output))
    print(
        "Invalid dropoff timestamps:",
        trip_output["dropoff_time_local"].isna().sum()
    )
    print(
        "Invalid durations:",
        trip_output["duration_minutes"].isna().sum()
    )

    del df, row_hash, keep_mask, trip_output

# ==========================================
# 11. Save extraction audit report
# ==========================================

trip_extraction_audit_df = pd.DataFrame(
    trip_extraction_audit
)

display(trip_extraction_audit_df)

trip_extraction_audit_df.to_csv(
    INTERIM_DIR / "trip_feature_extraction_audit.csv",
    index=False
)

print("\nTrip-level extraction completed successfully.")

```

    
    Processing trip features: 2024-01
    Saved: trip_features_2024-01.parquet
    Rows saved: 2964609
    Invalid dropoff timestamps: 0
    Invalid durations: 0
    
    Processing trip features: 2024-02
    Saved: trip_features_2024-02.parquet
    Rows saved: 3007523
    Invalid dropoff timestamps: 0
    Invalid durations: 0
    
    Processing trip features: 2024-03
    Saved: trip_features_2024-03.parquet
    Rows saved: 3582624
    Invalid dropoff timestamps: 3
    Invalid durations: 3
    


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
      <th>month</th>
      <th>raw_rows</th>
      <th>duplicate_rows_removed</th>
      <th>rows_in_window_and_resolved</th>
      <th>invalid_dropoff_timestamps</th>
      <th>invalid_duration_rows</th>
      <th>nonpositive_duration_rows</th>
      <th>saved_file</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-01</td>
      <td>2964624</td>
      <td>0</td>
      <td>2964609</td>
      <td>0</td>
      <td>0</td>
      <td>870</td>
      <td>trip_features_2024-01.parquet</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-02</td>
      <td>3007526</td>
      <td>1</td>
      <td>3007523</td>
      <td>0</td>
      <td>0</td>
      <td>803</td>
      <td>trip_features_2024-02.parquet</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-03</td>
      <td>3582628</td>
      <td>0</td>
      <td>3582624</td>
      <td>3</td>
      <td>3</td>
      <td>1128</td>
      <td>trip_features_2024-03.parquet</td>
    </tr>
  </tbody>
</table>
</div>


    
    Trip-level extraction completed successfully.
    

##  ساخت خلاصه ویژگی‌های مناطق و الگوهای مبدأ ـ مقصد


```python

import pandas as pd
import numpy as np

zone_parts = []
od_parts = []
hourly_zone_parts = []

for month in MONTHLY_FILES:

    path = PROCESSED_DIR / f"trip_features_{month}.parquet"
    trips = pd.read_parquet(path)

    print(f"Processing {month}: {len(trips):,} trips")

    # 1. ویژگی‌های مناطق مبدأ
    pickup_trips = trips.dropna(
        subset=["PULocationID"]
    ).copy()

    zone_month = pickup_trips.groupby(
        "PULocationID"
    ).agg(
        pickup_trips=("PULocationID", "size"),
        distance_sum=("trip_distance", "sum"),
        distance_n=("trip_distance", "count"),
        duration_sum=("duration_minutes", "sum"),
        duration_n=("duration_minutes", "count"),
        fare_sum=("fare_amount", "sum"),
        fare_n=("fare_amount", "count"),
        speed_sum=("speed_mph", "sum"),
        speed_n=("speed_mph", "count")
    ).reset_index()

    zone_parts.append(zone_month)

    # 2. الگوهای مبدأ ـ مقصد
    od_month = (
        trips.dropna(
            subset=["PULocationID", "DOLocationID"]
        )
        .groupby(["PULocationID", "DOLocationID"])
        .size()
        .rename("trip_count")
        .reset_index()
    )

    od_parts.append(od_month)

    # 3. تقاضای ساعتی مناطق مبدأ
    hourly_zone_month = (
        pickup_trips
        .groupby(["pickup_hour", "PULocationID"])
        .size()
        .rename("trip_count")
        .reset_index()
    )

    hourly_zone_parts.append(hourly_zone_month)

    del trips, pickup_trips

# تجمیع ویژگی‌های مناطق در کل سه ماه
zone_all = pd.concat(zone_parts, ignore_index=True)

zone_features = zone_all.groupby(
    "PULocationID"
).agg(
    pickup_trips=("pickup_trips", "sum"),
    distance_sum=("distance_sum", "sum"),
    distance_n=("distance_n", "sum"),
    duration_sum=("duration_sum", "sum"),
    duration_n=("duration_n", "sum"),
    fare_sum=("fare_sum", "sum"),
    fare_n=("fare_n", "sum"),
    speed_sum=("speed_sum", "sum"),
    speed_n=("speed_n", "sum")
).reset_index()

# محاسبه میانگین تجمیعی با تعداد مقادیر معتبر
for feature, total, count in [
    ("avg_trip_distance", "distance_sum", "distance_n"),
    ("avg_duration_minutes", "duration_sum", "duration_n"),
    ("avg_fare_amount", "fare_sum", "fare_n"),
    ("avg_speed_mph", "speed_sum", "speed_n")
]:
    zone_features[feature] = (
        zone_features[total] /
        zone_features[count].replace(0, np.nan)
    )

zone_features["is_unknown_zone"] = (
    zone_features["PULocationID"].isin([264, 265])
)

# تجمیع الگوهای مبدأ ـ مقصد
od_patterns = (
    pd.concat(od_parts, ignore_index=True)
    .groupby(
        ["PULocationID", "DOLocationID"],
        as_index=False
    )["trip_count"]
    .sum()
)

# تجمیع تقاضای ساعتی هر منطقه
hourly_zone_demand = (
    pd.concat(hourly_zone_parts, ignore_index=True)
    .groupby(
        ["pickup_hour", "PULocationID"],
        as_index=False
    )["trip_count"]
    .sum()
)

# ذخیره سه خروجی
zone_features.to_parquet(
    PROCESSED_DIR / "zone_features.parquet",
    index=False
)

od_patterns.to_parquet(
    PROCESSED_DIR / "od_patterns.parquet",
    index=False
)

hourly_zone_demand.to_parquet(
    PROCESSED_DIR / "hourly_zone_demand.parquet",
    index=False
)

print("\n--- SPATIAL OUTPUT SUMMARY ---")
print("Zones:", len(zone_features))
print("OD pairs:", len(od_patterns))
print("Hourly-zone observations:", len(hourly_zone_demand))

print(
    "Unknown zone IDs:",
    sorted(
        zone_features.loc[
            zone_features["is_unknown_zone"],
            "PULocationID"
        ].astype(int).tolist()
    )
)

assert zone_features["PULocationID"].is_unique
assert not od_patterns.duplicated(
    ["PULocationID", "DOLocationID"]
).any()

print("\nAll spatial outputs rebuilt successfully.")

display(
    zone_features.sort_values(
        "pickup_trips", ascending=False
    ).head(10)
)

display(
    od_patterns.sort_values(
        "trip_count", ascending=False
    ).head(10)
)

```

    Processing 2024-01: 2,964,609 trips
    Processing 2024-02: 3,007,523 trips
    Processing 2024-03: 3,582,624 trips
    
    --- SPATIAL OUTPUT SUMMARY ---
    Zones: 262
    OD pairs: 37255
    Hourly-zone observations: 240917
    Unknown zone IDs: [264, 265]
    
    All spatial outputs rebuilt successfully.
    


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
      <th>157</th>
      <td>161</td>
      <td>453825</td>
      <td>1147374.69</td>
      <td>453825</td>
      <td>6.803428e+06</td>
      <td>453825</td>
      <td>7055927.30</td>
      <td>453825</td>
      <td>4.382643e+06</td>
      <td>453752</td>
      <td>2.528232</td>
      <td>14.991302</td>
      <td>15.547683</td>
      <td>9.658675</td>
      <td>False</td>
    </tr>
    <tr>
      <th>233</th>
      <td>237</td>
      <td>439138</td>
      <td>812564.60</td>
      <td>439138</td>
      <td>5.253203e+06</td>
      <td>439138</td>
      <td>5486779.50</td>
      <td>439138</td>
      <td>4.146915e+06</td>
      <td>439069</td>
      <td>1.850363</td>
      <td>11.962534</td>
      <td>12.494431</td>
      <td>9.444792</td>
      <td>False</td>
    </tr>
    <tr>
      <th>128</th>
      <td>132</td>
      <td>429745</td>
      <td>6700446.10</td>
      <td>429745</td>
      <td>1.700340e+07</td>
      <td>429745</td>
      <td>25641994.26</td>
      <td>429745</td>
      <td>1.184371e+07</td>
      <td>429588</td>
      <td>15.591679</td>
      <td>39.566251</td>
      <td>59.667929</td>
      <td>27.569930</td>
      <td>False</td>
    </tr>
    <tr>
      <th>232</th>
      <td>236</td>
      <td>416508</td>
      <td>1013344.37</td>
      <td>416508</td>
      <td>5.066246e+06</td>
      <td>416508</td>
      <td>5437510.90</td>
      <td>416508</td>
      <td>5.093796e+06</td>
      <td>416428</td>
      <td>2.432953</td>
      <td>12.163621</td>
      <td>13.054998</td>
      <td>12.232118</td>
      <td>False</td>
    </tr>
    <tr>
      <th>158</th>
      <td>162</td>
      <td>336459</td>
      <td>861993.81</td>
      <td>336459</td>
      <td>4.848925e+06</td>
      <td>336459</td>
      <td>5092483.84</td>
      <td>336459</td>
      <td>3.596370e+06</td>
      <td>336394</td>
      <td>2.561958</td>
      <td>14.411638</td>
      <td>15.135526</td>
      <td>10.690945</td>
      <td>False</td>
    </tr>
    <tr>
      <th>226</th>
      <td>230</td>
      <td>331570</td>
      <td>1016020.35</td>
      <td>331570</td>
      <td>5.432746e+06</td>
      <td>331570</td>
      <td>5935449.00</td>
      <td>331570</td>
      <td>3.592884e+06</td>
      <td>331477</td>
      <td>3.064271</td>
      <td>16.384914</td>
      <td>17.901044</td>
      <td>10.839014</td>
      <td>False</td>
    </tr>
    <tr>
      <th>182</th>
      <td>186</td>
      <td>319705</td>
      <td>724305.33</td>
      <td>319705</td>
      <td>5.319699e+06</td>
      <td>319705</td>
      <td>5178728.46</td>
      <td>319705</td>
      <td>2.752303e+06</td>
      <td>319637</td>
      <td>2.265543</td>
      <td>16.639398</td>
      <td>16.198459</td>
      <td>8.610714</td>
      <td>False</td>
    </tr>
    <tr>
      <th>138</th>
      <td>142</td>
      <td>316314</td>
      <td>718161.20</td>
      <td>316314</td>
      <td>4.101386e+06</td>
      <td>316314</td>
      <td>4310080.46</td>
      <td>316314</td>
      <td>3.384561e+06</td>
      <td>316246</td>
      <td>2.270406</td>
      <td>12.966184</td>
      <td>13.625955</td>
      <td>10.702305</td>
      <td>False</td>
    </tr>
    <tr>
      <th>134</th>
      <td>138</td>
      <td>284359</td>
      <td>2731727.99</td>
      <td>284359</td>
      <td>8.147663e+06</td>
      <td>284359</td>
      <td>11894912.85</td>
      <td>284359</td>
      <td>6.472916e+06</td>
      <td>284280</td>
      <td>9.606617</td>
      <td>28.652736</td>
      <td>41.830619</td>
      <td>22.769509</td>
      <td>False</td>
    </tr>
    <tr>
      <th>235</th>
      <td>239</td>
      <td>281730</td>
      <td>930167.61</td>
      <td>281730</td>
      <td>3.512230e+06</td>
      <td>281730</td>
      <td>3867931.83</td>
      <td>281730</td>
      <td>4.220046e+06</td>
      <td>281673</td>
      <td>3.301628</td>
      <td>12.466652</td>
      <td>13.729215</td>
      <td>14.982074</td>
      <td>False</td>
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
      <th>PULocationID</th>
      <th>DOLocationID</th>
      <th>trip_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>32953</th>
      <td>237</td>
      <td>236</td>
      <td>64384</td>
    </tr>
    <tr>
      <th>32716</th>
      <td>236</td>
      <td>237</td>
      <td>57111</td>
    </tr>
    <tr>
      <th>32715</th>
      <td>236</td>
      <td>236</td>
      <td>47509</td>
    </tr>
    <tr>
      <th>32954</th>
      <td>237</td>
      <td>237</td>
      <td>44270</td>
    </tr>
    <tr>
      <th>21939</th>
      <td>161</td>
      <td>237</td>
      <td>31118</td>
    </tr>
    <tr>
      <th>19140</th>
      <td>142</td>
      <td>239</td>
      <td>27085</td>
    </tr>
    <tr>
      <th>32884</th>
      <td>237</td>
      <td>161</td>
      <td>26836</td>
    </tr>
    <tr>
      <th>33305</th>
      <td>239</td>
      <td>142</td>
      <td>26828</td>
    </tr>
    <tr>
      <th>21938</th>
      <td>161</td>
      <td>236</td>
      <td>26504</td>
    </tr>
    <tr>
      <th>33386</th>
      <td>239</td>
      <td>238</td>
      <td>25824</td>
    </tr>
  </tbody>
</table>
</div>


## ممیزی مقادیر گمشده و دامنه متغیرها


```python

import pandas as pd
import numpy as np

trip_audit_rows = []

NUMERIC_COLS = [
    "trip_distance",
    "fare_amount",
    "total_amount",
    "passenger_count",
    "duration_minutes",
    "speed_mph"
]

for month in MONTHLY_FILES:

    path = PROCESSED_DIR / f"trip_features_{month}.parquet"
    df = pd.read_parquet(path)

    row = {
        "month": month,
        "rows": len(df),
        "invalid_dropoff_timestamp": int(
            df["dropoff_time_local"].isna().sum()
        ),
        "invalid_duration": int(
            df["duration_minutes"].isna().sum()
        )
    }

    for col in NUMERIC_COLS:
        row[f"{col}_missing"] = int(df[col].isna().sum())

    row["negative_distance"] = int(
        (df["trip_distance"] < 0).sum()
    )
    row["negative_fare"] = int(
        (df["fare_amount"] < 0).sum()
    )
    row["negative_total"] = int(
        (df["total_amount"] < 0).sum()
    )
    row["nonpositive_duration"] = int(
        (df["duration_minutes"] <= 0).sum()
    )
    row["nonpositive_passengers"] = int(
        (df["passenger_count"] <= 0).sum()
    )

    for col in NUMERIC_COLS:
        row[f"{col}_min"] = df[col].min()
        row[f"{col}_median"] = df[col].median()
        row[f"{col}_max"] = df[col].max()

    trip_audit_rows.append(row)
    del df

trip_quality_audit = pd.DataFrame(trip_audit_rows)

display(trip_quality_audit.T)

trip_quality_audit.to_csv(
    INTERIM_DIR / "trip_quality_audit_corrected.csv",
    index=False
)

print("\nCorrected trip quality audit saved.")

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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>month</th>
      <td>2024-01</td>
      <td>2024-02</td>
      <td>2024-03</td>
    </tr>
    <tr>
      <th>rows</th>
      <td>2964609</td>
      <td>3007523</td>
      <td>3582624</td>
    </tr>
    <tr>
      <th>invalid_dropoff_timestamp</th>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>invalid_duration</th>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>trip_distance_missing</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>fare_amount_missing</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>total_amount_missing</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>passenger_count_missing</th>
      <td>140162</td>
      <td>185610</td>
      <td>426190</td>
    </tr>
    <tr>
      <th>duration_minutes_missing</th>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>speed_mph_missing</th>
      <td>870</td>
      <td>803</td>
      <td>1131</td>
    </tr>
    <tr>
      <th>negative_distance</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>negative_fare</th>
      <td>37447</td>
      <td>40655</td>
      <td>58464</td>
    </tr>
    <tr>
      <th>negative_total</th>
      <td>35503</td>
      <td>36080</td>
      <td>44311</td>
    </tr>
    <tr>
      <th>nonpositive_duration</th>
      <td>870</td>
      <td>803</td>
      <td>1128</td>
    </tr>
    <tr>
      <th>nonpositive_passengers</th>
      <td>31465</td>
      <td>34094</td>
      <td>40372</td>
    </tr>
    <tr>
      <th>trip_distance_min</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>trip_distance_median</th>
      <td>1.68</td>
      <td>1.69</td>
      <td>1.75</td>
    </tr>
    <tr>
      <th>trip_distance_max</th>
      <td>312722.3</td>
      <td>222478.29</td>
      <td>176836.3</td>
    </tr>
    <tr>
      <th>fare_amount_min</th>
      <td>-899.0</td>
      <td>-999.0</td>
      <td>-800.0</td>
    </tr>
    <tr>
      <th>fare_amount_median</th>
      <td>12.8</td>
      <td>12.8</td>
      <td>13.5</td>
    </tr>
    <tr>
      <th>fare_amount_max</th>
      <td>5000.0</td>
      <td>9792.0</td>
      <td>900.0</td>
    </tr>
    <tr>
      <th>total_amount_min</th>
      <td>-900.0</td>
      <td>-1000.0</td>
      <td>-800.99</td>
    </tr>
    <tr>
      <th>total_amount_median</th>
      <td>20.1</td>
      <td>20.16</td>
      <td>20.52</td>
    </tr>
    <tr>
      <th>total_amount_max</th>
      <td>5000.0</td>
      <td>9792.0</td>
      <td>1021.99</td>
    </tr>
    <tr>
      <th>passenger_count_min</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>passenger_count_median</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>passenger_count_max</th>
      <td>9.0</td>
      <td>9.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>duration_minutes_min</th>
      <td>-13.566667</td>
      <td>-52.066667</td>
      <td>-36.9</td>
    </tr>
    <tr>
      <th>duration_minutes_median</th>
      <td>11.633333</td>
      <td>12.0</td>
      <td>12.466667</td>
    </tr>
    <tr>
      <th>duration_minutes_max</th>
      <td>9455.4</td>
      <td>5671.983333</td>
      <td>9092.55</td>
    </tr>
    <tr>
      <th>speed_mph_min</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>speed_mph_median</th>
      <td>9.570997</td>
      <td>9.372263</td>
      <td>9.467938</td>
    </tr>
    <tr>
      <th>speed_mph_max</th>
      <td>1443333.692308</td>
      <td>1906956.771429</td>
      <td>816167.538462</td>
    </tr>
  </tbody>
</table>
</div>


    
    Corrected trip quality audit saved.
    

## علامت‌گذاری اولیه ناهنجاری‌ها


```python

anomaly_summaries = []
anomaly_samples = []
invalid_dropoff_samples = []

ANALYSIS_VARIABLES = [
    "trip_distance",
    "duration_minutes",
    "fare_amount",
    "total_amount",
    "speed_mph"
]

for month in MONTHLY_FILES:

    path = PROCESSED_DIR / f"trip_features_{month}.parquet"
    df = pd.read_parquet(path)

    flags = pd.DataFrame(index=df.index)

    for col in ANALYSIS_VARIABLES:

        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)

        flags[f"{col}_outside_1_99"] = (
            df[col].notna() &
            ((df[col] < lower) | (df[col] > upper))
        )

    flags["negative_fare"] = df["fare_amount"] < 0
    flags["negative_total"] = df["total_amount"] < 0
    flags["negative_distance"] = df["trip_distance"] < 0

    flags["nonpositive_duration"] = (
        df["duration_minutes"] <= 0
    )

    flags["speed_over_60_mph"] = (
        df["speed_mph"] > 60
    )

    for flag in flags.columns:
        anomaly_summaries.append({
            "month": month,
            "flag": flag,
            "records_flagged": int(flags[flag].sum()),
            "percent_of_month": round(
                flags[flag].mean() * 100, 4
            )
        })

    # نمونه رکوردهای علامت‌گذاری‌شده
    flagged_any = flags.any(axis=1)

    sample = df.loc[
        flagged_any,
        [
            "pickup_time_local",
            "dropoff_time_local",
            "PULocationID",
            "DOLocationID",
            "trip_distance",
            "duration_minutes",
            "fare_amount",
            "total_amount",
            "speed_mph"
        ]
    ].head(10).copy()

    sample.insert(0, "month", month)
    anomaly_samples.append(sample)

    # بررسی جداگانه زمان‌های پیاده‌شدن نامعتبر
    invalid_dropoff = df.loc[
        df["dropoff_time_local"].isna(),
        [
            "pickup_time_local",
            "dropoff_time_local",
            "PULocationID",
            "DOLocationID",
            "trip_distance",
            "duration_minutes"
        ]
    ].head(10).copy()

    invalid_dropoff.insert(0, "month", month)
    invalid_dropoff_samples.append(invalid_dropoff)

    del df, flags

anomaly_summary_df = pd.DataFrame(anomaly_summaries)

anomaly_sample_df = pd.concat(
    anomaly_samples,
    ignore_index=True
)

invalid_dropoff_sample_df = pd.concat(
    invalid_dropoff_samples,
    ignore_index=True
)

display(
    anomaly_summary_df.sort_values(
        ["month", "records_flagged"],
        ascending=[True, False]
    )
)

print("\nSample flagged records:")
display(anomaly_sample_df)

print("\nInvalid dropoff timestamp records:")
display(invalid_dropoff_sample_df)

anomaly_summary_df.to_csv(
    INTERIM_DIR / "anomaly_summary_corrected.csv",
    index=False
)

anomaly_sample_df.to_parquet(
    INTERIM_DIR / "anomaly_sample_records_corrected.parquet",
    index=False
)

invalid_dropoff_sample_df.to_csv(
    INTERIM_DIR / "invalid_dropoff_samples.csv",
    index=False
)

print("\nCorrected anomaly audit outputs saved.")

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
      <th>month</th>
      <th>flag</th>
      <th>records_flagged</th>
      <th>percent_of_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2024-01</td>
      <td>duration_minutes_outside_1_99</td>
      <td>59066</td>
      <td>1.9924</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-01</td>
      <td>total_amount_outside_1_99</td>
      <td>58754</td>
      <td>1.9818</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-01</td>
      <td>fare_amount_outside_1_99</td>
      <td>58258</td>
      <td>1.9651</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2024-01</td>
      <td>negative_fare</td>
      <td>37447</td>
      <td>1.2631</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2024-01</td>
      <td>negative_total</td>
      <td>35503</td>
      <td>1.1976</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-01</td>
      <td>speed_mph_outside_1_99</td>
      <td>29637</td>
      <td>0.9997</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2024-01</td>
      <td>trip_distance_outside_1_99</td>
      <td>29519</td>
      <td>0.9957</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2024-01</td>
      <td>speed_over_60_mph</td>
      <td>1289</td>
      <td>0.0435</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2024-01</td>
      <td>nonpositive_duration</td>
      <td>870</td>
      <td>0.0293</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2024-01</td>
      <td>negative_distance</td>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2024-02</td>
      <td>duration_minutes_outside_1_99</td>
      <td>60046</td>
      <td>1.9965</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2024-02</td>
      <td>fare_amount_outside_1_99</td>
      <td>59439</td>
      <td>1.9763</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2024-02</td>
      <td>total_amount_outside_1_99</td>
      <td>58666</td>
      <td>1.9506</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2024-02</td>
      <td>negative_fare</td>
      <td>40655</td>
      <td>1.3518</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2024-02</td>
      <td>negative_total</td>
      <td>36080</td>
      <td>1.1997</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2024-02</td>
      <td>speed_mph_outside_1_99</td>
      <td>30068</td>
      <td>0.9998</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2024-02</td>
      <td>trip_distance_outside_1_99</td>
      <td>30043</td>
      <td>0.9989</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2024-02</td>
      <td>speed_over_60_mph</td>
      <td>1124</td>
      <td>0.0374</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2024-02</td>
      <td>nonpositive_duration</td>
      <td>803</td>
      <td>0.0267</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2024-02</td>
      <td>negative_distance</td>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2024-03</td>
      <td>duration_minutes_outside_1_99</td>
      <td>71463</td>
      <td>1.9947</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2024-03</td>
      <td>total_amount_outside_1_99</td>
      <td>71303</td>
      <td>1.9902</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2024-03</td>
      <td>fare_amount_outside_1_99</td>
      <td>70603</td>
      <td>1.9707</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2024-03</td>
      <td>negative_fare</td>
      <td>58464</td>
      <td>1.6319</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2024-03</td>
      <td>negative_total</td>
      <td>44311</td>
      <td>1.2368</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2024-03</td>
      <td>speed_mph_outside_1_99</td>
      <td>35815</td>
      <td>0.9997</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2024-03</td>
      <td>trip_distance_outside_1_99</td>
      <td>35576</td>
      <td>0.9930</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2024-03</td>
      <td>speed_over_60_mph</td>
      <td>1435</td>
      <td>0.0401</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2024-03</td>
      <td>nonpositive_duration</td>
      <td>1128</td>
      <td>0.0315</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2024-03</td>
      <td>negative_distance</td>
      <td>0</td>
      <td>0.0000</td>
    </tr>
  </tbody>
</table>
</div>


    
    Sample flagged records:
    


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
      <th>month</th>
      <th>pickup_time_local</th>
      <th>dropoff_time_local</th>
      <th>PULocationID</th>
      <th>DOLocationID</th>
      <th>trip_distance</th>
      <th>duration_minutes</th>
      <th>fare_amount</th>
      <th>total_amount</th>
      <th>speed_mph</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-01</td>
      <td>2024-01-01 00:52:09-05:00</td>
      <td>2024-01-01 00:52:28-05:00</td>
      <td>237</td>
      <td>237</td>
      <td>0.00</td>
      <td>0.316667</td>
      <td>3.00</td>
      <td>8.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-01</td>
      <td>2024-01-01 00:14:29-05:00</td>
      <td>2024-01-01 00:14:29-05:00</td>
      <td>236</td>
      <td>264</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>3.00</td>
      <td>8.00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-01</td>
      <td>2024-01-01 00:42:05-05:00</td>
      <td>2024-01-01 01:16:49-05:00</td>
      <td>263</td>
      <td>265</td>
      <td>23.90</td>
      <td>34.733333</td>
      <td>120.00</td>
      <td>127.94</td>
      <td>41.285988</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-01</td>
      <td>2024-01-01 00:15:26-05:00</td>
      <td>2024-01-01 00:33:57-05:00</td>
      <td>262</td>
      <td>171</td>
      <td>11.51</td>
      <td>18.516667</td>
      <td>44.30</td>
      <td>67.49</td>
      <td>37.296130</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-01</td>
      <td>2024-01-01 00:46:16-05:00</td>
      <td>2024-01-01 01:20:10-05:00</td>
      <td>132</td>
      <td>239</td>
      <td>20.85</td>
      <td>33.900000</td>
      <td>70.00</td>
      <td>82.69</td>
      <td>36.902655</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2024-01</td>
      <td>2024-01-01 00:50:28-05:00</td>
      <td>2024-01-01 01:38:39-05:00</td>
      <td>132</td>
      <td>26</td>
      <td>20.34</td>
      <td>48.183333</td>
      <td>80.00</td>
      <td>86.25</td>
      <td>25.328260</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2024-01</td>
      <td>2024-01-01 00:18:24-05:00</td>
      <td>2024-01-01 00:30:39-05:00</td>
      <td>249</td>
      <td>232</td>
      <td>2.16</td>
      <td>12.250000</td>
      <td>-13.50</td>
      <td>-18.50</td>
      <td>10.579592</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2024-01</td>
      <td>2024-01-01 00:06:29-05:00</td>
      <td>2024-01-01 00:31:26-05:00</td>
      <td>132</td>
      <td>170</td>
      <td>16.40</td>
      <td>24.950000</td>
      <td>70.00</td>
      <td>82.69</td>
      <td>39.438878</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2024-01</td>
      <td>2024-01-01 00:16:21-05:00</td>
      <td>2024-01-01 00:46:36-05:00</td>
      <td>70</td>
      <td>265</td>
      <td>23.00</td>
      <td>30.250000</td>
      <td>125.50</td>
      <td>134.75</td>
      <td>45.619835</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2024-01</td>
      <td>2024-01-01 00:55:31-05:00</td>
      <td>2024-01-01 01:21:15-05:00</td>
      <td>132</td>
      <td>213</td>
      <td>18.42</td>
      <td>25.733333</td>
      <td>68.10</td>
      <td>79.29</td>
      <td>42.948187</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2024-02</td>
      <td>2024-02-01 00:07:50-05:00</td>
      <td>2024-02-01 00:43:12-05:00</td>
      <td>132</td>
      <td>261</td>
      <td>28.69</td>
      <td>35.366667</td>
      <td>70.00</td>
      <td>82.69</td>
      <td>48.672950</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2024-02</td>
      <td>2024-02-01 00:34:58-05:00</td>
      <td>2024-02-01 00:35:16-05:00</td>
      <td>170</td>
      <td>170</td>
      <td>0.07</td>
      <td>0.300000</td>
      <td>3.00</td>
      <td>8.00</td>
      <td>14.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2024-02</td>
      <td>2024-02-01 00:03:06-05:00</td>
      <td>2024-02-01 00:14:21-05:00</td>
      <td>239</td>
      <td>244</td>
      <td>5.55</td>
      <td>11.250000</td>
      <td>-23.30</td>
      <td>-28.30</td>
      <td>29.600000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2024-02</td>
      <td>2024-02-01 00:45:54-05:00</td>
      <td>2024-02-01 01:34:40-05:00</td>
      <td>132</td>
      <td>140</td>
      <td>20.30</td>
      <td>48.766667</td>
      <td>70.00</td>
      <td>92.69</td>
      <td>24.976077</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2024-02</td>
      <td>2024-02-01 00:10:01-05:00</td>
      <td>2024-02-01 00:21:10-05:00</td>
      <td>263</td>
      <td>48</td>
      <td>2.70</td>
      <td>11.150000</td>
      <td>-14.90</td>
      <td>-19.90</td>
      <td>14.529148</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2024-02</td>
      <td>2024-02-01 00:28:54-05:00</td>
      <td>2024-02-01 00:29:18-05:00</td>
      <td>48</td>
      <td>48</td>
      <td>0.00</td>
      <td>0.400000</td>
      <td>19.99</td>
      <td>23.49</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2024-02</td>
      <td>2024-02-01 00:07:47-05:00</td>
      <td>2024-02-01 00:46:01-05:00</td>
      <td>132</td>
      <td>237</td>
      <td>20.20</td>
      <td>38.233333</td>
      <td>70.00</td>
      <td>90.96</td>
      <td>31.700087</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2024-02</td>
      <td>2024-02-01 00:28:01-05:00</td>
      <td>2024-02-01 01:16:16-05:00</td>
      <td>138</td>
      <td>1</td>
      <td>31.83</td>
      <td>48.250000</td>
      <td>136.40</td>
      <td>165.47</td>
      <td>39.581347</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2024-02</td>
      <td>2024-02-01 00:14:19-05:00</td>
      <td>2024-02-01 00:41:08-05:00</td>
      <td>263</td>
      <td>229</td>
      <td>1.80</td>
      <td>26.816667</td>
      <td>-23.30</td>
      <td>-28.30</td>
      <td>4.027346</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2024-02</td>
      <td>2024-02-01 00:15:52-05:00</td>
      <td>2024-02-01 00:34:34-05:00</td>
      <td>239</td>
      <td>79</td>
      <td>6.63</td>
      <td>18.700000</td>
      <td>-30.30</td>
      <td>-35.30</td>
      <td>21.272727</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2024-03</td>
      <td>2024-03-01 00:06:56-05:00</td>
      <td>2024-03-01 00:23:12-05:00</td>
      <td>132</td>
      <td>39</td>
      <td>9.60</td>
      <td>16.266667</td>
      <td>37.30</td>
      <td>41.55</td>
      <td>35.409836</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2024-03</td>
      <td>2024-03-01 00:46:20-05:00</td>
      <td>2024-03-01 01:02:38-05:00</td>
      <td>132</td>
      <td>39</td>
      <td>9.44</td>
      <td>16.300000</td>
      <td>37.30</td>
      <td>41.55</td>
      <td>34.748466</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2024-03</td>
      <td>2024-03-01 00:03:54-05:00</td>
      <td>2024-03-01 00:43:18-05:00</td>
      <td>132</td>
      <td>244</td>
      <td>27.39</td>
      <td>39.400000</td>
      <td>70.00</td>
      <td>96.23</td>
      <td>41.710660</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2024-03</td>
      <td>2024-03-01 00:19:19-05:00</td>
      <td>2024-03-01 01:09:29-05:00</td>
      <td>132</td>
      <td>265</td>
      <td>38.36</td>
      <td>50.166667</td>
      <td>191.30</td>
      <td>212.49</td>
      <td>45.879070</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2024-03</td>
      <td>2024-03-01 00:33:51-05:00</td>
      <td>2024-03-01 00:58:25-05:00</td>
      <td>68</td>
      <td>80</td>
      <td>5.93</td>
      <td>24.566667</td>
      <td>-29.60</td>
      <td>-34.60</td>
      <td>14.483039</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2024-03</td>
      <td>2024-03-01 00:15:32-05:00</td>
      <td>2024-03-01 00:55:43-05:00</td>
      <td>132</td>
      <td>1</td>
      <td>37.21</td>
      <td>40.183333</td>
      <td>153.20</td>
      <td>223.99</td>
      <td>55.560348</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2024-03</td>
      <td>2024-03-01 00:25:48-05:00</td>
      <td>2024-03-01 00:26:04-05:00</td>
      <td>107</td>
      <td>107</td>
      <td>0.00</td>
      <td>0.266667</td>
      <td>15.70</td>
      <td>24.62</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2024-03</td>
      <td>2024-03-01 00:37:46-05:00</td>
      <td>2024-03-01 01:11:44-05:00</td>
      <td>132</td>
      <td>181</td>
      <td>26.60</td>
      <td>33.966667</td>
      <td>97.50</td>
      <td>120.00</td>
      <td>46.987242</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2024-03</td>
      <td>2024-03-01 00:41:47-05:00</td>
      <td>2024-03-01 01:21:58-05:00</td>
      <td>132</td>
      <td>231</td>
      <td>20.61</td>
      <td>40.183333</td>
      <td>70.00</td>
      <td>90.55</td>
      <td>30.773953</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2024-03</td>
      <td>2024-03-01 00:27:52-05:00</td>
      <td>2024-03-01 00:28:16-05:00</td>
      <td>79</td>
      <td>79</td>
      <td>0.00</td>
      <td>0.400000</td>
      <td>3.00</td>
      <td>9.60</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>


    
    Invalid dropoff timestamp records:
    


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
      <th>month</th>
      <th>pickup_time_local</th>
      <th>dropoff_time_local</th>
      <th>PULocationID</th>
      <th>DOLocationID</th>
      <th>trip_distance</th>
      <th>duration_minutes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-03</td>
      <td>2024-03-09 02:31:02-05:00</td>
      <td>NaT</td>
      <td>114</td>
      <td>48</td>
      <td>3.36</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-03</td>
      <td>2024-03-09 02:24:19-05:00</td>
      <td>NaT</td>
      <td>79</td>
      <td>230</td>
      <td>3.34</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-03</td>
      <td>2024-03-09 03:13:26-05:00</td>
      <td>NaT</td>
      <td>114</td>
      <td>170</td>
      <td>1.91</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


    
    Corrected anomaly audit outputs saved.
    
