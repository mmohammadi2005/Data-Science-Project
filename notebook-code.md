# NYC TLC
## پیش‌بینی تعداد سفرهای تاکسی در هر ساعت در شهر نیویورک با استفاده از داده‌های تاریخی سفرها
<div dir="rtl" style="text-align: right;">
### معرفی پروژه و هدف دفترچه

هدف از این دفترچه، شناخت مسئله پژوهش، آشنایی با ساختار و ویژگی‌های داده‌ها، ارزیابی کیفیت داده‌ها و انجام تحلیل‌های اکتشافی (Exploratory Data Analysis) در چارچوب فاز اول پروژه تحلیل داده است.

در این پروژه، مجموعه‌داده سفرهای تاکسی زرد شهر نیویورک (NYC TLC Yellow Taxi Trip Records) به‌عنوان منبع داده انتخاب شده است. به‌منظور مدیریت حجم داده‌ها و محدود کردن دامنه بررسی، سه ماه نخست سال ۲۰۲۴ (ژانویه، فوریه و مارس) به‌عنوان بازه زمانی مطالعه در نظر گرفته شده است.

مسئله اصلی پژوهش این است:

**آیا می‌توان با استفاده از تعداد سفرهای ثبت‌شده در ساعت‌های گذشته، تعداد سفرهای ساعت آینده را پیش‌بینی کرد؟**

تعداد سفرهای تاکسی در ساعات مختلف شبانه‌روز و روزهای متفاوت هفته یکسان نیست و تحت تأثیر الگوهای زمانی تقاضا تغییر می‌کند. پیش‌بینی تعداد سفرها در ساعات آینده می‌تواند به شناخت بهتر این الگوها و برنامه‌ریزی مؤثرتر برای تخصیص تاکسی‌ها و مدیریت خدمات حمل‌ونقل شهری کمک کند.

بر این اساس، در این فاز ابتدا داده‌های خام دریافت و ساختار آن‌ها بررسی می‌شود؛ سپس کیفیت داده‌ها، مقادیر گمشده، داده‌های نامعتبر و سایر ناهنجاری‌های احتمالی مورد ارزیابی قرار می‌گیرند. در ادامه، با انجام تحلیل‌های اکتشافی، شناخت اولیه‌ای از ویژگی‌ها و الگوهای موجود در داده‌ها به دست می‌آید تا زمینه مناسبی برای ادامه پروژه فراهم شود.
</div>
## ۱. نصب وابستگی‌ها (در صورت نیاز)
<div dir="rtl" style="text-align: right;">
بررسی نصب کتابخانه pyarrow در پایتون 
</div>
%pip install pandas pyarrow matplotlib

## تنطیم مسیر پروژه که فایل کد و نوت بوک ها که در آن سیو شود
# =========================================================
# تنظیم مسیر پروژه با مسیر ثابت (Hardcoded) برای Downloads
# =========================================================
import os
from pathlib import Path

PROJECT_ROOT = Path("C:/Users/Asus/Downloads")
DATA_DIR = PROJECT_ROOT / "data" / "raw"
FIGURES_DIR = PROJECT_ROOT / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"

# ساخت پوشه‌ها
for d in [DATA_DIR, FIGURES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("✅ مسیرها با موفقیت تنظیم شد:")
print("FIGURES_DIR:", FIGURES_DIR)
## ۲. تنظیم مسیرها و منابع رسمی

from pathlib import Path
from urllib.request import urlretrieve
import pandas as pd
import pyarrow.parquet as pq

PROJECT_DIR = Path.cwd()  #مسیر پوشهٔ پروژه را مشخص می‌کنند.
DATA_DIR = PROJECT_DIR / 'data' / 'raw' #پوشه‌ای به نام data/raw را برای ذخیرهٔ فایل‌های خام داده تعیین می‌کند.
DATA_DIR.mkdir(parents=True, exist_ok=True)  #اگر پوشه وجود نداشته باشد، آن را ایجاد می‌کند.

FILES = {
    '2024-01': 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet',
    '2024-02': 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-02.parquet',
    '2024-03': 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-03.parquet',
}

#مسیرها و فهرست فایل‌ها را نمایش می‌دهند تا بتوانیم تنظیمات را بررسی کنیم.:
print('Project directory:', PROJECT_DIR.resolve())
print('Data directory:', DATA_DIR.resolve())
print('Files to inspect:', list(FILES))

## ۳. دریافت فایل‌های ماهانه
<div dir="rtl" style="text-align: right;">
**این سلول فایل‌های رسمی را فقط در صورتی دانلود می‌کند که نسخه‌ای از آن‌ها در پوشه محلی وجود نداشته باشد.** اگر دانلود متوقف شد یا کامل نشد، سلول را دوباره اجرا کنید.
</div>
local_files = {}  #یک دیکشنری خالی برای نگهداری مسیر فایل‌ها می‌سازد.
for month, url in FILES.items(): #لینک هر سه ماه را یکی‌یکی برمی‌دارد.
    path = DATA_DIR / f'yellow_tripdata_{month}.parquet'
    if not path.exists():  #بررسی می‌کند که آیا فایل قبلاً دانلود شده است یا خیر.
        print(f'Downloading {month} ...')
        urlretrieve(url, path)
    else:
        print(f'Already downloaded: {month}')
    local_files[month] = path
    print(month, 'size (MB):', round(path.stat().st_size / (1024**2), 2))

##  بررسی ساختار فایل‌ها و ستون زمان سوارشدن

schema_rows = []       #یک فهرست خالی برای نگهداری گزارش می‌سازد
for month, path in local_files.items():  #فایل‌های سه ماه را یکی‌یکی بررسی می‌کند
    pf = pq.ParquetFile(path)
    names = pf.schema.names    #نام ستون‌ها را استخراج می‌کند
    schema_rows.append({    # اضافه می‌کنیم. اینجا اطلاعات مربوط به فایل فعلی را در قالب یک دیکشنری می‌سازیم و به فهرست 
        'month': month,
        'rows_in_file_metadata': pf.metadata.num_rows,
        'columns': len(names),
        'has_pickup_timestamp': 'tpep_pickup_datetime' in names,   #بررسی وجود ستون زمان شروع سفر
    
        'column_names': ', '.join(names),   #ثبت فهرست کامل نام ستون‌ها
    })

#تبدیل اطلاعات به جدول
schema_report = pd.DataFrame(schema_rows)
display(schema_report)
#بررسی شرط نهایی و جلوگیری از ادامه کار با ساختار نامناسب
if not schema_report['has_pickup_timestamp'].all():
    raise ValueError('ستون tpep_pickup_datetime در یک یا چند فایل پیدا نشد؛ پیش از ادامه، ساختار داده بررسی شود.')

# =========================================================
# بررسی سازگاری ساختار فایل‌ها برای ترکیب و تحلیل آن‌ها
# مقایسه نام و نوع ستون‌ها در فایل‌های ماهانه
# =========================================================

import pandas as pd
import pyarrow.parquet as pq

# مرحله 1: استخراج ساختار هر فایل ماهانه

schema_by_file = {}

for month, file_path in local_files.items():

    parquet_schema = pq.read_schema(file_path)

    schema_by_file[month] = {
        field.name: str(field.type)
        for field in parquet_schema
    }


# مرحله 2: استخراج تمام نام ستون‌های موجود در فایل‌ها

all_column_names = sorted(
    set().union(
        *(set(schema.keys()) for schema in schema_by_file.values())
    )
)


# مرحله 3: مقایسه ستون‌ها و نوع داده‌ها

schema_comparison_rows = []

for column in all_column_names:

    row = {"column": column}

    for month in local_files:

        row[f"{month}_type"] = schema_by_file[month].get(
            column,
            "COLUMN NOT PRESENT"
        )

    types = [
        schema_by_file[month].get(column)
        for month in local_files
    ]

    row["consistent_across_files"] = (
        len(set(types)) == 1 and None not in types
    )

    schema_comparison_rows.append(row)


# مرحله 4: ایجاد و نمایش جدول مقایسه

schema_comparison_df = pd.DataFrame(schema_comparison_rows)

display(schema_comparison_df)

# =========================================================
# پایان بررسی سازگاری ساختار فایل‌ها
# =========================================================
# =========================================================
# مشخص‌کردن ماهیت هر ستون و نقش احتمالی آن در تحلیل داده‌ها.
# دسته‌بندی اولیه ستون‌ها
# =========================================================

columns = list(schema_by_file[next(iter(local_files))].keys())

column_roles = []

for column in columns:

    name = column.lower()

    if "datetime" in name or "date" in name or "time" in name:
        role = "Temporal"

    elif "locationid" in name or name.endswith("_id") or name == "vendorid":
        role = "Identifier / Categorical Code"

    elif name in {
        "passenger_count",
        "trip_distance",
        "fare_amount",
        "extra",
        "mta_tax",
        "tip_amount",
        "tolls_amount",
        "improvement_surcharge",
        "total_amount",
        "congestion_surcharge",
        "airport_fee",
        "cbd_congestion_fee",
    }:
        role = "Numeric"

    elif "payment" in name or "ratecode" in name or "store_and_fwd" in name:
        role = "Categorical / Code"

    else:
        role = "Needs semantic review"

    column_roles.append({
        "column": column,
        "data_type": schema_by_file[next(iter(local_files))][column],
        "proposed_role": role,
    })

column_roles_df = pd.DataFrame(column_roles)

display(column_roles_df)
## ممیز کیفیت 
### رکوردهای تکراری 
import pandas as pd

# ============================================================
# Duplicate Record Audit — بررسی رکوردهای تکراری
# هدف: شناسایی رکوردهای کاملاً یکسان در سه فایل Yellow Taxi
# توجه: هیچ رکوردی در این مرحله حذف نمی‌شود.
# ============================================================

monthly_hashes = {}
monthly_duplicate_results = []

for month, path in local_files.items():

    print(f"Checking duplicate records: {month}")

    # خواندن تمام ستون‌های فایل ماهانه
    df = pd.read_parquet(path)

    # محاسبه اثرانگشت هر ردیف بر اساس تمام ستون‌ها
    row_hash = pd.util.hash_pandas_object(
        df,
        index=False
    )

    # ذخیره اثرانگشت‌ها برای بررسی همان ماه و کل سه ماه
    monthly_hashes[month] = row_hash.reset_index(drop=True)

    # بررسی تکرار درون همان فایل
    duplicate_mask = row_hash.duplicated(keep=False)
    extra_duplicate_mask = row_hash.duplicated(keep="first")

    total_records = len(df)

    monthly_duplicate_results.append({
        "month": month,
        "total_records": total_records,
        "rows_in_duplicate_groups": int(duplicate_mask.sum()),
        "extra_duplicate_rows": int(extra_duplicate_mask.sum()),
        "rows_in_duplicate_groups_percent": round(
            duplicate_mask.mean() * 100, 4
        ) if total_records else 0
    })

    print("Total records:", total_records)
    print("Rows in duplicate groups:", int(duplicate_mask.sum()))
    print("Extra duplicate rows:", int(extra_duplicate_mask.sum()))

    del df, row_hash, duplicate_mask, extra_duplicate_mask


# ============================================================
# Summary 1 — Duplicate audit within each monthly file
# ============================================================

duplicate_audit_monthly = pd.DataFrame(
    monthly_duplicate_results
).sort_values("month").reset_index(drop=True)

print("\nTABLE 1 — Duplicate Audit by Month")

display(
    duplicate_audit_monthly.style.format({
        "rows_in_duplicate_groups": "{:,.0f}",
        "extra_duplicate_rows": "{:,.0f}",
        "rows_in_duplicate_groups_percent": "{:.4f}"
    })
)


# ============================================================
# Summary 2 — Duplicate screening across all three files
# ============================================================

all_row_hashes = pd.concat(
    monthly_hashes,
    names=["month", "original_row_index"]
).reset_index(level="month")

all_row_hashes = all_row_hashes.rename(
    columns={0: "row_hash"}
)

# شناسایی ردیف‌هایی که اثرانگشتشان در کل سه فایل تکرار شده است
cross_file_duplicate_mask = all_row_hashes["row_hash"].duplicated(
    keep=False
)

# شمارش ردیف‌های اضافی پس از اولین مشاهده هر اثرانگشت
cross_file_extra_mask = all_row_hashes["row_hash"].duplicated(
    keep="first"
)

print("\nTABLE 2 — Duplicate Screening Across All Three Months")

print("Total records screened:", len(all_row_hashes))
print(
    "Rows in duplicate groups across all files:",
    int(cross_file_duplicate_mask.sum())
)
print(
    "Extra duplicate rows across all files:",
    int(cross_file_extra_mask.sum())
)

# خلاصه تعداد ردیف‌های عضو گروه‌های تکراری به تفکیک ماه
cross_file_monthly_summary = (
    all_row_hashes.assign(
        in_duplicate_group=cross_file_duplicate_mask.to_numpy(),
        is_extra_duplicate=cross_file_extra_mask.to_numpy()
    )
    .groupby("month", as_index=False)
    .agg(
        rows_screened=("row_hash", "size"),
        rows_in_duplicate_groups=("in_duplicate_group", "sum"),
        extra_duplicate_rows=("is_extra_duplicate", "sum")
    )
    .sort_values("month")
    .reset_index(drop=True)
)

display(cross_file_monthly_summary)
# ==============================================================
# Inspect and verify duplicate rows in February 2024
# بررسی و اعتبارسنجی ردیف‌های تکراری در فوریه ۲۰۲۴
# ==============================================================

feb_path = local_files["2024-02"]
feb_df = pd.read_parquet(feb_path)

# شناسایی رکوردهای تکراری بر اساس تمام ستون‌ها
duplicate_mask = feb_df.duplicated(keep=False)
duplicate_rows = feb_df.loc[duplicate_mask].copy()

print("Number of rows in duplicate groups:", len(duplicate_rows))
print("Number of columns:", len(feb_df.columns))

if len(duplicate_rows) > 0:
    # مقایسه مستقیم دو ردیف اول برای اطمینان از برابری کامل
    rows_identical = duplicate_rows.iloc[0].equals(duplicate_rows.iloc[1])
    
    # پیدا کردن ستون‌هایی که احیاناً مقادیر متفاوتی دارند
    different_columns = [
        col for col in feb_df.columns
        if not (
            pd.isna(duplicate_rows.iloc[0][col]) 
            and pd.isna(duplicate_rows.iloc[1][col])
        ) and duplicate_rows.iloc[0][col] == duplicate_rows.iloc[1][col] == False # یا مقایسه مساوی بودن
    ]
    
    print("Are all column values identical?", rows_identical)
    print("Columns with different values:", different_columns)
    
    display(duplicate_rows)
else:
    print("No duplicate rows found.")

### مقادیر گمشده 

# ============================================================
# Data Quality Audit — Missingness Across All Columns
#مقادیر گمشده با با مقادیر نامرتبط فرق دارد
# محاسبه تعداد و درصد مقادیر گمشده در تمام ستون‌های هر سه فایل
# ============================================================

missingness_rows = []

for month, path in local_files.items():

    parquet_file = pq.ParquetFile(path)
    total_rows = parquet_file.metadata.num_rows
    columns = parquet_file.schema_arrow.names

    for column in columns:

        table = pq.read_table(path, columns=[column])
        missing_count = table.column(0).null_count

        missing_percent = (
            missing_count / total_rows * 100
            if total_rows > 0
            else 0
        )

        missingness_rows.append({
            "month": month,
            "column": column,
            "data_type": schema_by_file[month].get(column),
            "total_records": total_rows,
            "missing_count": missing_count,
            "missing_percent": round(missing_percent, 4)
        })

missingness_report = pd.DataFrame(missingness_rows)

print("Missingness report for all columns:")
display(
    missingness_report.sort_values(
        ["month", "missing_percent", "column"],
        ascending=[True, False, True]
    ).reset_index(drop=True)
)

# خلاصه مقادیر گمشده برای هر ماه
missingness_summary = (
    missingness_report
    .assign(has_missing=lambda df: df["missing_count"] > 0)
    .groupby("month")
    .agg(
        total_columns=("column", "count"),
        columns_with_missing=("has_missing", "sum"),
        total_missing_values=("missing_count", "sum")
    )
    .reset_index()
)

print("Missingness summary by month:")
display(missingness_summary)


# بررسی اینکه آیا مقادیر گمشده در پنج ستون، در رکوردهای یکسان رخ می‌دهند

columns_to_check = [
    "Airport_fee",
    "RatecodeID",
    "congestion_surcharge",
    "passenger_count",
    "store_and_fwd_flag"
]

co_missingness_rows = []

for month, path in local_files.items():
    table = pq.read_table(path, columns=columns_to_check)
    df_check = table.to_pandas()

    missing_mask = df_check[columns_to_check].isna()

    co_missingness_rows.append({
        "month": month,
        "rows_with_any_missing": int(missing_mask.any(axis=1).sum()),
        "rows_with_all_five_missing": int(missing_mask.all(axis=1).sum()),
        "total_rows": len(df_check)
    })

co_missingness_report = pd.DataFrame(co_missingness_rows)
display(co_missingness_report)

import matplotlib.pyplot as plt
import pandas as pd

# رسم نمودار درصد مقادیر گمشده در ستون‌ها
missingness_plot_data = missingness_report.pivot(
    index="column",
    columns="month",
    values="missing_percent"
)

ax = missingness_plot_data.plot(
    kind="bar",
    figsize=(14, 6)
)

ax.set_title("Percentage of Missing Values by Column and Month")
ax.set_xlabel("Column")
ax.set_ylabel("Missing Values (%)")
ax.legend(title="Month")
ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt

missingness_heatmap_data = missingness_report.pivot(
    index="column",
    columns="month",
    values="missing_percent"
)

fig, ax = plt.subplots(figsize=(8, 9))

im = ax.imshow(
    missingness_heatmap_data.values,
    aspect="auto",
    interpolation="nearest"
)

ax.set_xticks(range(len(missingness_heatmap_data.columns)))
ax.set_xticklabels(missingness_heatmap_data.columns)

ax.set_yticks(range(len(missingness_heatmap_data.index)))
ax.set_yticklabels(missingness_heatmap_data.index)

ax.set_title("Missing Values (%) by Column and Month")
ax.set_xlabel("Month")
ax.set_ylabel("Column")

fig.colorbar(im, ax=ax, label="Missing Values (%)")
plt.tight_layout()
from pathlib import Path
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIGURES_DIR / "missing_values_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

##  ثبت مشخصات زیرمجموعهٔ نهایی

این جدول مشخص می‌کند که:
- سرویس انتخابی چیست؛
- منبع رسمی داده کجاست؛
- نام دقیق فایل‌های محلی چیست؛
- چه بازه‌ای برای پژوهش در نظر گرفته شده؛
- و رکوردها با چه قاعده‌ای انتخاب می‌شوند.
# =========================================================
# Research Period Definition
# تثبیت بازه زمانی پژوهش
# =========================================================

START = pd.Timestamp("2024-01-01 00:00:00")
END_EXCLUSIVE = pd.Timestamp("2024-04-01 00:00:00")

print("Research period:", START, "to", END_EXCLUSIVE)
print("The end timestamp is excluded.")
# =========================================================
# Dataset Manifest
# ثبت مشخصات زیرمجموعه داده انتخاب‌شده
# =========================================================

dataset_manifest = pd.DataFrame([
    {
        "service_type": "Yellow Taxi",

        "source": "NYC TLC Trip Record Data",

        "source_url": (
            "https://www.nyc.gov/site/tlc/about/"
            "tlc-trip-record-data.page"
        ),

        "month": month,

        "file_name": path.name,

        "file_size_MB": round(
            path.stat().st_size / (1024**2), 2
        ),

        "research_start": START,

        "research_end_exclusive": END_EXCLUSIVE,

        "time_column": "tpep_pickup_datetime",

        "timezone": "America/New_York",

        "timestamp_basis": "New York Local Time",

        "daylight_saving_time": "Applicable",

        "timestamp_timezone_status": "Naive datetime",

        "selection_rule": (
            "START <= pickup_time < END_EXCLUSIVE"
        ),
    }
    for month, path in local_files.items()
])

display(dataset_manifest)
### بررسی اولیه کیفیت زمان‌ها

**این سلول فقط ستون زمان سوارشدن را می‌خواند و برای هر ماه، تعداد مقادیر خالی یا نامعتبر و قدیمی‌ترین و جدیدترین زمان معتبر را گزارش می‌کند.**.

# =========================================================
# Phase 1 — Timestamp Audit with Duplicate Screening
# ممیزی زمان‌ها با کنترل رکوردهای تکراری
# Time Basis: America/New_York
# =========================================================

TIMEZONE = "America/New_York"

timestamp_frames = []
audit_rows = []

# بررسی فایل‌های سه ماه
for month, path in local_files.items():

    print(f"Processing {month}...")

    # خواندن زمان‌های سوارشدن
    table = pq.read_table(
        path,
        columns=["tpep_pickup_datetime"]
    )

    s = table.column("tpep_pickup_datetime").to_pandas()

    # بررسی وجود نتایج ممیزی تکراری‌ها
    if month not in monthly_hashes:
        raise ValueError(
            f"Duplicate audit results are missing for {month}. "
            "Run the duplicate audit cell first."
        )

    # دریافت اثرانگشت ردیف‌ها
    row_hash = monthly_hashes[month]

    # کنترل هماهنگی تعداد ردیف‌ها
    if len(s) != len(row_hash):
        raise ValueError(
            f"Row count mismatch for {month}: "
            f"{len(s)} timestamps vs {len(row_hash)} row hashes."
        )

    # نگه‌داشتن اولین رخداد هر ردیف کاملاً یکسان
    keep_mask = ~row_hash.duplicated(keep="first")

    # حذف ردیف‌های اضافی تکراری از جریان تحلیل
    s_clean = s.loc[keep_mask].copy()

    # تبدیل و بررسی زمان‌ها بدون تغییر منطقه زمانی
    parsed = pd.to_datetime(s_clean, errors="coerce")

    audit_rows.append({
        "month_file": month,
        "timezone": TIMEZONE,
        "timestamp_basis": "New York Local Time",
        "timestamp_timezone_status": "Naive datetime",
        "rows_read_raw": len(s),
        "extra_duplicate_rows_excluded": int((~keep_mask).sum()),
        "rows_after_duplicate_screening": len(s_clean),
        "missing_or_invalid_pickup_time": int(parsed.isna().sum()),
        "earliest_pickup_time": parsed.min(),
        "latest_pickup_time": parsed.max(),
    })

    # ذخیره زمان‌های معتبر
    timestamp_frames.append(
        pd.DataFrame({
            "pickup_time": parsed.dropna()
        })
    )

    print("Raw rows:", len(s))
    print("Extra duplicate rows excluded:", int((~keep_mask).sum()))
    print("Rows after duplicate screening:", len(s_clean))
    print("Missing or invalid pickup timestamps:", int(parsed.isna().sum()))


# ساخت جدول ممیزی زمانی
timestamp_audit = pd.DataFrame(audit_rows)

print("\nTIMESTAMP AUDIT AFTER DUPLICATE SCREENING")
display(timestamp_audit)


# ترکیب زمان‌های معتبر سه ماه
all_pickups = pd.concat(
    timestamp_frames,
    ignore_index=True
)["pickup_time"]

print(
    "\nTotal valid pickup timestamps after duplicate screening:",
    len(all_pickups)
)

print("Overall minimum:", all_pickups.min())
print("Overall maximum:", all_pickups.max())

print("Timezone reference:", TIMEZONE)
### آیا پوشش زمانی با بازه پیشنهادی هماهنگ است؟
<div dir="rtl" style="text-align: right;">
**بازه زمانی موردنظر پروژه از ابتدای ۱ ژانویه تا ابتدای ۱ آوریل ۲۰۲۴ است.** زمان انتهایی در این بازه لحاظ نمی‌شود؛ به این ترتیب، تمام زمان‌های ماه مارس در محدوده بررسی قرار می‌گیرند.
این شمارش بر اساس زمان‌های سوارشدن معتبر انجام می‌شود؛ بنابراین رکوردهایی که زمان سوارشدن خالی یا نامعتبر دارند در all_pickups نیستند
</div>
# =========================================================
# Research Window Audit
# ممیزی رکوردهای داخل و خارج بازه پژوهش
# =========================================================

TIMEZONE = "America/New_York"

# تثبیت بازه زمانی پژوهش
START = pd.Timestamp("2024-01-01 00:00:00")
END_EXCLUSIVE = pd.Timestamp("2024-04-01 00:00:00")

# شناسایی رکوردهای داخل بازه
inside_window = (
    all_pickups.ge(START) &
    all_pickups.lt(END_EXCLUSIVE)
)

# شمارش رکوردها
total_valid_timestamps = len(all_pickups)

inside_count = int(inside_window.sum())
outside_count = int((~inside_window).sum())

# محاسبه درصدها
inside_percent = (
    inside_count / total_valid_timestamps * 100
    if total_valid_timestamps else 0
)

outside_percent = (
    outside_count / total_valid_timestamps * 100
    if total_valid_timestamps else 0
)

# ساخت گزارش ممیزی بازه زمانی
range_audit = pd.DataFrame([
    {
        "category": "Inside research window",
        "start": START,
        "end_exclusive": END_EXCLUSIVE,
        "timezone": TIMEZONE,
        "timestamp_basis": "New York Local Time",
        "record_count": inside_count,
        "percentage_of_valid_timestamps": round(inside_percent, 4),
    },
    {
        "category": "Outside research window",
        "start": START,
        "end_exclusive": END_EXCLUSIVE,
        "timezone": TIMEZONE,
        "timestamp_basis": "New York Local Time",
        "record_count": outside_count,
        "percentage_of_valid_timestamps": round(outside_percent, 4),
    },
])

print("Research period:", START, "to", END_EXCLUSIVE)
print("Timezone:", TIMEZONE)
print("Timestamp basis: New York Local Time")
print("The end timestamp is excluded.")
print("Total valid pickup timestamps:", total_valid_timestamps)

display(range_audit)

# کنترل سازگاری شمارش
assert inside_count + outside_count == total_valid_timestamps
##  شمارش رکوردهای خام و تفکیک آن‌ها   

این جدول شمارش خام و زمان‌های معتبر را برای هر ماه کنار هم می‌گذارد.
# =========================================================
# خلاصه شمارش رکوردها برای هر فایل ماهانه
# =========================================================

raw_counts = schema_report[
    ["month", "rows_in_file_metadata"]
].copy()

raw_counts = raw_counts.rename(columns={
    "month": "month_file",
    "rows_in_file_metadata": "raw_records",
})

# افزودن تعداد زمان‌های سوارشدن خالی یا نامعتبر
raw_counts = raw_counts.merge(
    timestamp_audit[
        ["month_file", "missing_or_invalid_pickup_time"]
    ],
    on="month_file",
    how="left",
    validate="one_to_one",
)

# تعداد رکوردهای دارای زمان معتبر
raw_counts["valid_pickup_timestamps"] = (
    raw_counts["raw_records"]
    - raw_counts["missing_or_invalid_pickup_time"]
)

display(raw_counts)

# کنترل سازگاری تعداد رکوردها
assert (
    raw_counts["raw_records"]
    ==
    raw_counts["valid_pickup_timestamps"]
    + raw_counts["missing_or_invalid_pickup_time"]
).all()
# =========================================================
# Research Time-Window Filtering & Outlier Analysis
# =========================================================

# ۱. شناسایی رکوردهای داخل بازه (Masking)
inside_window_mask = (all_pickups.ge(START) & all_pickups.lt(END_EXCLUSIVE))
outside_window_mask = ~inside_window_mask

# ۲. جداسازی داده‌ها برای تحلیل و بررسی موارد خارج از بازه
analysis_pickups = all_pickups.loc[inside_window_mask].copy()
outside_window_timestamps = all_pickups.loc[outside_window_mask].sort_values()

# ۳. محاسبات آماری
total_count = len(all_pickups)
inside_count = len(analysis_pickups)
outside_count = len(outside_window_timestamps)

# ۴. گزارش نتایج کلی (خروجی فایل اول)
print(f"Raw records with valid pickup timestamps: {total_count}")
print(f"Records included in analysis window: {inside_count}")
print(f"Records outside analysis window: {outside_count}")

# ۵. کنترل سازگاری و صحت‌سنجی (Assertions)
assert inside_count + outside_count == total_count, "Error: Record count mismatch!"
assert analysis_pickups.ge(START).all() and analysis_pickups.lt(END_EXCLUSIVE).all(), "Error: Data leak outside window!"

# ۶. نمایش جزئیات رکوردهای خارج از بازه (خروجی فایل دوم)
print("\n--- Details of Records Outside the Research Window ---")
if outside_count > 0:
    display(outside_window_timestamps.to_frame(name="pickup_timestamp"))
else:
    print("No records found outside the defined window.")

## ساخت و ممیزی سری زمانی ساعتی با لحاظ منطقه زمانی و تغییر ساعت (DST) 
# =========================================================
# ساخت سری زمانی ساعتی با درنظرگرفتن منطقه زمانی نیویورک
# Hourly Trip Counts — DST-aware audit
# =========================================================

from pathlib import Path
import matplotlib.pyplot as plt

NY_TZ = "America/New_York"

# STEP 1 — انتخاب زمان‌های سوارشدن داخل بازه پژوهش
valid_in_window = all_pickups.loc[inside_window].copy()

# STEP 2 — تعریف منطقه زمانی نیویورک
valid_in_window_ny = valid_in_window.dt.tz_localize(
    NY_TZ,
    ambiguous="NaT",
    nonexistent="NaT"
)

unresolved_local_times = int(valid_in_window_ny.isna().sum())

print("Pickup timestamps inside research window:", len(valid_in_window))
print("Unresolved local timestamps after timezone localization:",
      unresolved_local_times)

# STEP 3 — ساخت شمارش مشاهده‌شده سفرها در هر ساعت محلی
hourly_counts_observed = (
    valid_in_window_ny
    .dropna()
    .dt.floor("h")
    .value_counts()
    .sort_index()
)

# STEP 4 — ساخت ابتدا و انتهای بازه با منطقه زمانی نیویورک
start_local = pd.Timestamp(START).tz_localize(NY_TZ)
end_local = pd.Timestamp(END_EXCLUSIVE).tz_localize(NY_TZ)

# STEP 5 — ساخت شاخص ساعت‌های واقعی در بازه پژوهش
full_hour_index = pd.date_range(
    start=start_local,
    end=end_local,
    freq="h",
    inclusive="left"
)

# STEP 6 — ساخت سری زمانی نهایی تعداد سفرها
hourly_counts = (
    hourly_counts_observed
    .reindex(full_hour_index, fill_value=0)
    .rename("trip_count")
)

# STEP 7 — گزارش کنترل‌های اولیه
print("\nExpected hourly rows:", len(full_hour_index))
print("Hourly series rows:", len(hourly_counts))
print("Hours with zero recorded trips:", int((hourly_counts == 0).sum()))

print("\nHourly index timezone:")
print(hourly_counts.index.tz)

print("\nHourly trip-count summary:")
display(hourly_counts.describe().to_frame("trip_count"))

print("\nFirst five hours:")
display(hourly_counts.head().to_frame())

print("\nLast five hours:")
display(hourly_counts.tail().to_frame())

# =========================================================
# STEP 8 — رسم و ذخیره نمودار سری زمانی
# =========================================================

# اطمینان از وجود پوشه ذخیره نمودارها
FIGURES_DIR = Path(FIGURES_DIR)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(14, 4))

hourly_counts.plot(
    ax=ax,
    title="Hourly Yellow Taxi Trip Counts — New York Local Time"
)

ax.set_xlabel("Pickup hour (America/New_York)")
ax.set_ylabel("Recorded trips")
ax.grid(alpha=0.3)

plt.tight_layout()

# ذخیره تصویر نمودار
plt.savefig(
    FIGURES_DIR / "hourly_trip_counts.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()
### بررسی اثر تغییر ساعت تابستانی (DST) روی داده‌ها
# =========================================================
# Daylight Saving Time (DST) Impact Documentation
# March 10, 2024 — America/New_York
# =========================================================

print("=" * 70)
print("DST IMPACT ON TIME-BASED ANALYSES (March 10, 2024)")
print("=" * 70)

# ---------------------------------------------------------
# STEP 1 — Number of days with data for each hour of day
# ---------------------------------------------------------
# Expectation: every hour has 91 days of data,
# except hour 02:00 which has only 90 days due to DST.

hours_per_hod = (
    hourly_counts
    .groupby(hourly_counts.index.hour)
    .size()
    .rename("days_with_data")
)

print("\n[1] Number of days with data for each hour of day:")
print("   (All hours should have 91 days, except hour 02:00")
print("    which has 90 days due to DST.)")
display(hours_per_hod.to_frame())

# ---------------------------------------------------------
# STEP 2 — Number of hours present on the DST day
# ---------------------------------------------------------
dst_day_start = pd.Timestamp("2024-03-10", tz=NY_TZ)
dst_day_end = dst_day_start + pd.Timedelta(days=1)

dst_day_hours = hourly_counts.loc[
    (hourly_counts.index >= dst_day_start) &
    (hourly_counts.index < dst_day_end)
]

print("\n[2] Number of hours present on the DST day (2024-03-10):")
print(f"   {len(dst_day_hours)} hours (instead of the usual 24)")
print("   Hour 02:00 does not exist on this day (it jumps to 03:00).")

display(dst_day_hours.to_frame("trip_count"))

# ---------------------------------------------------------
# STEP 3 — Number of days with data for each weekday
# ---------------------------------------------------------
# Sunday, March 10 has only 23 hours due to DST,
# so its daily trip total is slightly lower than a normal Sunday.

daily_counts = hourly_counts.resample("D").sum()

days_per_weekday = (
    daily_counts
    .groupby(daily_counts.index.dayofweek)
    .size()
    .rename("count")
)

weekday_labels = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}

days_per_weekday.index = days_per_weekday.index.map(weekday_labels)
days_per_weekday.index.name = "day_of_week"

print("\n[3] Number of days with data for each weekday:")
print("   (All weekdays should have 13 days; DST does not affect")
print("    the number of days, but it does affect the number of")
print("    hours on Sunday, March 10.)")
display(days_per_weekday.to_frame())

# ---------------------------------------------------------
# STEP 4 — DST impact summary
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("DST SUMMARY")
print("=" * 70)

dst_effects = pd.DataFrame([
    {
        "effect": "Hours present on March 10 (DST day)",
        "value": f"{len(dst_day_hours)} hours (instead of 24)",
    },
    {
        "effect": "Days of data for hour 02:00",
        "value": f"{hours_per_hod.loc[2]} days (instead of 91)",
    },
    {
        "effect": "Days of data for all other hours",
        "value": f"{hours_per_hod.drop(2).unique().tolist()} days",
    },
])

display(dst_effects)

print("\n✅ These effects are automatically taken into account")
print("   in all subsequent analyses, but are reported here")
print("   for transparency.")
# =========================================================
# بررسی کامل بودن سری زمانی ساعتی پس از اصلاح DST
# Phase 1 — Hourly Time Series Audit
# =========================================================

# STEP 1 — بررسی ساعت‌های بدون سفر ثبت‌شده
zero_trip_hours = hourly_counts[hourly_counts == 0]

print("=" * 60)
print("1. ZERO-TRIP HOURS")
print("=" * 60)

print("Number of zero-trip hours:", len(zero_trip_hours))
display(zero_trip_hours.to_frame("trip_count"))


# STEP 2 — بررسی تعداد ساعت‌ها و شاخص زمانی
print("\n" + "=" * 60)
print("2. HOURLY INDEX COMPLETENESS")
print("=" * 60)

print("Expected hourly timestamps:", len(full_hour_index))
print("Actual hourly timestamps:", len(hourly_counts))
print("Index contains duplicates:", hourly_counts.index.has_duplicates)
print("Index is monotonically increasing:", hourly_counts.index.is_monotonic_increasing)

print("Hourly index dtype:", hourly_counts.index.dtype)
print("Hourly index timezone:", hourly_counts.index.tz)


# STEP 3 — بررسی ساعت‌های فاقد رکورد معتبر در داده خام
observed_hours = (
    valid_in_window_ny
    .dropna()
    .dt.floor("h")
    .drop_duplicates()
    .sort_values()
)

missing_observed_hours = full_hour_index.difference(observed_hours)

print("\n" + "=" * 60)
print("3. HOURS WITHOUT VALID PICKUP RECORDS")
print("=" * 60)

print("Number of hours without valid pickup records:",
      len(missing_observed_hours))

display(
    pd.DataFrame({"hour": missing_observed_hours})
)


# STEP 4 — بررسی فاصله زمانی میان ساعت‌های متوالی
print("\n" + "=" * 60)
print("4. TIME INTERVAL AUDIT")
print("=" * 60)

time_differences = hourly_counts.index.to_series().diff().dropna()

print("Number of hourly intervals:", len(time_differences))
print("Minimum interval:", time_differences.min())
print("Maximum interval:", time_differences.max())

print("\nIntervals other than one hour:")
display(
    time_differences[time_differences != pd.Timedelta(hours=1)]
    .to_frame("interval")
)


# STEP 5 — بررسی تغییر ساعت تابستانی
dst_start = pd.Timestamp("2024-03-10 00:00:00", tz=NY_TZ)
dst_end = pd.Timestamp("2024-03-10 06:00:00", tz=NY_TZ)

print("\n" + "=" * 60)
print("5. DAYLIGHT SAVING TIME AUDIT")
print("=" * 60)

display(
    hourly_counts.loc[
        (hourly_counts.index >= dst_start) &
        (hourly_counts.index < dst_end)
    ].to_frame("trip_count")
)


# STEP 6 — خلاصه نهایی کنترل‌ها
audit_checks = pd.Series({
    "Expected and actual hourly rows match":
        len(full_hour_index) == len(hourly_counts),

    "No duplicate hourly timestamps":
        not hourly_counts.index.has_duplicates,

    "Hourly index is sorted":
        hourly_counts.index.is_monotonic_increasing,

    "Timezone is America/New_York":
        str(hourly_counts.index.tz) == NY_TZ,

    "No zero-trip hours":
        len(zero_trip_hours) == 0,

    "No hours missing valid pickup records":
        len(missing_observed_hours) == 0,

    "All consecutive intervals are one hour":
        bool((time_differences == pd.Timedelta(hours=1)).all())
})

print("\n" + "=" * 60)
print("6. FINAL AUDIT CHECKLIST")
print("=" * 60)

display(audit_checks.to_frame("passed"))

print("\nAll checks passed:", bool(audit_checks.all()))
## ۸. الگوهای تعداد سفرها را از سه زاویه بررسی کنیم
### تحلیل سفرها بر اساس ساعت شبانه‌روز
# =========================================================
# Hourly pattern of trip counts (based on New York local time)
# =========================================================

from pathlib import Path
import matplotlib.pyplot as plt

# Since hourly_counts already has tz = America/New_York,
# .index.hour directly gives the local NY hour.
hour_of_day = hourly_counts.index.hour

trips_by_hour = (
    hourly_counts
    .groupby(hour_of_day)
    .sum()
    .reindex(range(24), fill_value=0)
    .rename("total_trips")
)

# Note about DST
print("=" * 60)
print("HOURLY TRIP PATTERN — NEW YORK LOCAL TIME")
print("=" * 60)
print("Note: Hour 02:00 has only 90 days of data (instead of 91)")
print("      due to DST on March 10, 2024. This is reflected in the totals.")
print()

display(trips_by_hour.to_frame())

# ---------------------------------------------------------
# Plot: total trips by hour of day
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 4))

trips_by_hour.plot(
    kind="bar",
    ax=ax,
    color="steelblue",
    edgecolor="black",
    title="Total Yellow Taxi Trips by Hour of Day (New York Local Time)"
)

ax.set_xlabel("Hour of Day (0–23)")
ax.set_ylabel("Total Recorded Trips")
ax.grid(axis="y", alpha=0.3)
ax.tick_params(axis="x", rotation=0)

plt.tight_layout()

# Save figure
FIGURES_DIR = Path(FIGURES_DIR)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.savefig(
    FIGURES_DIR / "trips_by_hour_of_day.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print(f"✅ Figure saved: {FIGURES_DIR / 'trips_by_hour_of_day.png'}")
### محاسبهٔ میانگین سفرها در هر ساعت
# =========================================================
# Average trip counts by hour of day (New York local time)
# =========================================================

from pathlib import Path
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Aggregate statistics per hour of day
# ---------------------------------------------------------
trips_by_hour = hourly_counts.groupby(
    hourly_counts.index.hour
).agg(
    n_days=('count'),          # تعداد روزهای دارای داده
    mean_trips=('mean'),
    median_trips=('median'),
    min_trips=('min'),
    max_trips=('max'),
    std_trips=('std'),
)

trips_by_hour.index.name = 'hour_of_day'

print("=" * 60)
print("AVERAGE TRIP COUNTS BY HOUR OF DAY — NEW YORK LOCAL TIME")
print("=" * 60)
print("Note: Hour 02:00 is based on 90 days (instead of 91)")
print("      due to DST on March 10, 2024.")
print()

display(trips_by_hour.round(2))

# ---------------------------------------------------------
# Plot: average trips by hour of day
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 4))

trips_by_hour['mean_trips'].plot(
    kind='bar',
    ax=ax,
    color='steelblue',
    edgecolor='black',
    title='Average Yellow Taxi Trips by Hour of Day (New York Local Time)'
)

ax.set_xlabel('Hour of Day (0–23)')
ax.set_ylabel('Average Recorded Trips')
ax.tick_params(axis='x', rotation=0)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()

# Save figure
FIGURES_DIR = Path(FIGURES_DIR)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.savefig(
    FIGURES_DIR / "average_trips_by_hour_of_day.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print(f"✅ Figure saved: {FIGURES_DIR / 'average_trips_by_hour_of_day.png'}")
## تحلیل تعداد سفرها بر اساس روزهای هفته
# =========================================================
# Trip pattern by day of week (New York local time)
# =========================================================

from pathlib import Path
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Daily totals (using NY local days via resample)
# ---------------------------------------------------------
daily_counts = hourly_counts.resample('D').sum()

daily_counts_df = daily_counts.to_frame(name='daily_trips')
daily_counts_df['day_of_week'] = daily_counts_df.index.dayofweek

# ---------------------------------------------------------
# Weekly statistics
# ---------------------------------------------------------
weekday_stats = daily_counts_df.groupby('day_of_week')['daily_trips'].agg(
    n_days='size',            # مطمئن‌تر از count
    mean_trips='mean',
    median_trips='median',
    min_trips='min',
    max_trips='max',
    std_trips='std',
)

# Map day numbers to English names
weekday_names = {
    0: 'Monday',
    1: 'Tuesday',
    2: 'Wednesday',
    3: 'Thursday',
    4: 'Friday',
    5: 'Saturday',
    6: 'Sunday',
}

weekday_stats.index = weekday_stats.index.map(weekday_names)
weekday_stats.index.name = 'day_of_week'

# ---------------------------------------------------------
# Report
# ---------------------------------------------------------
print("=" * 60)
print("TRIP PATTERN BY DAY OF WEEK — NEW YORK LOCAL TIME")
print("=" * 60)
print("Note: Sunday, March 10 has only 23 hours due to DST.")
print("      This slightly lowers its daily total and the")
print("      Sunday average.")
print()
print("Expected n_days for every weekday: 13")
print("(91 days / 7 weekdays = 13)")
print()

display(weekday_stats.round(2))

# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4))

weekday_stats['mean_trips'].plot(
    kind='bar',
    ax=ax,
    color='steelblue',
    edgecolor='black',
    title='Average Daily Yellow Taxi Trips by Day of Week (New York Local Time)'
)

ax.set_xlabel('Day of Week')
ax.set_ylabel('Average Recorded Trips')
ax.tick_params(axis='x', rotation=30)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()

# Save figure
FIGURES_DIR = Path(FIGURES_DIR)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.savefig(
    FIGURES_DIR / "average_trips_by_day_of_week.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print(f"✅ Figure saved: {FIGURES_DIR / 'average_trips_by_day_of_week.png'}")
## تحلیل تغییرات سفرها در طول تاریخ
# =========================================================
# Daily trip counts over the research period (NY local time)
# =========================================================

from pathlib import Path
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Daily totals (resample on NY local index)
# ---------------------------------------------------------
daily_counts = hourly_counts.resample('D').sum()

# ---------------------------------------------------------
# Report
# ---------------------------------------------------------
print("=" * 60)
print("DAILY TRIP COUNTS — JANUARY TO MARCH 2024 (NY LOCAL TIME)")
print("=" * 60)

print("Number of calendar days:", len(daily_counts))
print("Days with zero recorded trips:", int((daily_counts == 0).sum()))
print("First date:", daily_counts.index.min())
print("Last date:", daily_counts.index.max())

print()
print("Note: Sunday, March 10 has only 23 hours due to DST.")
print("      Its daily total is therefore slightly lower than")
print("      a typical Sunday.")
print()

display(daily_counts.describe().to_frame('daily_trips'))

# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 4))

daily_counts.plot(
    ax=ax,
    color='steelblue',
    linewidth=1.2,
    title='Daily Yellow Taxi Trips: January–March 2024 (New York Local Time)'
)

ax.set_xlabel('Date (America/New_York)')
ax.set_ylabel('Recorded Trips')
ax.grid(alpha=0.3)

# Highlight the DST day
dst_day = pd.Timestamp("2024-03-10", tz=NY_TZ)
ax.axvline(
    dst_day,
    color='red',
    linestyle='--',
    alpha=0.6,
    label='DST day (2024-03-10)'
)
ax.legend()

plt.tight_layout()

# Save figure
FIGURES_DIR = Path(FIGURES_DIR)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.savefig(
    FIGURES_DIR / "daily_trip_counts.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print(f"✅ Figure saved: {FIGURES_DIR / 'daily_trip_counts.png'}")
### شناسایی روزهای کم‌تردد و پرتردد
# =========================================================
# Five lowest and five highest trip-volume days
# (New York local time)
# =========================================================

from pathlib import Path
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Build daily totals (only if not already built)
# ---------------------------------------------------------
if "daily_counts" not in globals():
    daily_counts = hourly_counts.resample("D").sum()

# ---------------------------------------------------------
# Helper: add weekday name and mark DST day
# ---------------------------------------------------------
dst_day = pd.Timestamp("2024-03-10", tz=NY_TZ)

def _enrich(series, label):
    df = series.to_frame("daily_trips")
    df["day_of_week"] = df.index.day_name()
    df["is_dst_day"] = df.index == dst_day
    df = df.rename_axis("date")
    df["note"] = df["is_dst_day"].map(
        {True: "DST day (23h)", False: ""}
    )
    print(f"\n{label}:")
    display(df[["daily_trips", "day_of_week", "note"]])

# ---------------------------------------------------------
# Report
# ---------------------------------------------------------
print("=" * 60)
print("FIVE LOWEST AND FIVE HIGHEST TRIP-VOLUME DAYS")
print("=" * 60)
print("Note: Sunday, March 10 (DST day) has only 23 hours,")
print("      so its total is slightly lower than a normal Sunday.")

_enrich(daily_counts.nsmallest(5), "Five days with the FEWEST recorded trips")
_enrich(daily_counts.nlargest(5), "Five days with the MOST recorded trips")

# ---------------------------------------------------------
# Optional: comparison plot
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 4))

top_bottom = pd.concat([
    daily_counts.nsmallest(5),
    daily_counts.nlargest(5),
])

colors = ["red"] * 5 + ["green"] * 5

top_bottom.plot(
    kind="bar",
    ax=ax,
    color=colors,
    edgecolor="black",
    title="Five Lowest (red) and Five Highest (green) Trip-Volume Days"
)

ax.set_xlabel("Date")
ax.set_ylabel("Daily Recorded Trips")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()

# Save figure
FIGURES_DIR = Path(FIGURES_DIR)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.savefig(
    FIGURES_DIR / "top_bottom_days.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print(f"✅ Figure saved: {FIGURES_DIR / 'top_bottom_days.png'}")
## شناسایی و تحلیل دوره‌های اوج ترافیک و ناهنجاری‌های زمانی
# =========================================================
# Cell — Peak Periods & Temporal Anomalies
# (تحلیل دوره‌های پرترافیک و روزهای غیرمعمول)
# =========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("PEAK PERIODS & TEMPORAL ANOMALIES ANALYSIS")
print("=" * 70)

# ۱. محاسبه آستانه‌های آماری
daily_mean = daily_counts.mean()
daily_std = daily_counts.std()
p95 = daily_counts.quantile(0.95)
p05 = daily_counts.quantile(0.05)

print(f"Daily Mean: {daily_mean:,.0f} trips")
print(f"P95 (Peak Threshold): {p95:,.0f} trips")
print(f"P05 (Low Threshold): {p05:,.0f} trips")

# ۲. شناسایی روزهای پرترافیک (بالاتر از P95)
peak_days = daily_counts[daily_counts >= p95].sort_values(ascending=False)
print(f"\n[1] Top Peak Days (>= P95):")
for date, count in peak_days.items():
    print(f"  - {date.date()} ({date.day_name()}): {count:,} trips")

# ۳. شناسایی روزهای کم‌ترافیک (پایین‌تر از P05)
low_days = daily_counts[daily_counts <= p05].sort_values()
print(f"\n[2] Lowest Traffic Days (<= P05):")
for date, count in low_days.items():
    print(f"  - {date.date()} ({date.day_name()}): {count:,} trips")

# ۴. شناسایی الگوهای غیرمعمول با Z-Score
# (هر روزی که بیش از ۲.۵ انحراف معیار از میانگین فاصله داشته باشد)
z_scores = (daily_counts - daily_mean) / daily_std
anomalies = z_scores[abs(z_scores) >= 2.5]

print(f"\n[3] Temporal Anomalies (|Z| >= 2.5):")
if len(anomalies) > 0:
    for date, z in anomalies.items():
        direction = "HIGH" if z > 0 else "LOW"
        print(f"  - {date.date()} ({date.day_name()}): {daily_counts[date]:,} trips (Z-score: {z:.2f} - Unusually {direction})")
else:
    print("  - No extreme anomalies detected (none exceeded Z-score of 2.5).")

# ۵. رسم نمودار روزانه با هایلایت کردن روزهای پرترافیک
plt.figure(figsize=(15, 5))
plt.plot(daily_counts.index, daily_counts.values, color='steelblue', label='Daily Trips')
plt.axhline(p95, color='green', linestyle='--', label=f'P95 Peak Threshold ({p95:,.0f})')
plt.axhline(p05, color='red', linestyle='--', label=f'P05 Low Threshold ({p05:,.0f})')

# علامت‌گذاری رخدادهای غیرمعمول
for date in anomalies.index:
    color = 'darkgreen' if anomalies[date] > 0 else 'darkred'
    plt.scatter(date, daily_counts[date], color=color, s=100, zorder=5)

plt.title("Daily Taxi Trips: Peak Periods and Anomalies (Q1 2024)")
plt.xlabel("Date")
plt.ylabel("Daily Recorded Trips")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "temporal_anomalies.png", dpi=200, bbox_inches="tight")
plt.show()

print(f"\n✅ نمودار رخدادهای زمانی در مسیر {FIGURES_DIR} ذخیره شد.")
## ممیزی مدت سفر، مسافت و کرایه

import pandas as pd
import pyarrow.parquet as pq

audit_columns = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "trip_distance",
    "fare_amount",
    "total_amount",
]

for month, path in local_files.items():
    df_audit = pd.read_parquet(path, columns=audit_columns)

    duration_min = (
        df_audit["tpep_dropoff_datetime"]
        - df_audit["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    checks = {
        "Total records": len(df_audit),
        "Missing pickup time": df_audit["tpep_pickup_datetime"].isna().sum(),
        "Missing dropoff time": df_audit["tpep_dropoff_datetime"].isna().sum(),
        "Duration <= 0 min": (duration_min <= 0).sum(),
        "Duration > 24 hours": (duration_min > 1440).sum(),
        "Missing trip distance": df_audit["trip_distance"].isna().sum(),
        "Distance <= 0": (df_audit["trip_distance"] <= 0).sum(),
        "Missing fare": df_audit["fare_amount"].isna().sum(),
        "Fare < 0": (df_audit["fare_amount"] < 0).sum(),
        "Missing total amount": df_audit["total_amount"].isna().sum(),
        "Total amount < 0": (df_audit["total_amount"] < 0).sum(),
    }

    print(f"\n--- {month} ---")
    for label, count in checks.items():
        print(f"{label}: {count}")

    del df_audit, duration_min


# ============================================================
# بررسی تکمیلی: تفکیک مدت سفر صفر از مدت سفر منفی
# ============================================================

duration_detail_rows = []

for month, path in local_files.items():

    df_time = pd.read_parquet(
        path,
        columns=[
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime"
        ]
    )

    # محاسبه مدت سفر بر حسب دقیقه
    duration_min = (
        df_time["tpep_dropoff_datetime"]
        - df_time["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    total_records = len(df_time)

    zero_duration = int((duration_min == 0).sum())
    negative_duration = int((duration_min < 0).sum())
    positive_duration = int((duration_min > 0).sum())
    missing_duration = int(duration_min.isna().sum())

    duration_detail_rows.append({
        "month": month,
        "total_records": total_records,
        "zero_duration_count": zero_duration,
        "negative_duration_count": negative_duration,
        "positive_duration_count": positive_duration,
        "missing_duration_count": missing_duration,
        "zero_duration_percent": round(
            zero_duration / total_records * 100, 4
        ),
        "negative_duration_percent": round(
            negative_duration / total_records * 100, 4
        )
    })

duration_detail_report = pd.DataFrame(duration_detail_rows)

display(duration_detail_report)
## بررسی سرعت تقریبی سفرها
import pandas as pd

speed_columns = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "trip_distance",
]

for month, path in local_files.items():
    df_speed = pd.read_parquet(path, columns=speed_columns)

    duration_hours = (
        df_speed["tpep_dropoff_datetime"]
        - df_speed["tpep_pickup_datetime"]
    ).dt.total_seconds() / 3600

    valid = (
        duration_hours.gt(0)
        & df_speed["trip_distance"].gt(0)
    )

    speed_mph = df_speed.loc[valid, "trip_distance"] / duration_hours[valid]

    print(f"\n--- {month} ---")
    print("Trips with valid positive duration and distance:", int(valid.sum()))
    print("Median speed (mph):", round(speed_mph.median(), 2))
    print("95th percentile speed (mph):", round(speed_mph.quantile(0.95), 2))
    print("Speeds above 80 mph:", int((speed_mph > 80).sum()))
    print("Maximum calculated speed (mph):", round(speed_mph.max(), 2))

    del df_speed, duration_hours, speed_mph
### پیدا کردن علت سرعت‌های غیرعادی
import pandas as pd

speed_check_columns = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "trip_distance",
]

for month, path in local_files.items():
    df_check = pd.read_parquet(
        path,
        columns=speed_check_columns
    )

    df_check["duration_seconds"] = (
        df_check["tpep_dropoff_datetime"]
        - df_check["tpep_pickup_datetime"]
    ).dt.total_seconds()

    valid = (
        (df_check["duration_seconds"] > 0)
        & (df_check["trip_distance"] > 0)
    )

    df_check = df_check.loc[valid].copy()

    df_check["duration_minutes"] = (
        df_check["duration_seconds"] / 60
    )

    df_check["speed_mph"] = (
        df_check["trip_distance"]
        / (df_check["duration_seconds"] / 3600)
    )

    print(f"\n--- {month}: 5 highest speeds ---")

    display(
        df_check.nlargest(5, "speed_mph")[
            [
                "tpep_pickup_datetime",
                "tpep_dropoff_datetime",
                "trip_distance",
                "duration_seconds",
                "duration_minutes",
                "speed_mph",
            ]
        ]
    )

    del df_check
### ساخت جدول تفکیکی ناهنجاری‌ها

import pandas as pd
import numpy as np

try:
    from IPython.display import display
except ImportError:
    display = print


# ============================================================
# MONTHLY ANOMALY AUDIT
# ============================================================

audit_columns = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "trip_distance",
    "fare_amount",
    "total_amount"
]

anomaly_results = []

for month, path in local_files.items():

    print(f"Processing monthly file: {month} ...")

    df = pd.read_parquet(path, columns=audit_columns)

    # Calculate trip duration
    duration_min = (
        df["tpep_dropoff_datetime"]
        - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    duration_hours = duration_min / 60

    # Identify valid trips for speed calculation
    valid_speed = (
        duration_hours.gt(0)
        & duration_hours.notna()
        & df["trip_distance"].gt(0)
        & df["trip_distance"].notna()
    )

    # Calculate speed
    speed_mph = pd.Series(float("nan"), index=df.index)

    speed_mph.loc[valid_speed] = (
        df.loc[valid_speed, "trip_distance"]
        / duration_hours.loc[valid_speed]
    )

    # Define anomaly checks — ENGLISH LABELS
    checks = {
        "Zero or Negative Trip Duration":
            duration_min.le(0),

        "Trip Duration Exceeding 24 Hours":
            duration_min.gt(1440),

        "Zero or Negative Trip Distance":
            df["trip_distance"].le(0),

        "Negative Fare Amount":
            df["fare_amount"].lt(0),

        "Negative Total Amount":
            df["total_amount"].lt(0),

        "Speed Exceeding 80 MPH":
            valid_speed & speed_mph.gt(80),

        "Trip Distance Exceeding 100 Miles":
            df["trip_distance"].gt(100)
    }

    total_records = len(df)

    # Calculate anomaly counts and percentages
    for anomaly_name, mask in checks.items():

        count = int(mask.fillna(False).sum())

        anomaly_results.append({
            "ماه": month,
            "نوع بررسی": anomaly_name,
            "تعداد رکورد": count,
            "درصد از کل رکوردهای ماه": round(
                count / total_records * 100, 4
            ) if total_records > 0 else 0
        })

    print(
        f"Completed: {month} | "
        f"Total Records: {total_records:,}"
    )

    del df, duration_min, duration_hours
    del speed_mph, valid_speed


# ============================================================
# BUILD ANOMALY SUMMARY
# ============================================================

anomaly_summary = pd.DataFrame(anomaly_results)

print("\n" + "=" * 75)
print("MONTHLY ANOMALY AUDIT COMPLETED")
print("=" * 75)


# ============================================================
# TABLE 1 — ANOMALY COUNTS
# ============================================================

print("\nTABLE 1 — ANOMALY COUNTS BY MONTH")

count_table = (
    anomaly_summary.pivot(
        index="نوع بررسی",
        columns="ماه",
        values="تعداد رکورد"
    )
    .rename_axis(index="Anomaly Type", columns="Month")
)

display(
    count_table.style
    .format("{:,.0f}")
    .set_properties(**{"text-align": "center"})
)


# ============================================================
# TABLE 2 — ANOMALY PERCENTAGES
# ============================================================

print("\nTABLE 2 — ANOMALY PERCENTAGES BY MONTH")

percentage_table = (
    anomaly_summary.pivot(
        index="نوع بررسی",
        columns="ماه",
        values="درصد از کل رکوردهای ماه"
    )
    .rename_axis(index="Anomaly Type", columns="Month")
)

display(
    percentage_table.style
    .format("{:.4f}%")
    .set_properties(**{"text-align": "center"})
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\nAUDIT SUMMARY")
print(f"Monthly Files Analyzed: {len(local_files)}")
print(f"Anomaly Categories Examined: {len(checks)}")
print("All anomaly summary tables were generated successfully.")

##  جدول جامع ممیزی

import pandas as pd

try:
    from IPython.display import display
except ImportError:
    display = print


# =========================================================
# STEP 1 — Total Records per Month
# =========================================================

monthly_totals = (
    timestamp_audit[["month_file", "rows_read_raw"]]
    .rename(columns={
        "month_file": "Month",
        "rows_read_raw": "Total Records"
    })
    .copy()
)


# =========================================================
# STEP 2 — Standardize Anomaly Summary Column Names
# =========================================================

anomaly_data = anomaly_summary.rename(columns={
    "ماه": "Month",
    "نوع بررسی": "Anomaly Type",
    "تعداد رکورد": "Record Count",
    "درصد از کل رکوردهای ماه": "Percentage of Monthly Records"
}).copy()

required_anomaly_columns = [
    "Month",
    "Anomaly Type",
    "Record Count",
    "Percentage of Monthly Records"
]

missing_anomaly_columns = [
    col for col in required_anomaly_columns
    if col not in anomaly_data.columns
]

if missing_anomaly_columns:
    raise ValueError(
        "Missing required columns in anomaly_summary: "
        + str(missing_anomaly_columns)
    )


# =========================================================
# STEP 3 — Monthly Anomaly Counts
# =========================================================

anomaly_counts = anomaly_data.pivot(
    index="Month",
    columns="Anomaly Type",
    values="Record Count"
)

anomaly_counts.columns = [
    f"Count | {column}"
    for column in anomaly_counts.columns
]

anomaly_counts = anomaly_counts.reset_index()


# =========================================================
# STEP 4 — Monthly Anomaly Percentages
# =========================================================

anomaly_percentages = anomaly_data.pivot(
    index="Month",
    columns="Anomaly Type",
    values="Percentage of Monthly Records"
)

anomaly_percentages.columns = [
    f"Percentage (%) | {column}"
    for column in anomaly_percentages.columns
]

anomaly_percentages = anomaly_percentages.reset_index()


# =========================================================
# STEP 5 — Combine Counts and Percentages
# =========================================================

anomaly_table = anomaly_counts.merge(
    anomaly_percentages,
    on="Month",
    how="outer",
    validate="one_to_one"
)


# =========================================================
# STEP 6 — Build Monthly Anomaly Audit Table
# =========================================================

anomaly_display = (
    monthly_totals
    .merge(
        anomaly_table,
        on="Month",
        how="left",
        validate="one_to_one"
    )
    .sort_values("Month")
    .reset_index(drop=True)
)


# =========================================================
# STEP 7 — Prepare Correlation Results
# =========================================================

relationship_columns = [
    "Month",
    "Scenario",
    "Records Used for Correlation",
    "Spearman: Distance-Duration",
    "Spearman: Distance-Fare",
    "Spearman: Duration-Fare",
    "Spearman: Distance-Speed",
    "Spearman: Duration-Speed"
]

if "relationship_summary" in globals():

    relationship_table = relationship_summary.rename(columns={
        "month": "Month",
        "scenario": "Scenario",
        "records": "Records Used for Correlation",
        "distance_vs_duration": "Spearman: Distance-Duration",
        "distance_vs_fare": "Spearman: Distance-Fare",
        "duration_vs_fare": "Spearman: Duration-Fare",
        "distance_vs_speed": "Spearman: Distance-Speed",
        "duration_vs_speed": "Spearman: Duration-Speed"
    }).copy()

    missing_columns = [
        col for col in relationship_columns
        if col not in relationship_table.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns in relationship_summary: "
            + str(missing_columns)
        )

    relationship_display = (
        relationship_table[relationship_columns]
        .sort_values(["Month", "Scenario"])
        .reset_index(drop=True)
    )

    correlation_columns = [
        col for col in relationship_display.columns
        if col.startswith("Spearman:")
    ]

    relationship_display[correlation_columns] = (
        relationship_display[correlation_columns].round(3)
    )

else:

    relationship_display = pd.DataFrame(
        columns=relationship_columns
    )

    print(
        "WARNING: relationship_summary was not found.\n"
        "Correlation results have not been calculated or loaded.\n"
        "Please run the correlation calculation cell first."
    )


# =========================================================
# STEP 8 — Display Table 1: Monthly Anomaly Audit
# =========================================================

print("\n" + "=" * 80)
print("TABLE 1 — MONTHLY ANOMALY AUDIT")
print("=" * 80)

display(
    anomaly_display.style
    .format(precision=4)
    .set_properties(**{
        "text-align": "center"
    })
)


# =========================================================
# STEP 9 — Display Table 2: Variable Relationships
# =========================================================

print("\n" + "=" * 80)
print("TABLE 2 — VARIABLE RELATIONSHIPS AND CORRELATION COVERAGE")
print("=" * 80)

if not relationship_display.empty:

    display(
        relationship_display.style
        .format(precision=3)
        .set_properties(**{
            "text-align": "center"
        })
    )

else:

    print(
        "Table 2 is currently unavailable because "
        "the correlation summary has not been created."
    )

print("\nAudit tables generated successfully.")

### ارزیابی کیفیت داده‌های مکانی 

# =========================================================
# Cell — Spatial Quality Audit & Initial Profile
# پوشش تمام ۵ مورد ممیزی کیفیت + نمای اولیه مکانی (مسیر A)
# =========================================================
import pandas as pd
import pyarrow.parquet as pq

print("=" * 70)
print("SPATIAL QUALITY AUDIT & INITIAL PROFILE (Q1 2024)")
print("=" * 70)

spatial_frames = []
for month, path in local_files.items():
    # خواندن داده‌ها
    table = pq.read_table(path, columns=["tpep_pickup_datetime", "PULocationID", "DOLocationID"])
    df_loc = table.to_pandas()
    
    # اعمال فیلتر بازه معتبر
    inside_mask = df_loc["tpep_pickup_datetime"].ge(START) & df_loc["tpep_pickup_datetime"].lt(END_EXCLUSIVE)
    spatial_frames.append(df_loc.loc[inside_mask])

df_all_spatial = pd.concat(spatial_frames, ignore_index=True)
total_trips = len(df_all_spatial)

print(f"Total valid trips analyzed: {total_trips:,}")

# ----------------------------------------------------------------------
# ۱. مقادیر گمشده (Missingness)
# ----------------------------------------------------------------------
missing_pu = df_all_spatial["PULocationID"].isna().sum()
missing_do = df_all_spatial["DOLocationID"].isna().sum()
print(f"\n[1] Missingness")
print(f"  Missing Pickup Zones:  {missing_pu:,} records ({missing_pu/total_trips*100:.4f}%)")
print(f"  Missing Dropoff Zones: {missing_do:,} records ({missing_do/total_trips*100:.4f}%)")

# ----------------------------------------------------------------------
# ۲. مقادیر ناممکن (Impossible Values)
# ----------------------------------------------------------------------
# در دیتابیس TLC، کدهای 264 و 265 به معنای منطقه ناشناخته هستند.
unknown_pu = df_all_spatial[df_all_spatial["PULocationID"].isin([264, 265])].shape[0]
unknown_do = df_all_spatial[df_all_spatial["DOLocationID"].isin([264, 265])].shape[0]
print(f"\n[2] Impossible / Unknown Zones (IDs 264, 265)")
print(f"  Unknown Pickup Zones:  {unknown_pu:,} records ({unknown_pu/total_trips*100:.4f}%)")
print(f"  Unknown Dropoff Zones: {unknown_do:,} records ({unknown_do/total_trips*100:.4f}%)")

# ----------------------------------------------------------------------
# ۳. ناسازگاری نوع داده (Type Consistency)
# ----------------------------------------------------------------------
# بررسی اینکه آیا ستون‌ها واقعاً عدد صحیح هستند
is_pu_int = pd.api.types.is_integer_dtype(df_all_spatial["PULocationID"])
is_do_int = pd.api.types.is_integer_dtype(df_all_spatial["DOLocationID"])
print(f"\n[3] Type Consistency")
print(f"  PULocationID is integer type: {is_pu_int}")
print(f"  DOLocationID is integer type: {is_do_int}")

# ----------------------------------------------------------------------
# ۴. رکوردهای تکراری (Duplicates)
# ----------------------------------------------------------------------
# چون ما سفرها را تجمیع کرده‌ایم، اگر یک رکورد در فایل اصلی تکراری بوده، اینجا هم هست.
# بررسی تکرار بر اساس ترکیب زمان و مکان
duplicate_count = df_all_spatial.duplicated(subset=["tpep_pickup_datetime", "PULocationID", "DOLocationID"]).sum()
print(f"\n[4] Exact Duplicate Trips (same time & zones)")
print(f"  Duplicate records: {duplicate_count:,} ({duplicate_count/total_trips*100:.4f}%)")

# ----------------------------------------------------------------------
# ۵. نمونه‌های پرت و عدم‌توازن (Outliers & Imbalance)
# ----------------------------------------------------------------------
top_pu = df_all_spatial["PULocationID"].value_counts().head(10)
top_do = df_all_spatial["DOLocationID"].value_counts().head(10)

print("\n[5] Top 10 Pickup Zones (Imbalance Analysis)")
for zone, count in top_pu.items():
    print(f"  Zone {zone}: {count:,} trips ({count/total_trips*100:.2f}%)")

print("\n    Top 10 Dropoff Zones")
for zone, count in top_do.items():
    print(f"  Zone {zone}: {count:,} trips ({count/total_trips*100:.2f}%)")

# درصد تمرکز (هرچه این عدد بالاتر باشد، یعنی تقاضا در چند منطقه متمرکزتر است)
top_5_pu_share = top_pu.head(5).sum() / total_trips * 100
print(f"\n  Concentration Check: Top 5 pickup zones account for {top_5_pu_share:.2f}% of all trips.")
##  تحلیل اکتشافی تک‌متغیره (Univariate EDA)
<div dir="rtl" style="text-align: right;">

در این بخش، توزیع، مرکز، پراکندگی، عدم‌توازن و شکل توزیع متغیرهای اصلی پروژه بررسی می‌شود.
متغیرهای اصلی در سطح رکورد سفر عبارت‌اند از: `trip_distance`, `fare_amount`, 
`duration_minutes`, `speed_mph`, `passenger_count`. متغیر هدف در سطح ساعت
(`trip_count`) نیز جداگانه تحلیل می‌شود.

</div>
# =========================================================
# Cell — Univariate exploratory analysis (Phase 1)
# توزیع، مرکز، پراکندگی، عدم‌توازن و شکل توزیع متغیرهای اصلی
# =========================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

print("=" * 70)
print("UNIVARIATE ANALYSIS — PART 1: TRIP-LEVEL VARIABLES")
print("=" * 70)

# ---------- ۱. جدول آمارهای خلاصه در سطح رکورد سفر ----------
TRIP_LEVEL_COLS = [
    "trip_distance",
    "fare_amount",
    "duration_minutes",   # مشتق‌شده
    "speed_mph",          # مشتق‌شده
    "passenger_count",
]

rows_summary = []

for month, path in local_files.items():
    tbl = pq.read_table(
        path,
        columns=[
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "trip_distance",
            "fare_amount",
            "passenger_count",
        ],
    )
    df_m = tbl.to_pandas()

    # ساخت متغیرهای مشتق‌شده
    dur_min = (
        df_m["tpep_dropoff_datetime"] - df_m["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    df_m["duration_minutes"] = dur_min
    df_m["speed_mph"] = np.where(
        dur_min > 0,
        df_m["trip_distance"] / (dur_min / 60),
        np.nan,
    )

    for col in TRIP_LEVEL_COLS:
        s = df_m[col].dropna()

        if len(s) == 0:
            continue

        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        rows_summary.append({
            "month": month,
            "variable": col,
            "n": len(s),
            # ---- مرکز ----
            "mean": s.mean(),
            "median": s.median(),
            # ---- پراکندگی ----
            "std": s.std(),
            "iqr": q3 - q1,
            "min": s.min(),
            "p01": s.quantile(0.01),
            "p25": q1,
            "p75": q3,
            "p99": s.quantile(0.99),
            "max": s.max(),
            # ---- عدم‌توازن ----
            "skewness": s.skew(),
            "kurtosis": s.kurt(),
            "mean_over_median": s.mean() / s.median() if s.median() != 0 else np.nan,
            "pct_zero": (s == 0).mean() * 100,
            "pct_negative": (s < 0).mean() * 100,
        })

univariate_trip_level = pd.DataFrame(rows_summary)

print("\n--- آمارهای خلاصه (سطح رکورد سفر، به تفکیک ماه) ---")
display(
    univariate_trip_level.round(3).style
    .format(precision=3)
    .set_properties(**{"text-align": "center"})
)
# =========================================================
# Cell — Univariate plots: histogram + boxplot
# توزیع و شکل توزیع متغیرها
# =========================================================
print("=" * 70)
print("UNIVARIATE ANALYSIS — PART 2: DISTRIBUTION SHAPES")
print("=" * 70)

# ---------- نمونه‌گیری برای نمایش سریع‌تر ----------
SAMPLE_PER_MONTH = 100_000
sample_frames = []

for month, path in local_files.items():
    tbl = pq.read_table(
        path,
        columns=[
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "trip_distance",
            "fare_amount",
            "passenger_count",
        ],
    )
    df_m = tbl.to_pandas()

    dur_min = (
        df_m["tpep_dropoff_datetime"] - df_m["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    df_m["duration_minutes"] = dur_min
    df_m["speed_mph"] = np.where(
        dur_min > 0,
        df_m["trip_distance"] / (dur_min / 60),
        np.nan,
    )

    if len(df_m) > SAMPLE_PER_MONTH:
        df_m = df_m.sample(SAMPLE_PER_MONTH, random_state=42)

    sample_frames.append(df_m[TRIP_LEVEL_COLS])

sample_df = pd.concat(sample_frames, ignore_index=True)

# ---------- رسم هیستوگرام + Boxplot برای هر متغیر ----------
fig, axes = plt.subplots(
    nrows=len(TRIP_LEVEL_COLS),
    ncols=2,
    figsize=(14, 3.2 * len(TRIP_LEVEL_COLS)),
)

for i, col in enumerate(TRIP_LEVEL_COLS):
    s = sample_df[col].dropna()

    # برای متغیرهای با دُم بسیار بلند، محدوده نمایش را منطقی می‌کنیم
    # تا شکل توزیع دیده شود
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    s_plot = s[(s >= lo) & (s <= hi)]

    # هیستوگرام با y لگاریتمی
    axes[i, 0].hist(s_plot, bins=80, color="steelblue", edgecolor="black", alpha=0.85)
    axes[i, 0].set_yscale("log")
    axes[i, 0].set_title(f"Histogram (log-y) — {col}\n(1%-99% range)")
    axes[i, 0].set_xlabel(col)
    axes[i, 0].set_ylabel("frequency (log)")
    axes[i, 0].grid(alpha=0.3)

    # Boxplot (با نمایش پرت‌ها)
    axes[i, 1].boxplot(s, vert=True, showfliers=True,
                       flierprops=dict(marker='.', markersize=1, alpha=0.3))
    axes[i, 1].set_title(f"Boxplot — {col}\n(all values, outliers shown)")
    axes[i, 1].set_ylabel(col)
    axes[i, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "univariate_trip_level.png", dpi=200, bbox_inches="tight")
plt.show()
print(f"Saved: {FIGURES_DIR / 'univariate_trip_level.png'}")
# =========================================================
# Cell — Univariate analysis of the TARGET variable (hourly trip_count)
# =========================================================
print("=" * 70)
print("UNIVARIATE ANALYSIS — PART 3: TARGET VARIABLE (trip_count)")
print("=" * 70)

# hourly_counts در سلول‌های قبلی ساخته شده است
s = hourly_counts.dropna()

q1, q3 = s.quantile(0.25), s.quantile(0.75)

target_summary = pd.DataFrame([{
    "variable": "trip_count (hourly)",
    "n_hours": len(s),
    # مرکز
    "mean": s.mean(),
    "median": s.median(),
    # پراکندگی
    "std": s.std(),
    "iqr": q3 - q1,
    "min": s.min(),
    "p01": s.quantile(0.01),
    "p25": q1,
    "p75": q3,
    "p99": s.quantile(0.99),
    "max": s.max(),
    # عدم‌توازن
    "skewness": s.skew(),
    "kurtosis": s.kurt(),
    "mean_over_median": s.mean() / s.median() if s.median() != 0 else np.nan,
    "pct_zero": (s == 0).mean() * 100,
    "pct_negative": (s < 0).mean() * 100,
}])

print("\n--- آمارهای خلاصه متغیر هدف (trip_count ساعتی) ---")
display(target_summary.round(3).style.format(precision=3))

# ---------- نمودار ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# هیستوگرام
axes[0].hist(s, bins=60, color="darkorange", edgecolor="black", alpha=0.85)
axes[0].set_title("Histogram — hourly trip_count")
axes[0].set_xlabel("trip_count per hour")
axes[0].set_ylabel("frequency")
axes[0].axvline(s.mean(), color="red", linestyle="--", label=f"mean = {s.mean():,.0f}")
axes[0].axvline(s.median(), color="blue", linestyle="--", label=f"median = {s.median():,.0f}")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Boxplot
axes[1].boxplot(s, vert=True)
axes[1].set_title("Boxplot — hourly trip_count")
axes[1].set_ylabel("trip_count per hour")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "univariate_target_hourly.png", dpi=200, bbox_inches="tight")
plt.show()
print(f"Saved: {FIGURES_DIR / 'univariate_target_hourly.png'}")
## تحلیل اکتشافی تک متغیره  مکان 
# =========================================================
# Cell — Spatial Univariate Analysis (تحلیل تک‌متغیره مکانی)
# پوشش الزامات: توزیع، مرکز، پراکندگی، عدم‌توازن و شکل توزیع
# =========================================================
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- تنظیمات اولیه (در صورت اجرای مستقل) ---
# FIGURES_DIR = Path("../figures/phase1")  # اگر قبلاً تعریف نشده است
# FIGURES_DIR.mkdir(parents=True, exist_ok=True)
# total_trips = len(df_all_spatial) # اگر قبلاً تعریف نشده است

print("=" * 70)
print("SPATIAL UNIVARIATE ANALYSIS (Subgroup: Pickup & Dropoff Zones)")
print("=" * 70)

# ۱. توزیع (Distribution) - با استفاده از value_counts
pu_counts = df_all_spatial["PULocationID"].value_counts()
do_counts = df_all_spatial["DOLocationID"].value_counts()

# ۲. مرکز (Center) - مُد (Mode)
mode_pu = pu_counts.index[0]
mode_pu_count = pu_counts.iloc[0]
mode_do = do_counts.index[0]
mode_do_count = do_counts.iloc[0]

print(f"\n[2] Center (Mode):")
print(f"  - Most frequent Pickup Zone (Mode): {mode_pu} with {mode_pu_count:,} trips")
print(f"  - Most frequent Dropoff Zone (Mode): {mode_do} with {mode_do_count:,} trips")

# ۳. پراکندگی (Dispersion)
num_unique_pu = df_all_spatial["PULocationID"].nunique()
num_unique_do = df_all_spatial["DOLocationID"].nunique()
print(f"\n[3] Dispersion (Number of unique zones):")
print(f"  - Trips originated from {num_unique_pu} distinct zones.")
print(f"  - Trips ended in {num_unique_do} distinct zones.")

# ۴. عدم‌توازن (Imbalance)
top_5_pu_share = pu_counts.head(5).sum() / total_trips * 100
top_5_do_share = do_counts.head(5).sum() / total_trips * 100
print(f"\n[4] Imbalance (Concentration Check):")
print(f"  - Top 5 Pickup Zones account for {top_5_pu_share:.2f}% of all trips.")
print(f"  - Top 5 Dropoff Zones account for {top_5_do_share:.2f}% of all trips.")

# ۵. شکل توزیع (Shape of Distribution)
# برای متغیرهای دسته‌ای، شکل توزیع را با درصد تجمعی ۱۰ و ۵۰ منطقه اول می‌سنجیم.
top_10_pu_share = pu_counts.head(10).sum() / total_trips * 100
top_50_pu_share = pu_counts.head(50).sum() / total_trips * 100
print(f"\n[5] Shape of Distribution (Long-tail / Skewness):")
print(f"  - Top 10 Pickup Zones cover {top_10_pu_share:.2f}% of all trips.")
print(f"  - Top 50 Pickup Zones cover {top_50_pu_share:.2f}% of all trips.")
print(f"  - The distribution is heavily right-skewed (Long-tail) towards a few central/airport zones.")

# -----------------------------------------
# رسم نمودارها (نمایش بصری توزیع و عدم‌توازن)
# -----------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# نمودار مبدأها (Top 10)
top_pu = pu_counts.head(10)
axes[0].barh(top_pu.index.astype(str), top_pu.values, color="steelblue")
axes[0].set_title("Top 10 Pickup Zones (Distribution)")
axes[0].set_xlabel("Number of Trips")
axes[0].set_ylabel("Zone ID")
axes[0].invert_yaxis()
axes[0].grid(axis="x", alpha=0.3)

# نمودار مقصدها (Top 10)
top_do = do_counts.head(10)
axes[1].barh(top_do.index.astype(str), top_do.values, color="darkorange")
axes[1].set_title("Top 10 Dropoff Zones (Distribution)")
axes[1].set_xlabel("Number of Trips")
axes[1].set_ylabel("Zone ID")
axes[1].invert_yaxis()
axes[1].grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "spatial_top10_zones.png", dpi=200, bbox_inches="tight")
plt.show()

# -----------------------------------------
# نمودار تکمیلی: نمایش Long-tail (شکل توزیع)
# -----------------------------------------
plt.figure(figsize=(10, 5))
# نمایش ۱۰۰ منطقه اول برای دیدن شیب تند (Long-tail)
top_100_pu = pu_counts.head(100)
plt.plot(range(1, len(top_100_pu) + 1), top_100_pu.values, marker='o', markersize=3, color='teal', linestyle='-')
plt.title("Pickup Zone Distribution (Top 100) - Long-tail Shape")
plt.xlabel("Zone Rank (1 = Most Frequent)")
plt.ylabel("Number of Trips")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "spatial_distribution_longtail.png", dpi=200, bbox_inches="tight")
plt.show()
## تحلیل اکتشافی چندمتغیره 
### محاسبهٔ رابطهٔ میان متغیرها
import pandas as pd
import numpy as np

relationship_columns = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "trip_distance",
    "fare_amount",
    "passenger_count"  # <--- اضافه شد
]

relationship_results = []

for month, path in local_files.items():
    df_rel = pd.read_parquet(
        path,
        columns=relationship_columns
    )

    # محاسبه مدت سفر به دقیقه
    df_rel["duration_minutes"] = (
        df_rel["tpep_dropoff_datetime"]
        - df_rel["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    # محاسبه سرعت تقریبی بر حسب مایل بر ساعت
    df_rel["speed_mph"] = (
        df_rel["trip_distance"]
        / (df_rel["duration_minutes"] / 60)
    )

    # شرایط اولیه برای استفاده در محاسبه همبستگی
    valid = (
        df_rel["duration_minutes"].gt(0)
        & df_rel["trip_distance"].gt(0)
        & df_rel["fare_amount"].ge(0)
        & np.isfinite(df_rel["speed_mph"])
    )

    analysis_df = df_rel.loc[
        valid,
        [
            "duration_minutes",
            "trip_distance",
            "fare_amount",
            "speed_mph",
            "passenger_count"  # <--- اضافه شد
        ]
    ].copy()

    # دو مجموعه برای مقایسه:
    # ۱. تمام رکوردهای معتبر طبق شرایط بالا
    # ۲. رکوردهای معتبر با سرعت حداکثر ۸۰ مایل بر ساعت
    datasets = {
        "all_valid": analysis_df,
        "speed_le_80": analysis_df.loc[
            analysis_df["speed_mph"].le(80)
        ],
    }

    for scenario, data in datasets.items():
        corr = data.corr(method="spearman")

        relationship_results.append({
            "month": month,
            "scenario": scenario,
            "records": len(data),
            "distance_vs_duration": corr.loc["trip_distance", "duration_minutes"],
            "distance_vs_fare": corr.loc["trip_distance", "fare_amount"],
            "duration_vs_fare": corr.loc["duration_minutes", "fare_amount"],
            "distance_vs_speed": corr.loc["trip_distance", "speed_mph"],
            "duration_vs_speed": corr.loc["duration_minutes", "speed_mph"],
            # روابط جدید مربوط به تعداد مسافر:
            "distance_vs_passenger": corr.loc["trip_distance", "passenger_count"],
            "fare_vs_passenger": corr.loc["fare_amount", "passenger_count"],
            "duration_vs_passenger": corr.loc["duration_minutes", "passenger_count"],
        })

    del df_rel, analysis_df

relationship_summary = pd.DataFrame(relationship_results)

display(relationship_summary.round(3))
# =========================================================
# Cell — Multivariate EDA & Subgroup Differences (گام ۳ - اصلاح‌شده)
# تحلیل اکتشافی چندمتغیره و تفاوت زیرگروه‌ها
# =========================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

print("=" * 70)
print("MULTIVARIATE ANALYSIS — Subgroups and Relationships")
print("=" * 70)

# برای جلوگیری از کندی در رسم نمودارها، یک نمونه از داده‌ها برمی‌داریم
SAMPLE_PER_MONTH = 100_000
multi_frames = []

for month, path in local_files.items():
    tbl = pq.read_table(
        path,
        columns=[
            "tpep_pickup_datetime", "tpep_dropoff_datetime", 
            "trip_distance", "fare_amount", "passenger_count"
        ],
    )
    df_m = tbl.to_pandas()
    
    # ساخت متغیرهای مشتق‌شده
    dur_min = (df_m["tpep_dropoff_datetime"] - df_m["tpep_pickup_datetime"]).dt.total_seconds() / 60
    df_m["duration_minutes"] = dur_min
    df_m["speed_mph"] = np.where(dur_min > 0, df_m["trip_distance"] / (dur_min / 60), np.nan)
    
    # استخراج ساعت سوار شدن
    df_m["pickup_hour"] = df_m["tpep_pickup_datetime"].dt.hour
    
    # اعمال شرایط معتبرسازی (مثل کد جدول همبستگی)
    valid = (
        df_m["duration_minutes"].gt(0)
        & df_m["trip_distance"].gt(0)
        & df_m["fare_amount"].ge(0)
        & np.isfinite(df_m["speed_mph"])
    )
    df_m = df_m[valid]
    
    # نمونه‌گیری برای رسم نمودارها
    if len(df_m) > SAMPLE_PER_MONTH:
        df_m = df_m.sample(SAMPLE_PER_MONTH, random_state=42)
        
    multi_frames.append(df_m)

sample_multi = pd.concat(multi_frames, ignore_index=True)

# ------------------------------
# نمودار ۱: تفاوت ساعت شبانه‌روز در مسافت و کرایه (زیرگروه)
# ------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# فیلتر کردن پرت‌ها برای خوانایی بهتر نمودار
dist_plot = sample_multi[sample_multi["trip_distance"].between(0, 20)]
axes[0].boxplot(
    [dist_plot.loc[dist_plot["pickup_hour"] == h, "trip_distance"].dropna() for h in range(24)],
    tick_labels=range(24), showfliers=False 
)
axes[0].set_title("Trip Distance by Pickup Hour\n(Subgroup comparison, outliers hidden)")
axes[0].set_xlabel("Pickup Hour (0-23)")
axes[0].set_ylabel("Trip Distance (miles)")
axes[0].grid(alpha=0.3)

fare_plot = sample_multi[sample_multi["fare_amount"].between(0, 100)]
axes[1].boxplot(
    [fare_plot.loc[fare_plot["pickup_hour"] == h, "fare_amount"].dropna() for h in range(24)],
    tick_labels=range(24), showfliers=False 
)
axes[1].set_title("Fare Amount by Pickup Hour\n(Subgroup comparison, outliers hidden)")
axes[1].set_xlabel("Pickup Hour (0-23)")
axes[1].set_ylabel("Fare Amount ($)")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "multivariate_hourly_differences.png", dpi=200, bbox_inches="tight")
plt.show()

# ------------------------------
# نمودار ۲: تفاوت کرایه به تفکیک تعداد مسافر (زیرگروه)
# ------------------------------
pax_plot = sample_multi[sample_multi["passenger_count"].between(0, 6)]
fig, ax = plt.subplots(figsize=(8, 4))
ax.boxplot(
    [pax_plot.loc[pax_plot["passenger_count"] == p, "fare_amount"].dropna() for p in range(7)],
    tick_labels=range(7), showfliers=False 
)
ax.set_title("Fare Amount by Passenger Count (Subgroup comparison)")
ax.set_xlabel("Passenger Count")
ax.set_ylabel("Fare Amount ($)")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "multivariate_passenger_differences.png", dpi=200, bbox_inches="tight")
plt.show()

# ------------------------------
# نمودار ۳: ماتریس همبستگی (روابط بین متغیرها)
# ------------------------------
corr_cols = ["trip_distance", "duration_minutes", "fare_amount", "speed_mph", "passenger_count"]
# محاسبه همبستگی با متد اسپیرمن
corr_matrix = sample_multi[corr_cols].corr(method="spearman")

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(corr_matrix.values, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_cols)))
ax.set_yticks(range(len(corr_cols)))
ax.set_xticklabels(corr_cols, rotation=45, ha="right")
ax.set_yticklabels(corr_cols)
ax.set_title("Spearman Correlation Matrix (Sample)")

for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        ax.text(j, i, f"{corr_matrix.values[i, j]:.2f}", ha="center", va="center", 
                color="white" if abs(corr_matrix.values[i, j]) > 0.5 else "black")

fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "multivariate_corr_matrix.png", dpi=200, bbox_inches="tight")
plt.show()

print(f"\n✅ نمودارهای چندمتغیره با موفقیت در مسیر زیر ذخیره شدند:\n{FIGURES_DIR}")
### محاسبه رابطه متغیرها با مکان  
# =========================================================
# Cell — Spatial Relationship Summary
# خلاصه رابطه متغیرها با ۵ منطقه پرتردد
# =========================================================

import pandas as pd
import pyarrow.parquet as pq

# خواندن داده‌ها و اعمال بازه زمانی
spatial_frames = []

for month, path in local_files.items():
    table = pq.read_table(
        path,
        columns=[
            "tpep_pickup_datetime",
            "PULocationID",
            "fare_amount",
            "trip_distance",
            "passenger_count",
        ],
    )

    month_df = table.to_pandas()

    # اطمینان از نوع datetime و فیلتر نیمه‌باز زمانی
    month_df["tpep_pickup_datetime"] = pd.to_datetime(
        month_df["tpep_pickup_datetime"], errors="coerce"
    )

    inside_mask = (
        month_df["tpep_pickup_datetime"].ge(pd.Timestamp(START))
        & month_df["tpep_pickup_datetime"].lt(pd.Timestamp(END_EXCLUSIVE))
    )

    spatial_frames.append(month_df.loc[inside_mask])

if not spatial_frames or all(frame.empty for frame in spatial_frames):
    raise ValueError(
        "در بازه زمانی تعیین‌شده رکوردی پیدا نشد؛ "
        "مقادیر START و END_EXCLUSIVE را بررسی کنید."
    )

df_sub = pd.concat(spatial_frames, ignore_index=True)

# انتخاب ۵ منطقه پرتردد مبدأ
top5_pu_ids = (
    df_sub["PULocationID"]
    .dropna()
    .value_counts()
    .head(5)
    .index
    .tolist()
)

if not top5_pu_ids:
    raise ValueError("هیچ PULocationID معتبری در داده‌های بازه انتخاب‌شده وجود ندارد.")

df_top5 = df_sub[df_sub["PULocationID"].isin(top5_pu_ids)].copy()

# ساخت جدول خلاصه
spatial_relationship_summary = (
    df_top5
    .groupby("PULocationID")
    .agg(
        Total_Trips=("PULocationID", "size"),
        Mean_Fare=("fare_amount", "mean"),
        Median_Fare=("fare_amount", "median"),
        Mean_Distance=("trip_distance", "mean"),
        Median_Distance=("trip_distance", "median"),
        Mean_Passengers=("passenger_count", "mean"),
    )
    .reset_index()
    .rename(columns={"PULocationID": "Zone_ID"})
)

display(
    spatial_relationship_summary
    .round(2)
    .style
    .format(precision=2)
    .set_properties(**{"text-align": "center"})
)

# =========================================================
# Cell — Comprehensive Spatial Multivariate Analysis (Plots Only)
# (تحلیل اکتشافی چندمتغیره مکانی - فقط نمودارها)
# =========================================================
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import seaborn as sns
import numpy as np

print("=" * 70)
print("COMPREHENSIVE SPATIAL MULTIVARIATE PLOTS")
print("=" * 70)

# ۱. خواندن داده‌ها (شامل تمام متغیرهای اصلی و زمان‌ها برای ساخت متغیرهای جدید)
spatial_sub_frames = []
for month, path in local_files.items():
    table = pq.read_table(
        path, 
        columns=["tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID", "fare_amount", "trip_distance", "passenger_count"]
    )
    df_loc = table.to_pandas()
    # فیلتر زمانی
    inside_mask = df_loc["tpep_pickup_datetime"].ge(START) & df_loc["tpep_pickup_datetime"].lt(END_EXCLUSIVE)
    spatial_sub_frames.append(df_loc.loc[inside_mask])

df_sub = pd.concat(spatial_sub_frames, ignore_index=True)

# ۲. ساخت متغیرهای اشتقاقی (مدت زمان و سرعت)
df_sub["duration_minutes"] = (df_sub["tpep_dropoff_datetime"] - df_sub["tpep_pickup_datetime"]).dt.total_seconds() / 60
df_sub["speed_mph"] = np.where(
    df_sub["duration_minutes"] > 0, 
    df_sub["trip_distance"] / (df_sub["duration_minutes"] / 60), 
    np.nan
)

# انتخاب ۵ منطقه پرتردد مبدأ
top5_pu_ids = df_sub["PULocationID"].value_counts().head(5).index.tolist()
df_top5 = df_sub[df_sub["PULocationID"].isin(top5_pu_ids)].copy()

# -----------------------------------------
# ۳. نمودارها: تفاوت زیرگروه‌ها (Boxplots)
# -----------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Distribution of Main Variables across Top 5 Pickup Zones", fontsize=16)

variables = [
    ("fare_amount", "Fare Amount ($)"),
    ("trip_distance", "Trip Distance (miles)"),
    ("passenger_count", "Passenger Count"),
    ("duration_minutes", "Duration (minutes)"),
    ("speed_mph", "Speed (mph)")
]

# رسم Boxplot برای ۵ متغیر اصلی
for i, (col, ylabel) in enumerate(variables):
    row, col_idx = divmod(i, 3)
    data = [df_top5.loc[df_top5["PULocationID"] == z, col].dropna() for z in top5_pu_ids]
    axes[row, col_idx].boxplot(data, tick_labels=[f"Zone {z}" for z in top5_pu_ids], showfliers=False)
    axes[row, col_idx].set_title(f"{col} by Zone")
    axes[row, col_idx].set_ylabel(ylabel)
    axes[row, col_idx].grid(alpha=0.3)

# حذف محور خالی ششم
fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.savefig(FIGURES_DIR / "spatial_multivariate_boxplots.png", dpi=200, bbox_inches="tight")
plt.show()

# -----------------------------------------
# ۴. نمودار تعامل: Scatter Plot (مسافت در برابر کرایه با رنگ‌بندی مناطق)
# -----------------------------------------
plt.figure(figsize=(10, 6))
# نمونه‌گیری برای جلوگیری از شلوغی بیش از حد (۱۰ هزار نمونه تصادفی)
df_sample = df_top5.sample(n=min(10000, len(df_top5)), random_state=42)
colors = ['blue', 'orange', 'green', 'red', 'purple']

for i, zone in enumerate(top5_pu_ids):
    zone_data = df_sample[df_sample["PULocationID"] == zone]
    plt.scatter(zone_data["trip_distance"], zone_data["fare_amount"], 
                label=f"Zone {zone}", alpha=0.5, s=15, color=colors[i])

plt.title("Interaction: Trip Distance vs Fare Amount (Top 5 Zones)")
plt.xlabel("Trip Distance (miles)")
plt.ylabel("Fare Amount ($)")
plt.legend()
plt.grid(alpha=0.3)
# (محدودیت‌های محور برای نمایش بهتر حفظ شده‌اند، اما داده‌ها حذف نشده‌اند)
plt.ylim(0, 150) 
plt.xlim(0, 30)
plt.savefig(FIGURES_DIR / "spatial_multivariate_scatter.png", dpi=200, bbox_inches="tight")
plt.show()

# -----------------------------------------
# ۵. ماتریس همبستگی برای زیرگروه‌ها (رابطه بین متغیرها)
# -----------------------------------------
fig, axes = plt.subplots(1, 5, figsize=(25, 5))
fig.suptitle("Correlation Matrices within Top 5 Zones", fontsize=16)

cols_to_corr = ["trip_distance", "fare_amount", "duration_minutes", "speed_mph", "passenger_count"]

for i, zone in enumerate(top5_pu_ids):
    zone_data = df_top5[df_top5["PULocationID"] == zone]
    corr_matrix = zone_data[cols_to_corr].corr(method='spearman')
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, 
                ax=axes[i], cbar=False if i < 4 else True)
    axes[i].set_title(f"Zone {zone}")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "spatial_multivariate_corr.png", dpi=200, bbox_inches="tight")
plt.show()

print(f"\n✅ تمام نمودارها در مسیر {FIGURES_DIR} ذخیره شدند.")
