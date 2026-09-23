"""
app.py  —  MEGHA-AI
All helper modules inlined. No external .py files needed.
Run via: streamlit run app.py
"""

# ════════════════════════════════════════════════════════
# DATE_UTILS
# ════════════════════════════════════════════════════════
import re
import json
import glob
import os
from datetime import datetime, timedelta, date as date_cls

try:
    from dateutil import parser as _dateutil_parser
    _HAVE_DATEUTIL = True
except ImportError:
    _HAVE_DATEUTIL = False

IST_OFFSET = timedelta(hours=5, minutes=30)

RELATIVE_TERMS = {"today": 0, "tomorrow": 1, "day after tomorrow": 2, "yesterday": -1}


def now_ist():
    return datetime.utcnow() + IST_OFFSET


def resolve_date_term(term):
    if not term:
        return None
    t = str(term).strip().lower()
    if t in RELATIVE_TERMS:
        return (now_ist() + timedelta(days=RELATIVE_TERMS[t])).date()
    m = re.match(r"^(\d{2})-(\d{2})-(\d{4})$", t)
    if m:
        d, mo, y = map(int, m.groups())
        return date_cls(y, mo, d)
    m = re.match(r"^(\d{2})[/.](\d{2})[/.](\d{4})$", t)
    if m:
        d, mo, y = map(int, m.groups())
        return date_cls(y, mo, d)
    if _HAVE_DATEUTIL:
        try:
            dt = _dateutil_parser.parse(t, dayfirst=True, fuzzy=True, default=now_ist())
            return dt.date()
        except (ValueError, OverflowError, TypeError):
            return None
    return None


def get_date_range(start_date, n_days):
    return [start_date + timedelta(days=i) for i in range(n_days)]


def day_label_for_date(target_date, today=None):
    if today is None:
        today = now_ist().date()
    diff = (target_date - today).days
    if diff == 0:
        return "Today"
    if diff == 1:
        return "Tomorrow"
    if diff == 2:
        return "Day after tomorrow"
    return target_date.strftime("%A, %d %b")


# ════════════════════════════════════════════════════════
# NC_DATA
# ════════════════════════════════════════════════════════
import numpy as np
import pandas as pd
import xarray as xr

STEP_HOURS = 6
TP_IS_CUMULATIVE = True

_RAIN_STEPS_24H = [
    (0.1, "No rain"),
    (2.5, "Very light rain"),
    (7.6, "Light rain"),
    (35.6, "Moderate rain"),
    (64.5, "Heavy rain"),
    (124.5, "Very heavy rain"),
]
RAIN_CATEGORY_ORDER = [
    "No rain", "Very light rain", "Light rain", "Moderate rain",
    "Heavy rain", "Very heavy rain", "Extremely heavy rain",
]
_THRESHOLD_ALIASES = {
    "light": "Light rain",
    "moderate": "Moderate rain",
    "heavy": "Heavy rain",
    "very heavy": "Very heavy rain",
    "extremely heavy": "Extremely heavy rain",
}
DAYPART_ORDER = ["Morning", "Afternoon", "Evening", "Night"]
VARIABLE_META = {
    "rainfall":        {"label": "Rainfall",         "unit": "mm",  "cmap": "viridis_r"},
    "temperature_max": {"label": "Max Temperature",  "unit": "\u00b0C", "cmap": "jet"},
    "temperature_min": {"label": "Min Temperature",  "unit": "\u00b0C", "cmap": "jet"},
    "wind":            {"label": "Wind Speed",        "unit": "km/h","cmap": "Greens"},
    "humidity":        {"label": "Humidity",          "unit": "%",   "cmap": "PuBu"},
}

# Paths resolved from env var set in Cell 2 of the notebook
_APP_DIR = os.environ.get("MEGHA_APP_DIR", os.path.dirname(os.path.abspath(__file__)))
NC_DIR   = _APP_DIR
NC_PATH  = None
GEONAMES_PATH       = os.path.join(_APP_DIR, "IN.txt")
DISTRICT_GEOJSON_PATH = os.path.join(_APP_DIR, "IND-DIS-732.json")


def find_latest_nc_file(directory="."):
    pattern = os.path.join(directory, "ecmwf_aifs_india_*_00z_merged.nc")
    candidates = []
    for path in glob.glob(pattern):
        m = re.search(r"ecmwf_aifs_india_(\d{8})_00z_merged\.nc$", os.path.basename(path))
        if m:
            candidates.append((m.group(1), path))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def parse_ic_from_filename(nc_path):
    fname = os.path.basename(nc_path)
    m = re.search(r"(\d{8})_(\d{2})z", fname)
    if not m:
        raise ValueError(f"Could not parse IC date/hour from filename: {fname}")
    date_str, hour_str = m.groups()
    return datetime.strptime(date_str + hour_str, "%Y%m%d%H")


def open_forecast(nc_path, tp_is_cumulative=TP_IS_CUMULATIVE):
    ds = xr.open_dataset(nc_path)
    try:
        ic_utc = parse_ic_from_filename(nc_path)
    except ValueError:
        run_date = ds.attrs.get("run_date")
        run_hour = ds.attrs.get("run_hour", "00z").replace("z", "")
        if not run_date:
            raise
        ic_utc = datetime.strptime(f"{run_date}{run_hour}", "%Y%m%d%H")
    n_steps = ds.sizes["time"]
    step_hours = np.arange(n_steps) * STEP_HOURS
    valid_utc = pd.to_datetime([ic_utc + timedelta(hours=int(h)) for h in step_hours])
    valid_ist = valid_utc + IST_OFFSET
    if tp_is_cumulative and "tp" in ds.data_vars:
        tp_vals = ds["tp"].values
        tp_interval = np.empty_like(tp_vals)
        tp_interval[0] = tp_vals[0]
        tp_interval[1:] = np.diff(tp_vals, axis=0)
        tp_interval = np.clip(tp_interval, 0, None)
        ds["tp_cumulative"] = (ds["tp"].dims, tp_vals)
        ds["tp"] = (ds["tp"].dims, tp_interval.astype(np.float32))
    ds = ds.assign_coords(
        valid_time_utc=("time", valid_utc),
        valid_time_ist=("time", valid_ist),
    )
    ds.attrs["ic_utc"] = ic_utc.isoformat()
    ds.attrs["nc_path"] = nc_path
    return ds


def get_point_timeseries(ds, lat, lon):
    lat_min, lat_max = float(ds["latitude"].min()), float(ds["latitude"].max())
    lon_min, lon_max = float(ds["longitude"].min()), float(ds["longitude"].max())
    if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
        raise ValueError(
            f"Point ({lat:.3f}, {lon:.3f}) is outside the forecast domain "
            f"(lat {lat_min}-{lat_max}, lon {lon_min}-{lon_max})."
        )
    point = ds.interp(latitude=lat, longitude=lon, method="linear")
    return point.to_dataframe().reset_index()


def classify_rainfall(mm, hours=24):
    scale = hours / 24.0
    for limit, label in _RAIN_STEPS_24H:
        if mm <= limit * scale:
            return label
    return "Extremely heavy rain"


def category_at_least(category, threshold_word):
    key = _THRESHOLD_ALIASES.get((threshold_word or "").strip().lower(), "Heavy rain")
    if category not in RAIN_CATEGORY_ORDER or key not in RAIN_CATEGORY_ORDER:
        return False
    return RAIN_CATEGORY_ORDER.index(category) >= RAIN_CATEGORY_ORDER.index(key)


def classify_cloud(pct):
    if pct <= 10: return "clear skies"
    if pct <= 40: return "mostly clear skies"
    if pct <= 70: return "partly cloudy skies"
    if pct <= 90: return "generally cloudy skies"
    return "overcast skies"


def classify_humidity(pct):
    if pct < 40: return "low"
    if pct < 70: return "moderate"
    return "high"


def wind_dir_label(u, v):
    deg = (270 - np.degrees(np.arctan2(v, u))) % 360
    dirs = [
        "Northerly", "North-Easterly", "Easterly", "South-Easterly",
        "Southerly", "South-Westerly", "Westerly", "North-Westerly",
    ]
    idx = int(((deg + 22.5) % 360) // 45)
    return dirs[idx], deg


def get_daypart(hour):
    if 4 <= hour < 10:  return "Morning"
    if 10 <= hour < 16: return "Afternoon"
    if 16 <= hour < 20: return "Evening"
    return "Night"


def aggregate_day(df, target_date, only_future=False, now=None):
    day_df = df[df["valid_time_ist"].dt.date == target_date].copy()
    if only_future:
        now = now or now_ist()
        day_df = day_df[day_df["valid_time_ist"] >= now]
    if day_df.empty:
        return None
    day_df["daypart"] = day_df["valid_time_ist"].dt.hour.apply(get_daypart)
    return day_df


def summarize_period(sub_df):
    tp_total = float(sub_df["tp"].sum())
    tmax = float((sub_df["t2m"] - 273.15).max())
    tmin = float((sub_df["t2m"] - 273.15).min())
    tcc_avg = float(sub_df["tcc"].mean())
    q_avg = float(sub_df["q1000"].mean())
    u_avg = float(sub_df["u10"].mean())
    v_avg = float(sub_df["v10"].mean())
    wind_speed_kmh = float(np.hypot(u_avg, v_avg) * 3.6)
    wind_label, wind_deg = wind_dir_label(u_avg, v_avg)
    hours = len(sub_df) * STEP_HOURS
    return {
        "tp_mm":         round(tp_total, 1),
        "tmax_c":        round(tmax, 1),
        "tmin_c":        round(tmin, 1),
        "cloud_pct":     round(tcc_avg, 1),
        "humidity_pct":  round(q_avg, 1),
        "wind_kmh":      round(wind_speed_kmh, 1),
        "wind_dir":      wind_label,
        "rain_category": classify_rainfall(tp_total, hours=hours),
        "hours":         hours,
    }


def should_split_daypart(day_df):
    parts = day_df.groupby("daypart")["tp"].sum()
    if len(parts) < 2:
        return False
    cats = {classify_rainfall(v, hours=STEP_HOURS) for v in parts.values}
    return len(cats) > 1 and parts.max() >= 3.0


def format_period_text(label, stats, show_minmax=True):
    cloud_desc    = classify_cloud(stats["cloud_pct"])
    humidity_desc = classify_humidity(stats["humidity_pct"])
    cat = stats["rain_category"]
    mm  = stats["tp_mm"]
    if cat == "No rain":
        rain_phrase = "No significant rainfall is expected"
    else:
        rain_phrase = f"{cat} of {mm:.0f} mm is likely"
    text = f"{label}: {rain_phrase} with {cloud_desc}. "
    if show_minmax:
        text += (
            f"Maximum temperature may be around {stats['tmax_c']:.0f}\u00b0C, "
            f"while minimum temperature may be around {stats['tmin_c']:.0f}\u00b0C. "
        )
    else:
        avg_t = (stats["tmax_c"] + stats["tmin_c"]) / 2
        text += f"Temperature may be around {avg_t:.0f}\u00b0C. "
    text += (
        f"Winds will be from the {stats['wind_dir']} direction at around "
        f"{stats['wind_kmh']:.0f} km/h. "
        f"Humidity will remain {humidity_desc} at around {stats['humidity_pct']:.0f}%."
    )
    return text


def build_daily_forecast_text(day_label, df, target_date):
    now = now_ist()
    is_today = target_date == now.date()
    day_df = aggregate_day(df, target_date, only_future=is_today, now=now)
    if day_df is None:
        if is_today:
            return "No further forecast periods remain for today."
        return None
    prefix = ""
    if is_today and len(day_df) < 4:
        prefix = "_(Showing the remaining forecast periods for today.)_\n\n"
    if should_split_daypart(day_df):
        lines = []
        for part in DAYPART_ORDER:
            part_df = day_df[day_df["daypart"] == part]
            if part_df.empty:
                continue
            stats = summarize_period(part_df)
            lines.append(format_period_text(f"{day_label} {part}", stats, show_minmax=False))
        return prefix + "\n\n".join(lines)
    stats = summarize_period(day_df)
    return prefix + format_period_text(day_label, stats, show_minmax=True)


def build_compact_day_summary(df, target_date):
    day_df = aggregate_day(df, target_date)
    if day_df is None:
        return None
    return summarize_period(day_df)


def build_multiday_table(df, dates):
    now = now_ist()
    today = now.date()
    rows = []
    for d in dates:
        is_today = d == today
        day_df = aggregate_day(df, d, only_future=is_today, now=now)
        label = day_label_for_date(d, today)
        if day_df is None:
            rows.append({
                "Date": d.strftime("%d-%m-%Y"), "Day": label,
                "Rain Category": "No data", "Rainfall (mm)": None,
                "Max Temp (\u00b0C)": None, "Min Temp (\u00b0C)": None,
                "Wind": None, "Humidity (%)": None,
            })
            continue
        stats = summarize_period(day_df)
        rows.append({
            "Date": d.strftime("%d-%m-%Y"),
            "Day": label + (" (remaining)" if is_today and len(day_df) < 4 else ""),
            "Rain Category": stats["rain_category"],
            "Rainfall (mm)": stats["tp_mm"],
            "Max Temp (\u00b0C)": stats["tmax_c"],
            "Min Temp (\u00b0C)": stats["tmin_c"],
            "Wind": f"{stats['wind_dir']} {stats['wind_kmh']:.0f} km/h",
            "Humidity (%)": stats["humidity_pct"],
        })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════
# DISTRICTS
# ════════════════════════════════════════════════════════
import geopandas as gpd
from shapely.geometry import Point


def load_district_gdf(geojson_path):
    gdf = gpd.read_file(geojson_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    gdf = gdf.reset_index(drop=True)
    gdf["ST_NM_UP"] = gdf["ST_NM"].str.upper().str.strip()
    return gdf


def is_india_scope(state_name):
    return state_name is None or state_name.strip().upper() in ("INDIA", "ALL INDIA", "ALL-INDIA")


def build_grid_district_index(ds, gdf):
    lats = ds["latitude"].values
    lons = ds["longitude"].values
    glon, glat = np.meshgrid(lons, lats)
    pts = gpd.GeoDataFrame(
        {"grid_pos": np.arange(glon.size)},
        geometry=[Point(xy) for xy in zip(glon.ravel(), glat.ravel())],
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(pts, gdf[["geometry"]], how="left", predicate="within")
    joined = joined[~joined["grid_pos"].duplicated(keep="first")].sort_values("grid_pos")
    id_grid = joined["index_right"].values
    id_grid = np.where(pd.isna(id_grid), -1, id_grid).astype(int)
    return id_grid.reshape(glon.shape)


def _day_time_indices(ds, target_date):
    times = pd.to_datetime(ds["valid_time_ist"].values)
    mask = np.array([t.date() == target_date for t in times])
    if not mask.any():
        return None
    return np.where(mask)[0]


def day_total_grid(ds, target_date):
    idx = _day_time_indices(ds, target_date)
    if idx is None:
        return None
    return ds["tp"].isel(time=idx).sum(dim="time").values


def compute_day_variable_grid(ds, target_date, variable):
    idx = _day_time_indices(ds, target_date)
    if idx is None:
        return None
    if variable == "rainfall":
        return ds["tp"].isel(time=idx).sum(dim="time").values
    if variable == "temperature_max":
        return (ds["t2m"].isel(time=idx) - 273.15).max(dim="time").values
    if variable == "temperature_min":
        return (ds["t2m"].isel(time=idx) - 273.15).min(dim="time").values
    if variable == "wind":
        u = ds["u10"].isel(time=idx).mean(dim="time")
        v = ds["v10"].isel(time=idx).mean(dim="time")
        return (np.hypot(u, v) * 3.6).values
    if variable == "humidity":
        return ((ds["t2m"].isel(time=idx) / 0.622) * 100).min(dim="time").values
    raise ValueError(f"Unknown variable '{variable}'")


def _scope_rows(gdf, state_name):
    if is_india_scope(state_name):
        return gdf
    return gdf[gdf["ST_NM_UP"] == state_name.strip().upper()]


def district_rainfall_table(ds, gdf, id_grid, state_name, target_date):
    day_grid = day_total_grid(ds, target_date)
    if day_grid is None:
        return None
    state_rows = _scope_rows(gdf, state_name)
    if state_rows.empty:
        return pd.DataFrame(columns=["district", "avg_rain_mm", "category"])
    results = []
    for row_idx, row in state_rows.iterrows():
        cell_mask = id_grid == row_idx
        if not cell_mask.any():
            continue
        avg_rain = float(np.nanmean(day_grid[cell_mask]))
        results.append({
            "district":    row["DISTRICT"],
            "avg_rain_mm": round(avg_rain, 1),
            "category":    classify_rainfall(avg_rain, hours=24),
        })
    return pd.DataFrame(results).sort_values("avg_rain_mm", ascending=False).reset_index(drop=True)


def state_rainfall_table(ds, gdf, id_grid, target_date):
    day_grid = day_total_grid(ds, target_date)
    if day_grid is None:
        return None
    records = []
    for state_nm, group in gdf.groupby("ST_NM_UP"):
        idxs = group.index.values
        cell_mask = np.isin(id_grid, idxs)
        if not cell_mask.any():
            continue
        avg_rain = float(np.nanmean(day_grid[cell_mask]))
        records.append({
            "state":       state_nm.title(),
            "avg_rain_mm": round(avg_rain, 1),
            "category":    classify_rainfall(avg_rain, hours=24),
        })
    return pd.DataFrame(records).sort_values("avg_rain_mm", ascending=False).reset_index(drop=True)


def district_full_table(ds, gdf, id_grid, target_date, state_name=None):
    rain_grid = compute_day_variable_grid(ds, target_date, "rainfall")
    if rain_grid is None:
        return None
    tmax_grid = compute_day_variable_grid(ds, target_date, "temperature_max")
    tmin_grid = compute_day_variable_grid(ds, target_date, "temperature_min")
    hum_grid  = compute_day_variable_grid(ds, target_date, "humidity")
    idx       = _day_time_indices(ds, target_date)
    u_grid    = ds["u10"].isel(time=idx).mean(dim="time").values
    v_grid    = ds["v10"].isel(time=idx).mean(dim="time").values
    rows_source = _scope_rows(gdf, state_name)
    if rows_source.empty:
        return pd.DataFrame()
    results = []
    for row_idx, row in rows_source.iterrows():
        cell_mask = id_grid == row_idx
        if not cell_mask.any():
            continue
        u_mean = float(np.nanmean(u_grid[cell_mask]))
        v_mean = float(np.nanmean(v_grid[cell_mask]))
        wind_label, _ = wind_dir_label(u_mean, v_mean)
        rain_val = float(np.nanmean(rain_grid[cell_mask]))
        results.append({
            "State":          row["ST_NM"].title(),
            "District":       row["DISTRICT"].title(),
            "Rainfall (mm)":  round(rain_val, 1),
            "Max Temp (\u00b0C)": round(float(np.nanmean(tmax_grid[cell_mask])), 1),
            "Min Temp (\u00b0C)": round(float(np.nanmean(tmin_grid[cell_mask])), 1),
            "Wind":           f"{wind_label} {np.hypot(u_mean, v_mean) * 3.6:.0f} km/h",
            "Humidity (%)":   round(float(np.nanmean(hum_grid[cell_mask])), 1),
            "Category":       classify_rainfall(rain_val, hours=24),
        })
    return pd.DataFrame(results).sort_values("Rainfall (mm)", ascending=False).reset_index(drop=True)


def summarize_district_table(table, scope_label, target_date):
    if table is None or table.empty:
        return f"No forecast data available for {scope_label} on {target_date}."
    avg_rain = table["Rainfall (mm)"].mean()
    avg_tmax = table["Max Temp (\u00b0C)"].mean()
    avg_tmin = table["Min Temp (\u00b0C)"].mean()
    wettest  = table.iloc[0]
    category = classify_rainfall(avg_rain, hours=24)
    return (
        f"**{scope_label}** \u2014 {target_date}: average rainfall across districts is "
        f"~{avg_rain:.0f} mm ({category}), with temperatures ranging roughly "
        f"{avg_tmin:.0f}\u2013{avg_tmax:.0f}\u00b0C. Highest rainfall expected in "
        f"**{wettest['District']}** (~{wettest['Rainfall (mm)']:.0f} mm)."
    )


def district_variable_series(ds, gdf, id_grid, target_date, variable, state_name=None):
    grid = compute_day_variable_grid(ds, target_date, variable)
    if grid is None:
        return None
    rows_source = _scope_rows(gdf, state_name)
    if rows_source.empty:
        return rows_source
    values = []
    for row_idx in rows_source.index:
        cell_mask = id_grid == row_idx
        values.append(float(np.nanmean(grid[cell_mask])) if cell_mask.any() else np.nan)
    out = rows_source.copy()
    out["value"] = values
    return out


def get_state_geometry(gdf, state_name):
    rows = _scope_rows(gdf, state_name)
    if rows.empty:
        return None
    try:
        return rows.union_all()
    except AttributeError:
        return rows.unary_union


# ════════════════════════════════════════════════════════
# REGION_PLOT
# ════════════════════════════════════════════════════════
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def _scope_title(state_name):
    return "India" if is_india_scope(state_name) else state_name.title()


def plot_region_pattern(ds, gdf, id_grid, state_name, target_date, variable="rainfall", grid_res=220):
    if variable not in VARIABLE_META:
        raise ValueError(f"Unknown variable '{variable}'.")
    grid = compute_day_variable_grid(ds, target_date, variable)
    if grid is None:
        raise ValueError(f"No forecast data available for {target_date}.")
    rows = gdf if is_india_scope(state_name) else gdf[gdf["ST_NM_UP"] == state_name.strip().upper()]
    if not is_india_scope(state_name) and rows.empty:
        raise ValueError(f"No district data found for '{state_name}'.")
    scope_idx = rows.index.values
    cell_mask = np.isin(id_grid, scope_idx)
    if not cell_mask.any():
        raise ValueError(f"No forecast grid cells fall inside '{state_name}'.")
    lats = ds["latitude"].values
    lons = ds["longitude"].values
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)
    valid_lat = lat_mesh[cell_mask]
    valid_lon = lon_mesh[cell_mask]
    valid_val = grid[cell_mask]
    lat_idx, lon_idx = np.where(cell_mask)
    lat_min, lat_max = lats[lat_idx].min(), lats[lat_idx].max()
    lon_min, lon_max = lons[lon_idx].min(), lons[lon_idx].max()
    pad = 0.15
    fine_lon = np.linspace(lon_min, lon_max, grid_res)
    fine_lat = np.linspace(lat_min, lat_max, grid_res)
    flon, flat = np.meshgrid(fine_lon, fine_lat)
    fine_val = griddata((valid_lon, valid_lat), valid_val, (flon, flat), method="linear")
    region_geom = get_state_geometry(gdf, state_name)
    pts = gpd.GeoSeries([Point(xy) for xy in zip(flon.ravel(), flat.ravel())], crs="EPSG:4326")
    inside = pts.within(region_geom).values.reshape(flon.shape)
    fine_val = np.where(inside, fine_val, np.nan)
    meta = VARIABLE_META[variable]
    is_india = is_india_scope(state_name)
    fig, ax = plt.subplots(figsize=(8, 9) if is_india else (8, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    cf = ax.contourf(flon, flat, fine_val, levels=20, cmap=meta["cmap"])
    plt.colorbar(cf, ax=ax, label=f"{meta['label']} ({meta['unit']})")
    rows.boundary.plot(ax=ax, color="black", linewidth=0.35, alpha=0.6)
    gpd.GeoSeries([region_geom], crs="EPSG:4326").boundary.plot(ax=ax, color="black", linewidth=1.3)
    ax.set_xlim(lon_min - pad, lon_max + pad)
    ax.set_ylim(lat_min - pad, lat_max + pad)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    lbl = day_label_for_date(target_date)
    ax.set_title(f"{meta['label']} \u2014 {_scope_title(state_name)} \u2014 {lbl} ({target_date})")
    plt.tight_layout()
    return fig


def plot_region_spatial(ds, gdf, id_grid, state_name, target_date, variable="rainfall"):
    if variable not in VARIABLE_META:
        raise ValueError(f"Unknown variable '{variable}'.")
    table = district_variable_series(
        ds, gdf, id_grid, target_date, variable,
        state_name=None if is_india_scope(state_name) else state_name,
    )
    if table is None or table.empty:
        raise ValueError(f"No forecast data for '{state_name}' on {target_date}.")
    meta = VARIABLE_META[variable]
    is_india = is_india_scope(state_name)
    fig, ax = plt.subplots(figsize=(8, 9) if is_india else (8, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    table.plot(
        column="value", ax=ax, cmap=meta["cmap"], edgecolor="black", linewidth=0.4,
        legend=True,
        legend_kwds={"label": f"{meta['label']} ({meta['unit']})", "shrink": 0.6},
        missing_kwds={"color": "white", "hatch": "///"},
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    lbl = day_label_for_date(target_date)
    ax.set_title(f"District-wise {meta['label']} \u2014 {_scope_title(state_name)} \u2014 {lbl} ({target_date})")
    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════
# GEO_PLACES
# ════════════════════════════════════════════════════════
from rapidfuzz import process, fuzz


class IndiaGeoAgent:
    def __init__(self, geonames_path="IN.txt", extra_csvs=None):
        cols = [
            "geonameid","name","asciiname","alternatenames",
            "latitude","longitude","feature_class","feature_code",
            "country_code","cc2","admin1","admin2","admin3","admin4",
            "population","elevation","dem","timezone","modification_date",
        ]
        df = pd.read_csv(
            geonames_path, sep="\t", header=None, names=cols,
            dtype={"latitude": float, "longitude": float},
            low_memory=False, encoding="utf-8",
        )
        df = df[df["country_code"] == "IN"][[
            "name","asciiname","alternatenames","latitude","longitude",
            "feature_class","feature_code","admin1","population",
        ]].copy()
        if extra_csvs:
            for csv in extra_csvs:
                extra = pd.read_csv(csv)
                df = pd.concat([df, extra], ignore_index=True)
        self.df = df.reset_index(drop=True)
        names = []
        self.name_to_idx = []
        for idx, row in self.df.iterrows():
            candidates = [row["name"], row["asciiname"]]
            if pd.notna(row["alternatenames"]):
                candidates += str(row["alternatenames"]).split(",")
            for n in candidates:
                n = str(n).strip()
                if n:
                    names.append(n)
                    self.name_to_idx.append(idx)
        self.names = names

    def geocode(self, query, limit=5, min_score=75):
        if not self.names:
            return []
        matches = process.extract(query, self.names, scorer=fuzz.WRatio, limit=limit * 5)
        results = []
        seen = set()
        for name, score, list_idx in matches:
            if score < min_score:
                continue
            idx = self.name_to_idx[list_idx]
            row = self.df.iloc[idx]
            key = (round(row["latitude"], 5), round(row["longitude"], 5))
            if key in seen:
                continue
            seen.add(key)
            results.append({
                "name": row["name"],
                "lat":  float(row["latitude"]),
                "lon":  float(row["longitude"]),
                "score": score,
            })
            if len(results) >= limit:
                break
        return results

    def get_lat_lon(self, place):
        res = self.geocode(place, limit=1)
        if res:
            return res[0]["lat"], res[0]["lon"]
        return None


# ════════════════════════════════════════════════════════
# INTENT PARSING  (rule-based default, optional Groq LLM)
# ════════════════════════════════════════════════════════

# Leave empty to use built-in rule-based parser (no API key needed).
# Get a FREE Groq key at https://console.groq.com and paste it here
# for smarter handling of complex / unusual phrasings.
# Optional: set GROQ_API_KEY in Streamlit Cloud Secrets or as an environment variable.
# If no key is configured, MEGHA-AI automatically uses its built-in rule-based parser.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
try:
    import streamlit as _st_for_secrets
    GROQ_API_KEY = _st_for_secrets.secrets.get("GROQ_API_KEY", GROQ_API_KEY)
except Exception:
    pass

DEFAULT_PARSED = {
    "intent": "other", "location": None, "date": None, "range_days": None,
    "threshold": None, "plot": False, "plot_variables": ["rainfall"], "plot_style": "pattern",
}

INDIA_STATES = {
    "andhra pradesh","arunachal pradesh","assam","bihar","chhattisgarh",
    "goa","gujarat","haryana","himachal pradesh","jharkhand","karnataka",
    "kerala","madhya pradesh","maharashtra","manipur","meghalaya",
    "mizoram","nagaland","odisha","punjab","rajasthan","sikkim",
    "tamil nadu","telangana","tripura","uttar pradesh","uttarakhand",
    "west bengal","delhi","jammu and kashmir","ladakh",
    "andaman and nicobar","chandigarh","dadra","lakshadweep","puducherry",
}


def _rule_based_parse(question):
    """Fast, zero-dependency intent parser covering common query patterns."""
    q = question.lower().strip()
    parsed = dict(DEFAULT_PARSED)

    # date
    if "day after tomorrow" in q:
        parsed["date"] = "day after tomorrow"
    elif "tomorrow" in q:
        parsed["date"] = "tomorrow"
    elif "yesterday" in q:
        parsed["date"] = "yesterday"
    else:
        parsed["date"] = "today"
    dm = re.search(r"(\d{1,2})[/\-. ](\d{1,2})[/\-. ](\d{4})", q)
    if dm:
        parsed["date"] = dm.group(0)
    else:
        dm2 = re.search(r"(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*(?:\s+\d{4})?", q)
        if dm2:
            parsed["date"] = dm2.group(0)

    # range_days
    rm = re.search(r"next\s+(\d+)\s+day", q)
    if rm:
        parsed["range_days"] = int(rm.group(1))

    # plot
    parsed["plot"] = any(w in q for w in ["plot", "chart", "graph", "map", "show map", "visuali"])

    # plot_variables
    pv = []
    if any(w in q for w in ["rain", "precip", "rainfall"]): pv.append("rainfall")
    if any(w in q for w in ["temp", "temperature", "heat", "cold"]): pv.append("temperature")
    if "wind" in q: pv.append("wind")
    if "humid" in q: pv.append("humidity")
    parsed["plot_variables"] = pv if pv else ["rainfall"]

    # plot_style
    parsed["plot_style"] = "spatial" if any(w in q for w in ["spatial", "district-wise", "districtwise"]) else "pattern"

    # threshold
    for t in ["very heavy", "extremely heavy", "heavy", "moderate", "light"]:
        if t in q:
            parsed["threshold"] = t
            break

    # region plot
    if parsed["plot"] and any(w in q for w in ["map", "plot", "spatial", "chart"]):
        if any(w in q for w in ["india", "all india", "entire india", "whole india"]):
            parsed["intent"] = "region_plot"
            parsed["location"] = "India"
            return parsed
        for state in INDIA_STATES:
            if state in q:
                parsed["intent"] = "region_plot"
                parsed["location"] = state.title()
                return parsed

    # state threshold query
    if re.search(r"which\s+states?", q) and parsed.get("threshold"):
        parsed["intent"] = "state_threshold_query"
        parsed["location"] = None
        return parsed

    # district rain query
    if re.search(r"which\s+districts?", q) or re.search(r"districts?\s+(?:of|in)\s+", q):
        for state in INDIA_STATES:
            if state in q:
                parsed["intent"] = "district_rain_query"
                parsed["location"] = state.title()
                if not parsed.get("threshold"):
                    parsed["threshold"] = "heavy"
                return parsed

    # state district table
    table_kw = ["all districts", "district table", "district weather",
                "weather of all districts", "weather of districts",
                "show districts", "list districts"]
    if any(kw in q for kw in table_kw):
        for state in INDIA_STATES:
            if state in q:
                parsed["intent"] = "state_district_table"
                parsed["location"] = state.title()
                return parsed

    # state summary
    for state in INDIA_STATES:
        if state in q:
            parsed["intent"] = "state_summary"
            parsed["location"] = state.title()
            return parsed

    # city forecast — extract location after common prepositions
    city_patterns = [
        r"(?:weather|forecast|rainfall|rain|temperature|temp)\s+(?:in|for|at|of|around|near)\s+([a-z][a-z ]{1,25?})(?:\s+(?:today|tomorrow|on|for|next)|$|\?)",
        r"(?:in|for|at|of|around|near)\s+([a-z][a-z ]{1,25?})(?:\s+(?:today|tomorrow|weather|rain|forecast|on|for|next)|$|\?)",
        r"(?:will it rain|is it going to rain)\s+(?:in|at|near)\s+([a-z][a-z ]{1,25?})(?:\s|$|\?)",
    ]
    for pat in city_patterns:
        cm = re.search(pat, q)
        if cm:
            loc = cm.group(1).strip().rstrip("?").strip()
            if loc and len(loc) > 1 and loc not in ("india", "today", "tomorrow", "the"):
                parsed["intent"] = "city_forecast"
                parsed["location"] = loc.title()
                if parsed["plot"] and not parsed.get("range_days"):
                    parsed["range_days"] = 7
                return parsed

    # fallback: word after "in/for/at"
    fb = re.search(r"(?:in|for|at|of)\s+([A-Za-z][a-z]+(?:\s[A-Za-z][a-z]+){0,2})", question)
    if fb:
        loc = fb.group(1).strip()
        if loc.lower() not in ("india", "today", "tomorrow", "the", "a", "an"):
            parsed["intent"] = "city_forecast"
            parsed["location"] = loc.title()
            return parsed

    return parsed


def _groq_parse(question):
    """Call Groq API for intent parsing (optional upgrade)."""
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": (
                    "You are an intent parser for a rainfall/weather forecast assistant covering India.\n"
                    "Classify into JSON: intent (city_forecast|state_summary|state_district_table|"
                    "district_rain_query|state_threshold_query|region_plot|other), location, "
                    "date (today/tomorrow/day after tomorrow/null), range_days (int or null), "
                    "threshold (light/moderate/heavy/very heavy/null), plot (bool), "
                    "plot_variables (list), plot_style (spatial/pattern).\n"
                    "Output ONLY valid JSON, no markdown, no explanation."
                )},
                {"role": "user", "content": question},
            ],
            temperature=0,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            p = json.loads(m.group())
            merged = dict(DEFAULT_PARSED)
            merged.update({k: v for k, v in p.items() if v is not None})
            return merged
    except Exception as e:
        import streamlit as _st
        _st.sidebar.warning(f"Groq parse failed, using rule-based parser: {e}")
    return _rule_based_parse(question)


def parse_intent(question):
    """Use Groq LLM if API key is set, otherwise use rule-based parser."""
    if GROQ_API_KEY.strip():
        return _groq_parse(question)
    return _rule_based_parse(question)


def expand_plot_variables(vars_list):
    out = []
    for v in vars_list or []:
        if v == "temperature":
            out += ["temperature_max", "temperature_min"]
        elif v in ("rainfall", "wind", "humidity", "temperature_max", "temperature_min"):
            out.append(v)
    seen, result = set(), []
    for v in out:
        if v not in seen:
            seen.add(v)
            result.append(v)
    return result or ["rainfall"]


# ════════════════════════════════════════════════════════
# STREAMLIT APP
# ════════════════════════════════════════════════════════
import streamlit as st

st.set_page_config(page_title="MEGHA-AI", page_icon="\U0001f326\ufe0f", layout="wide")

PLOT_VAR_META = {
    "rainfall":        {"ylabel": "Rainfall (mm)",  "title": "Daily Rainfall",    "color": "#2b6cb0"},
    "temperature_max": {"ylabel": "\u00b0C",         "title": "Max Temperature",   "color": "#e53e3e"},
    "temperature_min": {"ylabel": "\u00b0C",         "title": "Min Temperature",   "color": "#3182ce"},
    "wind":            {"ylabel": "km/h",            "title": "Wind Speed",        "color": "#38a169"},
    "humidity":        {"ylabel": "%",               "title": "Humidity",          "color": "#805ad5"},
}


@st.cache_resource
def get_forecast():
    path = NC_PATH or find_latest_nc_file(NC_DIR)
    if path is None:
        raise FileNotFoundError(
            f"No file matching 'ecmwf_aifs_india_YYYYMMDD_00z_merged.nc' found in '{NC_DIR}'."
        )
    return open_forecast(path)


@st.cache_resource
def get_geo_agent():
    return IndiaGeoAgent(GEONAMES_PATH)


@st.cache_resource
def get_district_data():
    gdf = load_district_gdf(DISTRICT_GEOJSON_PATH)
    ds  = get_forecast()
    id_grid = build_grid_district_index(ds, gdf)
    return gdf, id_grid


def build_multiday_chart(df, dates, variables, location_label):
    n = len(variables)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.8 * n), sharex=True)
    if n == 1:
        axes = [axes]
    x_labels = [day_label_for_date(d) for d in dates]
    field_map = {
        "rainfall": "tp_mm", "temperature_max": "tmax_c", "temperature_min": "tmin_c",
        "wind": "wind_kmh", "humidity": "humidity_pct",
    }
    for ax, var in zip(axes, variables):
        y = []
        for d in dates:
            stats = build_compact_day_summary(df, d)
            y.append(stats[field_map[var]] if stats else None)
        meta = PLOT_VAR_META[var]
        ax.plot(x_labels, y, marker="o", linewidth=2, color=meta["color"])
        ax.set_ylabel(meta["ylabel"])
        ax.set_title(meta["title"])
        ax.grid(alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    fig.suptitle(f"{len(dates)}-day outlook \u2014 {location_label.title()}")
    plt.tight_layout()
    return fig


def _result(text, table=None, figs=None):
    return {"text": text, "table": table, "figs": figs or []}


def geocode(location):
    return get_geo_agent().get_lat_lon(location)


def handle_city_forecast(location, date_term, range_days, plot, plot_variables):
    if not location:
        return _result("Please tell me which city, village, or landmark you'd like the forecast for.")
    coords = geocode(location)
    if coords is None:
        return _result(f"Could not find '{location}' in the places database.")
    lat, lon = coords
    try:
        df = get_point_timeseries(get_forecast(), lat, lon)
    except ValueError as e:
        return _result(str(e))
    header = f"**{location}** (lat {lat:.3f}, lon {lon:.3f})\n\n"
    if range_days:
        dates = get_date_range(now_ist().date(), range_days)
        table = build_multiday_table(df, dates)
        figs = []
        if plot:
            variables = expand_plot_variables(plot_variables)
            figs = [build_multiday_chart(df, dates, variables, location)]
        return _result(header + f"{range_days}-day outlook (IMD categories):", table=table, figs=figs)
    target_date = resolve_date_term(date_term) or now_ist().date()
    label = day_label_for_date(target_date)
    text = build_daily_forecast_text(label, df, target_date)
    if text is None:
        return _result(header + f"No forecast data available for {target_date}.")
    return _result(header + text)


def handle_state_summary(state_name, date_term):
    target_date = resolve_date_term(date_term) or now_ist().date()
    label = day_label_for_date(target_date)
    gdf, id_grid = get_district_data()
    stab = state_rainfall_table(get_forecast(), gdf, id_grid, target_date)
    if stab is None:
        return _result(f"No forecast data available for {target_date}.")
    row = stab[stab["state"].str.upper() == state_name.strip().upper()]
    if row.empty:
        return _result(f"No district data found for '{state_name}'.")
    dtab = district_rainfall_table(get_forecast(), gdf, id_grid, state_name, target_date)
    top = dtab.head(3)
    top_text = ", ".join(f"{r.district.title()} ({r.avg_rain_mm:.0f} mm)" for r in top.itertuples())
    avg_rain = row.iloc[0]["avg_rain_mm"]
    category = row.iloc[0]["category"]
    text = (
        f"**{state_name.title()}** \u2014 {label} ({target_date}): state-wide average is "
        f"**{category}** (~{avg_rain:.0f} mm). Highest expected rainfall: {top_text}."
    )
    return _result(text)


def handle_state_district_table(state_name, date_term):
    target_date = resolve_date_term(date_term) or now_ist().date()
    gdf, id_grid = get_district_data()
    table = district_full_table(get_forecast(), gdf, id_grid, target_date, state_name=state_name)
    if table is None or table.empty:
        return _result(f"No district data found for '{state_name}'.")
    summary = summarize_district_table(table, state_name.title(), target_date)
    display_table = table.drop(columns=["State", "Category"])
    return _result(summary, table=display_table)


def handle_district_query(state_name, date_term, threshold):
    threshold = (threshold or "heavy").lower()
    target_date = resolve_date_term(date_term) or now_ist().date()
    gdf, id_grid = get_district_data()
    table = district_rainfall_table(get_forecast(), gdf, id_grid, state_name, target_date)
    if table is None or table.empty:
        return _result(f"No district data found for '{state_name}'.")
    filtered = table[table["category"].apply(lambda c: category_at_least(c, threshold))]
    label = day_label_for_date(target_date)
    if filtered.empty:
        return _result(
            f"No districts in {state_name.title()} are expected to see "
            f"{threshold}-or-higher rainfall {label.lower()} ({target_date})."
        )
    lines = [
        f"Districts in **{state_name.title()}** expected to see **{threshold} or higher** "
        f"rainfall \u2014 {label} ({target_date}):\n"
    ]
    for r in filtered.itertuples():
        lines.append(f"- {r.district.title()}: {r.category} (~{r.avg_rain_mm:.0f} mm)")
    return _result("\n".join(lines))


def handle_state_threshold_query(date_term, threshold):
    threshold = (threshold or "moderate").lower()
    target_date = resolve_date_term(date_term) or now_ist().date()
    gdf, id_grid = get_district_data()
    table = state_rainfall_table(get_forecast(), gdf, id_grid, target_date)
    if table is None or table.empty:
        return _result("No forecast data available for that date.")
    filtered = table[table["category"].apply(lambda c: category_at_least(c, threshold))]
    label = day_label_for_date(target_date)
    if filtered.empty:
        return _result(f"No states are expected to see {threshold}-or-higher rainfall {label.lower()} ({target_date}).")
    lines = [f"States expected to see **{threshold} or higher** rainfall \u2014 {label} ({target_date}):\n"]
    for r in filtered.itertuples():
        lines.append(f"- {r.state}: {r.category} (~{r.avg_rain_mm:.0f} mm)")
    return _result("\n".join(lines))


def handle_region_plot(location, date_term, plot_variables, plot_style):
    target_date = resolve_date_term(date_term) or now_ist().date()
    gdf, id_grid = get_district_data()
    variables = expand_plot_variables(plot_variables)
    label = day_label_for_date(target_date)
    plot_fn = plot_region_spatial if plot_style == "spatial" else plot_region_pattern
    figs = []
    for var in variables:
        try:
            figs.append(plot_fn(get_forecast(), gdf, id_grid, location, target_date, variable=var))
        except ValueError as e:
            return _result(str(e))
    scope_label = "India" if is_india_scope(location) else location.title()
    style_word = "District-wise spatial" if plot_style == "spatial" else "Smoothed pattern"
    return _result(f"{style_word} map(s) for **{scope_label}** \u2014 {label} ({target_date})", figs=figs)


def route(parsed):
    intent        = parsed.get("intent")
    location      = parsed.get("location")
    date_term     = parsed.get("date")
    range_days    = parsed.get("range_days")
    threshold     = parsed.get("threshold")
    plot          = parsed.get("plot")
    plot_variables= parsed.get("plot_variables")
    plot_style    = parsed.get("plot_style")
    if intent == "city_forecast":
        return handle_city_forecast(location, date_term, range_days, plot, plot_variables)
    if intent == "state_summary" and location:
        return handle_state_summary(location, date_term)
    if intent == "state_district_table" and location:
        return handle_state_district_table(location, date_term)
    if intent == "district_rain_query" and location:
        return handle_district_query(location, date_term, threshold)
    if intent == "state_threshold_query":
        return handle_state_threshold_query(date_term, threshold)
    if intent == "region_plot" and location:
        return handle_region_plot(location, date_term, plot_variables, plot_style)
    return _result(
        "I couldn't understand that. Try:\n"
        "- 'weather in Pune' / 'rainfall forecast for Pune tomorrow'\n"
        "- 'rainfall for Delhi for the next 15 days'\n"
        "- 'weather of all districts in Maharashtra'\n"
        "- 'which districts of Maharashtra expect heavy rainfall'\n"
        "- 'which states expect moderate or higher rainfall tomorrow'\n"
        "- 'rainfall map of Maharashtra' / 'spatial plot of rainfall for India'"
    )


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("\U0001f326\ufe0f MEGHA-AI")
st.title("(Multi-model Ensemble for Geospatial & Hyperlocal Agentic Atmospheric Intelligence)")
st.caption("Design and Developed By - Ashish Alone")
st.caption("Guided By - Prof. Anoop Kumar Shukla, Dr D. R. Pattanaik & Prof. Gopal Nandan")
st.caption(
    "Ask things like *'weather in Pune'*, *'rainfall for Delhi next 15 days'*, "
    "*'weather of all districts in Maharashtra'*, or *'spatial plot of rainfall for India'*."
)

with st.sidebar:
    st.markdown("**Forecast source**")
    try:
        _ds = get_forecast()
        st.caption(f"Model init (UTC): {_ds.attrs.get('ic_utc', 'unknown')}")
        st.caption(f"Current IST time: {now_ist().strftime('%Y-%m-%d %H:%M')}")
    except Exception as e:
        st.error(f"Could not load NetCDF file: {e}")
    st.markdown("---")
    '''
    st.markdown("**Intent parser**")
    if GROQ_API_KEY.strip():
        st.success("Groq LLM (active)")
    else:
        st.info("Rule-based (no API key needed)\n\nOptionally add a free Groq key in app.py for smarter parsing.")
    '''

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("table") is not None:
            st.dataframe(msg["table"], use_container_width=True)
        for fig in msg.get("figs", []):
            st.pyplot(fig)

question = st.chat_input("Ask about weather forecast... (e.g. 'weather in Pune', 'rainfall map of Maharashtra')")

if question:
    st.session_state.messages.append({"role": "user", "content": question, "table": None, "figs": []})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            parsed = parse_intent(question)
            result = route(parsed)
        st.markdown(result["text"])
        if result.get("table") is not None:
            st.dataframe(result["table"], use_container_width=True)
        for fig in result.get("figs", []):
            st.pyplot(fig)
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["text"],
        "table": result.get("table"),
        "figs":  result.get("figs", []),
    })
