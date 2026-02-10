from __future__ import annotations

import os, json, re
from dataclasses import dataclass
from datetime import date, timedelta, datetime, time
from typing import List, Dict, Optional, Any, Tuple

import numpy as np
import pandas as pd


DATA_DIR = "../../../DatasetX/sites/site_F"
OUT_DIR  = "./outputs"
os.makedirs(OUT_DIR, exist_ok=True)

START_DATE = date(2025, 1, 1)
END_DATE   = date(2025, 4, 30)

WORK_START = time(9, 0)
WORK_END   = time(17, 0)
WORK_DOWS  = {0,1,2,3,4}

VOLTAGE_V = 230.0
PHASES    = 1

RESAMPLE_MIN = 1

BASELINE_WINDOW_MIN = 180
BASELINE_Q = 0.10 #10th percentile as baseline proxy

ABOVE_BASELINE_PPM = 75.0
RISE_THRESH_PPM_30MIN = 40.0

MIN_ON_MIN  = 10
HOLD_OFF_MIN = 20

AUTO_PICK_TOP_N_HVAC_CTS = 3

HVAC_CTS_EXPLICIT: List[str] = []

#parsing
def daterange(d0: date, d1: date):
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)

def load_grids(path: str) -> List[dict]:
    with open(path, "r") as f:
        data = json.load(f)

    grids: List[dict] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, list):
                grids.extend([g for g in item if isinstance(g, dict)])
            elif isinstance(item, dict):
                grids.append(item)
    elif isinstance(data, dict):
        grids = [data]
    return grids

def extract_series_from_grid(grid: dict) -> List[dict]:
    cols = grid.get("cols", [])
    rows = grid.get("rows", [])
    if not cols or not rows:
        return []

    # timestamps
    ts_vals = []
    for r in rows:
        t = r.get("ts")
        if isinstance(t, dict) and t.get("_kind") == "dateTime":
            ts_vals.append(t.get("val"))
        else:
            ts_vals.append(None)
    ts = pd.to_datetime(ts_vals, utc=True, errors="coerce")
    if ts.isna().all():
        return []

    out = []
    for c in cols:
        name = c.get("name")
        if not name or name == "ts":
            continue

        meta = c.get("meta", {}) if isinstance(c.get("meta", {}), dict) else {}
        point_dis = (meta.get("id", {}) or {}).get("dis")
        equip_dis = (meta.get("equipRef", {}) or {}).get("dis")
        room_dis  = (meta.get("roomRef", {}) or {}).get("dis")
        unit      = meta.get("unit")

        if not point_dis:
            continue

        vals = []
        for r in rows:
            vobj = r.get(name)
            if isinstance(vobj, dict) and vobj.get("_kind") == "number":
                vals.append(vobj.get("val"))
            else:
                vals.append(np.nan)

        s = pd.Series(vals, index=ts, dtype="float64").sort_index()
        s = s[~s.index.isna()]
        out.append({
            "point_dis": point_dis,
            "equip_dis": equip_dis,
            "room_dis": room_dis,
            "unit": unit,
            "series": s,
        })
    return out


def work_mask(index_utc: pd.DatetimeIndex) -> np.ndarray:
    idx = index_utc.tz_convert("UTC")
    dow_ok = np.isin(idx.dayofweek, list(WORK_DOWS))
    t = idx.time
    hours_ok = np.array([(ti >= WORK_START and ti < WORK_END) for ti in t], dtype=bool)
    return dow_ok & hours_ok

def energy_kwh_from_ct(series: pd.Series, unit: Optional[str]) -> float:
    s = series.dropna().sort_index()
    if s.empty:
        return 0.0
    u = (unit or "").strip().lower()

    if u == "ah":
        diffs = s.diff()
        ah = diffs[diffs > 0].sum()
        return float((ah * VOLTAGE_V * PHASES) / 1000.0)

    if u == "a":
        t = s.index.view("int64") / 1e9
        i = s.values
        a_s = np.trapz(i, t)
        ah = a_s / 3600.0
        return float((ah * VOLTAGE_V * PHASES) / 1000.0)

    return 0.0


def rolling_quantile(x: pd.Series, window: int, q: float) -> pd.Series:
    return x.rolling(window=window, min_periods=max(10, window//10)).quantile(q)

def co2_occupancy_proxy(co2_utc: pd.Series) -> pd.Series:
    s = co2_utc.dropna().sort_index()
    if s.empty:
        return pd.Series(dtype=bool)

    s1 = s.resample(f"{RESAMPLE_MIN}min").median().interpolate(limit=5)

    w_base = int(BASELINE_WINDOW_MIN / RESAMPLE_MIN)
    baseline = rolling_quantile(s1, w_base, BASELINE_Q)

    above = s1 > (baseline + ABOVE_BASELINE_PPM)

    w_rise = int(30 / RESAMPLE_MIN)
    rise = (s1 - s1.shift(w_rise)) >= RISE_THRESH_PPM_30MIN

    raw_on = (above | rise).fillna(False)

    k_on = int(MIN_ON_MIN / RESAMPLE_MIN)
    on = (raw_on.rolling(k_on, min_periods=k_on).sum() >= k_on)

    hold = int(HOLD_OFF_MIN / RESAMPLE_MIN)
    if hold > 0:
        on_int = on.astype(int)
        held = pd.Series(
            np.maximum.accumulate(on_int[::-1].rolling(hold, min_periods=1).max())[::-1].values,
            index=on.index
        ).astype(bool)
        return held.rename("occupied")
    return on.rename("occupied")


@dataclass
class DayResult:
    day: date
    ct_kwh: Dict[str, float]
    room_occ_min: Dict[str, int]
    room_co2_stats: Dict[str, Tuple[float,float,float]]

def run_day(path: str, day: date) -> DayResult:
    grids = load_grids(path)

    all_series = []
    for g in grids:
        all_series.extend(extract_series_from_grid(g))

    co2_by_room: Dict[str, pd.Series] = {}
    co2_unit_by_room: Dict[str, str] = {}

    ct_points: List[Tuple[str,str,Optional[str],pd.Series]] = []

    for it in all_series:
        point = it["point_dis"] or ""
        equip = it.get("equip_dis") or ""
        room  = it.get("room_dis") or ""
        unit  = it.get("unit")
        s     = it["series"]

        if re.search(r"\bco2\b", point, re.IGNORECASE):
            if room:
                if room in co2_by_room:
                    co2_by_room[room] = pd.concat([co2_by_room[room], s], axis=1).median(axis=1)
                else:
                    co2_by_room[room] = s
                    co2_unit_by_room[room] = unit or ""

        if ("CT" in point) and re.search(r"\bTotal Current\b", point, re.IGNORECASE):
            ct_points.append((equip or point, point, unit, s))

    room_occ_min: Dict[str, int] = {}
    room_co2_stats: Dict[str, Tuple[float,float,float]] = {}

    for room, co2s in co2_by_room.items():
        occ = co2_occupancy_proxy(co2s)
        if occ.empty:
            continue

        wm = work_mask(occ.index)
        occ_w = occ[wm]
        room_occ_min[room] = int(occ_w.sum())

        c1 = co2s.resample(f"{RESAMPLE_MIN}min").median().interpolate(limit=5)
        c1w = c1[work_mask(c1.index)]
        if not c1w.empty:
            room_co2_stats[room] = (float(c1w.min()), float(c1w.median()), float(c1w.max()))

    ct_kwh: Dict[str, float] = {}
    for ct_equip, _point, unit, s in ct_points:
        sw = s[work_mask(s.index)]
        kwh = energy_kwh_from_ct(sw, unit)
        ct_kwh[ct_equip] = ct_kwh.get(ct_equip, 0.0) + kwh

    return DayResult(day=day, ct_kwh=ct_kwh, room_occ_min=room_occ_min, room_co2_stats=room_co2_stats)


def main():
    daily_ct_rows = []
    daily_occ_rows = []
    daily_co2debug_rows = []

    for d in daterange(START_DATE, END_DATE):
        path = os.path.join(DATA_DIR, f"{d.isoformat()}.json")
        if not os.path.exists(path):
            print(f"missing {path}")
            continue

        res = run_day(path, d)

        for ct, kwh in res.ct_kwh.items():
            daily_ct_rows.append({"day": d.isoformat(), "ct_equip": ct, "kwh_workhours": kwh})

        for room, mins in res.room_occ_min.items():
            daily_occ_rows.append({
                "day": d.isoformat(),
                "room": room,
                "occupied_minutes_workhours": mins,
                "occupied_hours_workhours": mins / 60.0
            })

        for room, (mn, med, mx) in res.room_co2_stats.items():
            daily_co2debug_rows.append({
                "day": d.isoformat(),
                "room": room,
                "co2_min_workhours": mn,
                "co2_median_workhours": med,
                "co2_max_workhours": mx,
            })

        print(f"[OK] {d.isoformat()}  CTs:{len(res.ct_kwh)}  CO2rooms:{len(res.room_occ_min)}")

    df_ct = pd.DataFrame(daily_ct_rows)
    df_occ = pd.DataFrame(daily_occ_rows)
    df_dbg = pd.DataFrame(daily_co2debug_rows)

    # save raw
    p_ct  = os.path.join(OUT_DIR, "daily_ct_kwh_workhours.csv")
    p_occ = os.path.join(OUT_DIR, "daily_room_occ_minutes.csv")
    p_dbg = os.path.join(OUT_DIR, "daily_room_co2_debug.csv")

    df_ct.to_csv(p_ct, index=False)
    df_occ.to_csv(p_occ, index=False)
    df_dbg.to_csv(p_dbg, index=False)

    if not df_ct.empty:
        df_ct_sum = df_ct.groupby("ct_equip", as_index=False)["kwh_workhours"].sum().sort_values("kwh_workhours", ascending=False)
        df_ct_sum.to_csv(os.path.join(OUT_DIR, "sum_ct_kwh_workhours.csv"), index=False)

    if not df_occ.empty:
        df_occ_sum = df_occ.groupby("room", as_index=False)["occupied_minutes_workhours"].sum().sort_values("occupied_minutes_workhours", ascending=False)
        df_occ_sum.to_csv(os.path.join(OUT_DIR, "sum_room_occ_minutes.csv"), index=False)

    if HVAC_CTS_EXPLICIT:
        hvac_cts = HVAC_CTS_EXPLICIT
    else:
        df_ct_sum = pd.read_csv(os.path.join(OUT_DIR, "sum_ct_kwh_workhours.csv"))
        hvac_cts = df_ct_sum["ct_equip"].head(AUTO_PICK_TOP_N_HVAC_CTS).tolist()

    print("\nHVAC CT circuits are:", hvac_cts)

    df_ct["day"] = pd.to_datetime(df_ct["day"])
    df_occ["day"] = pd.to_datetime(df_occ["day"])

    hvac_daily = (
        df_ct[df_ct["ct_equip"].isin(hvac_cts)]
        .groupby("day", as_index=False)["kwh_workhours"].sum()
        .rename(columns={"kwh_workhours": "hvac_kwh_baseline"})
    )

    if df_occ.empty:
        hvac_daily["occ_fraction"] = 0.0
    else:
        occ_piv = df_occ.pivot_table(index="day", columns="room", values="occupied_minutes_workhours", aggfunc="sum").fillna(0)
        total_work_min = int((datetime.combine(date.today(), WORK_END) - datetime.combine(date.today(), WORK_START)).seconds / 60)

        occ_minutes = occ_piv.max(axis=1)
        occ_fraction = (occ_minutes / total_work_min).clip(0, 1)
        occ_fraction = occ_fraction.rename("occ_fraction").reset_index()

        hvac_daily = pd.merge(hvac_daily, occ_fraction, on="day", how="left").fillna({"occ_fraction": 0.0})

    hvac_daily["hvac_kwh_proxy"] = hvac_daily["hvac_kwh_baseline"] * hvac_daily["occ_fraction"]
    hvac_daily["hvac_kwh_savings"] = hvac_daily["hvac_kwh_baseline"] - hvac_daily["hvac_kwh_proxy"]

    hvac_daily.to_csv(os.path.join(OUT_DIR, "daily_hvac_baseline_proxy_savings.csv"), index=False)

    totals = hvac_daily[["hvac_kwh_baseline", "hvac_kwh_proxy", "hvac_kwh_savings"]].sum()
    totals.to_csv(os.path.join(OUT_DIR, "total_hvac_baseline_proxy_savings.csv"), header=False)

    print("\nSaved:")
    print(" ", p_ct)
    print(" ", p_occ)
    print(" ", p_dbg)
    print(" ", os.path.join(OUT_DIR, "daily_hvac_baseline_proxy_savings.csv"))
    print(" ", os.path.join(OUT_DIR, "total_hvac_baseline_proxy_savings.csv"))
    print("\nTotals over period:")
    print(totals)


if __name__ == "__main__":
    main()
