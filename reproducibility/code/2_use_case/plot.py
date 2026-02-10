import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = "./outputs"
FIG_DIR = os.path.join(OUT_DIR, "figs")
os.makedirs(FIG_DIR, exist_ok=True)

P_CT_DAILY   = os.path.join(OUT_DIR, "daily_ct_kwh_workhours.csv")
P_OCC_DAILY  = os.path.join(OUT_DIR, "daily_room_occ_minutes.csv")
P_CO2_DBG    = os.path.join(OUT_DIR, "daily_room_co2_debug.csv")
P_HVAC_DAILY = os.path.join(OUT_DIR, "daily_hvac_baseline_proxy_savings.csv")

SAVE_FIGS = True

# current CO2-capable set (selected subset)
CURRENT_CO2_MEETING_ROOMS = 1  # Conference Room
CURRENT_CO2_OFFICE_ROOMS  = 1  # Technical Office
CURRENT_CO2_TOTAL_ROOMS   = 8  # CO2 rooms total

def savefig(name: str):
    if SAVE_FIGS:
        path = os.path.join(FIG_DIR, name)
        plt.savefig(path, dpi=220, bbox_inches="tight")
        print("Saved", path)


def load():
    df_ct = pd.read_csv(P_CT_DAILY)
    df_occ = pd.read_csv(P_OCC_DAILY) if os.path.exists(P_OCC_DAILY) else pd.DataFrame()
    df_dbg = pd.read_csv(P_CO2_DBG) if os.path.exists(P_CO2_DBG) else pd.DataFrame()
    df_hvac = pd.read_csv(P_HVAC_DAILY) if os.path.exists(P_HVAC_DAILY) else pd.DataFrame()

    df_ct["day"] = pd.to_datetime(df_ct["day"])
    if not df_occ.empty:
        df_occ["day"] = pd.to_datetime(df_occ["day"])
    if not df_dbg.empty:
        df_dbg["day"] = pd.to_datetime(df_dbg["day"])
    if not df_hvac.empty:
        df_hvac["day"] = pd.to_datetime(df_hvac["day"])
    return df_ct, df_occ, df_dbg, df_hvac

def plot_total_ct_kwh(df_ct: pd.DataFrame):
    daily_total = df_ct.groupby("day", as_index=False)["kwh_workhours"].sum()

    plt.figure()
    plt.plot(daily_total["day"], daily_total["kwh_workhours"])
    plt.title("Total CT energy during workhours (kWh/day)")
    plt.xlabel("Day")
    plt.ylabel("kWh (workhours)")
    plt.xticks(rotation=45)
    plt.grid(True, axis="y")
    savefig("01_ct_total_kwh_per_day.png")
    plt.show()


def plot_top_ct_lines(df_ct: pd.DataFrame, topn: int = 3):
    ct_sum = df_ct.groupby("ct_equip", as_index=False)["kwh_workhours"].sum()
    top_cts = ct_sum.sort_values("kwh_workhours", ascending=False)["ct_equip"].head(topn).tolist()

    piv = df_ct[df_ct["ct_equip"].isin(top_cts)].pivot_table(
        index="day", columns="ct_equip", values="kwh_workhours", aggfunc="sum"
    ).fillna(0)

    plt.figure()
    for c in piv.columns:
        plt.plot(piv.index, piv[c], label=c)
    plt.title(f"Top {topn} CT circuits: kWh/day during workhours")
    plt.xlabel("Day")
    plt.ylabel("kWh (workhours)")
    plt.xticks(rotation=45)
    plt.grid(True, axis="y")
    plt.legend()
    savefig("02_ct_top_circuits_lines.png")
    plt.show()

def plot_occupancy_aggregate(df_occ: pd.DataFrame):
    # Conservative: max occupied minutes across CO2 rooms each day
    occ_piv = df_occ.pivot_table(
        index="day",
        columns="room",
        values="occupied_minutes_workhours",
        aggfunc="sum"
    ).fillna(0)
    occ_max_hours = (occ_piv.max(axis=1) / 60.0).rename("occ_max_hours")
    occ_sum_hours = (occ_piv.sum(axis=1) / 60.0).rename("occ_sum_hours")

    plt.figure()
    plt.plot(occ_max_hours.index, occ_max_hours.values, label="Max occupied hours (across CO2 rooms)")
    plt.plot(occ_sum_hours.index, occ_sum_hours.values, label="Sum occupied hours (across CO2 rooms)")
    plt.title("CO₂-proxy occupancy during workhours (aggregate)")
    plt.xlabel("Day")
    plt.ylabel("Hours")
    plt.xticks(rotation=45)
    plt.grid(True, axis="y")
    plt.legend()
    savefig("03_occ_aggregate.png")
    plt.show()

def plot_occupancy_by_room(df_occ: pd.DataFrame, max_rooms: int = 8):
    if df_occ.empty:
        return
    room_sum = df_occ.groupby("room")["occupied_minutes_workhours"].sum().sort_values(ascending=False)
    rooms = room_sum.head(max_rooms).index.tolist()

    piv = df_occ[df_occ["room"].isin(rooms)].pivot_table(
        index="day", columns="room", values="occupied_hours_workhours", aggfunc="sum"
    ).fillna(0)

    plt.figure()
    for r in piv.columns:
        plt.plot(piv.index, piv[r], label=r)
    plt.title(f"CO₂-proxy occupied hours per room (top {len(rooms)} rooms)")
    plt.xlabel("Day")
    plt.ylabel("Occupied hours (0–8)")
    plt.ylim(0, 8.2)
    plt.xticks(rotation=45)
    plt.grid(True, axis="y")
    plt.legend()
    savefig("04_occ_by_room.png")
    plt.show()


def plot_hvac_baseline_proxy_savings(df_hvac: pd.DataFrame):
    plt.figure()
    plt.plot(df_hvac["day"], df_hvac["hvac_kwh_baseline"], label="Baseline HVAC kWh (workhours)")
    plt.plot(df_hvac["day"], df_hvac["hvac_kwh_proxy"], label="Proxy HVAC kWh (CO₂ occupancy)")
    plt.plot(df_hvac["day"], df_hvac["hvac_kwh_savings"], label="Potential savings (baseline - proxy)")
    plt.title("HVAC energy: baseline vs CO₂-proxy vs potential savings (per day)")
    plt.xlabel("Day")
    plt.ylabel("kWh (workhours)")
    plt.xticks(rotation=45)
    plt.grid(True, axis="y")
    plt.legend()
    savefig("05_hvac_baseline_proxy_savings.png")
    plt.show()

def plot_combined(df_hvac: pd.DataFrame, df_occ: pd.DataFrame):
    if df_hvac.empty or df_occ.empty:
        print("[INFO] Need both hvac + occupancy for combined plot; skipping.")
        return

    occ_piv = df_occ.pivot_table(
        index="day",
        columns="room",
        values="occupied_minutes_workhours",
        aggfunc="sum"
    ).fillna(0)
    occ_max_hours = (occ_piv.max(axis=1) / 60.0).rename("occ_max_hours").reset_index()

    m = pd.merge(df_hvac, occ_max_hours, on="day", how="left").fillna({"occ_max_hours": 0.0})

    scale = m["hvac_kwh_baseline"].max() / max(1e-6, m["occ_max_hours"].max())
    occ_scaled = m["occ_max_hours"] * scale

    plt.figure()
    plt.plot(m["day"], m["hvac_kwh_baseline"], label="Baseline HVAC kWh")
    plt.plot(m["day"], m["hvac_kwh_proxy"], label="Proxy HVAC kWh")
    plt.plot(m["day"], m["hvac_kwh_savings"], label="Savings kWh")
    plt.plot(m["day"], occ_scaled, label=f"Occupancy (max hours) × {scale:.1f} (scaled)")

    plt.title("Combined: HVAC baseline/proxy/savings + CO₂ occupancy (scaled)")
    plt.xlabel("Day")
    plt.ylabel("kWh (workhours) / scaled occupancy")
    plt.xticks(rotation=45)
    plt.grid(True, axis="y")
    plt.legend()
    savefig("06_combined_consumption_occupancy_savings.png")
    plt.show()

# We do a per-room normalization scaling to compare against our current instrumented room subset.
REFERENCE_SITE_TOTALS_KWH = {
    "Meeting": {"rooms": 5,  "always_on": 1920,  "proxy": 384},
    "Offices": {"rooms": 19, "always_on": 20900, "proxy": 8800},
    "Total":   {"rooms": 24, "always_on": 22820, "proxy": 9200},
}

def scale_reference_totals_to_current_rooms(
    current_meeting: int,
    current_office: int,
    current_total: int,
):
    def scale(rowtype: str, n_rooms_current: int):
        base_rooms = REFERENCE_SITE_TOTALS_KWH[rowtype]["rooms"]
        factor = (n_rooms_current / base_rooms) if base_rooms > 0 else 0.0
        always = REFERENCE_SITE_TOTALS_KWH[rowtype]["always_on"] * factor
        proxy  = REFERENCE_SITE_TOTALS_KWH[rowtype]["proxy"] * factor
        return always, proxy

    meet_always, meet_proxy = scale("Meeting", current_meeting)
    off_always,  off_proxy  = scale("Offices", current_office)
    tot_always,  tot_proxy  = scale("Total", current_total)

    df = pd.DataFrame({
        "Type": ["Meeting", "Offices", "Total"],
        "#rooms (reference site)": [
            REFERENCE_SITE_TOTALS_KWH["Meeting"]["rooms"],
            REFERENCE_SITE_TOTALS_KWH["Offices"]["rooms"],
            REFERENCE_SITE_TOTALS_KWH["Total"]["rooms"],
        ],
        "#rooms (current instrumented)": [current_meeting, current_office, current_total],
        "Always-on (scaled kWh)": [meet_always, off_always, tot_always],
        "Proxy (scaled kWh)": [meet_proxy, off_proxy, tot_proxy],
    })
    df["Savings (scaled kWh)"] = df["Always-on (scaled kWh)"] - df["Proxy (scaled kWh)"]
    return df


def plot_reference_scaled_bars(df_scaled: pd.DataFrame):
    labels = df_scaled["Type"].tolist()
    always = df_scaled["Always-on (scaled kWh)"].tolist()
    proxy  = df_scaled["Proxy (scaled kWh)"].tolist()

    x = range(len(labels))
    width = 0.4

    plt.figure()
    plt.bar([i - width/2 for i in x], always, width=width, label="Always-on (room-scaled from reference site)")
    plt.bar([i + width/2 for i in x], proxy,  width=width, label="Proxy (room-scaled from reference site)")
    plt.title("Reference site totals scaled to current room counts (linear per-room normalization)")
    plt.ylabel("kWh (scaled)")
    plt.xticks(list(x), labels)
    plt.grid(True, axis="y")
    plt.legend()
    savefig("07_reference_scaled_bars.png")
    plt.show()


def plot_compare_measured_vs_reference_scaled(df_hvac: pd.DataFrame, df_scaled: pd.DataFrame):
    measured_baseline = float(df_hvac["hvac_kwh_baseline"].sum())
    measured_proxy    = float(df_hvac["hvac_kwh_proxy"].sum())

    # Use "Total" row of room-scaled reference for comparison
    ref_total = df_scaled[df_scaled["Type"] == "Total"].iloc[0]
    ref_baseline = float(ref_total["Always-on (scaled kWh)"])
    ref_proxy    = float(ref_total["Proxy (scaled kWh)"])

    labels = ["Measured baseline", "Measured proxy", "Reference-scaled baseline", "Reference-scaled proxy"]
    vals   = [measured_baseline, measured_proxy, ref_baseline, ref_proxy]

    plt.figure()
    plt.bar(range(len(labels)), vals)
    plt.title("Measured HVAC totals vs reference-scaled totals (note: different date ranges)")
    plt.ylabel("kWh")
    plt.xticks(range(len(labels)), labels, rotation=20, ha="right")
    plt.grid(True, axis="y")
    savefig("08_measured_vs_reference_scaled.png")
    plt.show()

def main():
    df_ct, df_occ, df_dbg, df_hvac = load()

    plot_total_ct_kwh(df_ct)
    plot_top_ct_lines(df_ct, topn=3)

    plot_occupancy_aggregate(df_occ)
    plot_occupancy_by_room(df_occ, max_rooms=8)

    plot_hvac_baseline_proxy_savings(df_hvac)

    plot_combined(df_hvac, df_occ)

    df_scaled = scale_reference_totals_to_current_rooms(
        current_meeting=CURRENT_CO2_MEETING_ROOMS,
        current_office=CURRENT_CO2_OFFICE_ROOMS,
        current_total=CURRENT_CO2_TOTAL_ROOMS,
    )
    df_scaled.to_csv(os.path.join(FIG_DIR, "reference_scaled_table.csv"), index=False)
    print("\nSaved scaled reference table:", os.path.join(FIG_DIR, "reference_scaled_table.csv"))
    print(df_scaled)

    plot_reference_scaled_bars(df_scaled)
    plot_compare_measured_vs_reference_scaled(df_hvac, df_scaled)

    print(f"\nAll figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
