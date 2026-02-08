import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import re

FONT_SIZE = 16
DATA_DIR = Path("../../data/use_case_1")

START_DATE = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
END_DATE   = START_DATE + pd.Timedelta(days=14)

N_MEASUREMENTS = 3000

ALL_SENSORS = [1, 2, 3, 4, 5]

SENSOR_COLORS = {
    1: "#325754",
    2: "#509994",
    3: "#95a9a6",
    4: "#d2a6aa",
    5: "#f7a8ac",
}

MODALITY_LABELS = {
    "temp": ("Temperature (°C)", "Ex. temp. variation (Site B indoor)"),
    "co2": ("CO2 (ppm)", "Ex. CO2 variation (Site B indoor)"),
    "hum": ("Humidity (%)", "Ex. humidity variation (Site B indoor)"),
    "tvoc": ("TVOC", "Ex. TVOC variation (Site B indoor)"),
    "pir": ("PIR signal", "Ex. PIR signal variation (Site B indoor)")
}

def parse_value(raw: str):
    raw = raw.strip()

    if raw.lower() in {"true", "t"}:
        return 1
    if raw.lower() in {"false", "f"}:
        return 0

    num = re.findall(r"[-+]?\d*\.?\d+", raw)
    return float(num[0]) if num else None

def load_sensor_file(path: Path):
    sensor_id = int(path.name.split("_")[0])
    modality = path.stem.split("_")[-1]

    rows = []
    count = 0

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines[2:]:
        if count >= N_MEASUREMENTS:
            break

        if "," not in line:
            continue

        ts_raw, val_raw = line.split(",", 1)

        try:
            ts = pd.to_datetime(ts_raw.replace(" Bucharest", ""), utc=True)
            val = parse_value(val_raw)
        except Exception:
            continue

        # Only keep values in the first 2 weeks of 2025 for snapshot
        if ts < START_DATE or ts > END_DATE:
            continue

        if modality == "co2" and val is not None and val > 10000:
            continue

        if val is not None:
            rows.append((ts, val))
            count += 1

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["dateTime", "value"])
    df["sensor_id"] = sensor_id
    df["modality"] = modality
    return df

dfs = []
for file in DATA_DIR.glob("*_sensor_*.txt"):
    df = load_sensor_file(file)
    if df is not None:
        dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)


for modality, (ylabel, title) in MODALITY_LABELS.items():

    df_m = all_df[all_df["modality"] == modality]
    if df_m.empty:
        continue

    plt.figure(figsize=(6, 2))

    if modality == "pir":
        full_time = pd.date_range(
            start=START_DATE,
            end=END_DATE,
            freq="1h",
            tz="UTC"
        )

        for sensor_id in ALL_SENSORS:

            g = df_m[df_m["sensor_id"] == sensor_id].copy()

            g = g.set_index("dateTime").reindex(full_time)

            g["value"] = g["value"].fillna(0)

            plt.plot(
                full_time,
                g["value"],
                label=f"S. {sensor_id}",
                color=SENSOR_COLORS.get(sensor_id),
                linewidth=2,
            )


    else:
        for sensor_id, g in df_m.groupby("sensor_id"):
            plt.plot(
                g["dateTime"],
                g["value"],
                label=f"S. {sensor_id}",
                color=SENSOR_COLORS.get(sensor_id),
                linewidth=2,
            )

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 5, 9, 13]))


    plt.xlabel("Time", fontsize=FONT_SIZE - 5)
    plt.ylabel(ylabel, fontsize=FONT_SIZE - 3)
    plt.title(title, fontsize=FONT_SIZE - 2)

    legend = plt.legend(
        title="Sensor",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        edgecolor="black",
        fontsize=FONT_SIZE - 5,
        title_fontsize=FONT_SIZE - 5,
    )
    legend.get_frame().set_alpha(0.5)

    plt.tight_layout()
    plt.grid(False)

    plt.savefig(f"{modality}.pdf", dpi=300, bbox_inches="tight")
