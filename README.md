
# <img width="200" alt="riot_logo" src="https://github.com/user-attachments/assets/15f6355c-4f61-45cf-9e7f-b453613acda2" /><br>
[![Paper DOI](https://img.shields.io/badge/Paper%20DOI-10.1145%2F3744256.3812556-green)](https://doi.org/10.1145/3744256.3812556) [![Dataset DOI](https://img.shields.io/badge/Dataset%20DOI-10.25835%2Fgqhuoyjd-blue)](https://doi.org/10.25835/gqhuoyjd) <br>
`RIoT` is a real-world IoT dataset for evaluating smart building systems in the wild, anonymized from proprietary BMS deployments and collected in collaboration with [`Innon Energy LTD`](https://www.innon.com). `RIoT`'s value proposition was presented at BuildSys '26 in the article titled [RIoT: An IoT Dataset for Robust Smart Building Systems in the Wild](https://dl.acm.org/doi/10.1145/3744256.3812556). The full dataset is available in this repository and is also archived at the [LUH Research Data Repository](https://doi.org/10.25835/gqhuoyjd) for long-term preservation.

## Role
Below, we showcase a schematic of the data collection process for `RIoT` (steps 1-2) and hint at its aim to facilitate evaluating building systems that posses 'smart modules' (steps 3-8), i.e., solutions that try to move away from passive monitoring toward providing insight about the building or ways to actuate (influence) its state.

<img width="99%" alt="datasetx_pipeline" src="https://github.com/user-attachments/assets/f4e44757-c034-4e86-8f54-4a6868b36597" />

## Dataset Overview

The dataset spans **6 anonymized residential sites** across Europe, with data collected from LoRaWAN-connected sensors via the [Project Haystack](https://project-haystack.org/) protocol. Each site contains daily JSON files with time-series telemetry for all installed sensor points.

| Site | Location | Sensors | Date Range | Total Days | Status |
|------|----------|---------|------------|------------|--------|
| **Site A** | Southern Europe | 44-68 | 2024-03-31 to 2026-07-07 | 598 | Active |
| **Site B** | Eastern Europe | 62-89 | 2024-03-31 to 2026-07-07 | 480 | Active |
| **Site C** | Northern Europe | 153 | 2024-03-31 to 2024-11-20 | 226 | Offline |
| **Site D** | Central Europe | 17 | 2024-07-01 to 2025-09-03 | 430 | Offline |
| **Site E** | Northern Europe | 15-20 | 2024-08-24 to 2026-07-07 | 683 | Active |
| **Site F** | Northern Europe | 165 | 2024-12-18 to 2025-09-01 | 258 | Offline |

### Sensor types

The dataset includes the following set of sensor modalities:

- **Air quality**: CO2 (ppm), TVOC (ppb), PM2.5 (ug/m3), PM10 (ug/m3), Formaldehyde (mg/m3), Ozone
- **Environmental**: Temperature (C), Humidity (%RH), Barometric Pressure (hPa), Light (lx)
- **Occupancy**: PIR motion sensors
- **Energy**: Current (A), Cumulative current (Ah), Active Power (W), Energy consumption (Wh), Voltage (V)
- **Weather**: Outdoor temperature, humidity, wind speed (m/s), wind direction (deg)
- **Odor**: Ammonia NH3 (ppm), Hydrogen sulfide H2S (ppm)
- **Safety**: Gas valve/relay status, Leak detection, Water level
- **HVAC**: TRV motor position, temperature setpoints, valve opening (Sites C, F)
- **Device health**: Battery level (%), sensor status

## Getting Started

### Data structure

```
RIoT/
  sites/
    site_A/
      2024-03-31.json
      2024-04-01.json
      ...
    site_B/
      ...
```

Each JSON file contains a list of sensor arrays for that day. Each sensor array wraps a [Project Haystack](https://project-haystack.org/doc/docHaystack/Json) grid with time-series rows.

### Loading data with Python / pandas

**Load a single day for one site:**

```python
import json
import pandas as pd

with open("RIoT/sites/site_A/2026-07-06.json") as f:
    day_data = json.load(f)

# Each element is a sensor's daily time series
print(f"Number of sensors: {len(day_data)}")

# Extract one sensor's data into a DataFrame
sensor = day_data[0][0]  # first sensor's grid
sensor_name = sensor["cols"][1]["meta"]["id"]["dis"]
unit = sensor["cols"][1]["meta"].get("unit", "")

rows = []
for row in sensor["rows"]:
    ts = row["ts"]["val"]
    val = row["v0"]
    if isinstance(val, dict):
        val = val["val"]
    rows.append({"timestamp": ts, "value": val})

df = pd.DataFrame(rows)
df["timestamp"] = pd.to_datetime(df["timestamp"])
print(f"\n{sensor_name} ({unit}):")
print(df.head())
```

**Load all sensors for a day into a single wide DataFrame:**

```python
import json
import pandas as pd

def load_day(filepath):
    """Load a daily JSON file and return a wide DataFrame with all sensors."""
    with open(filepath) as f:
        day_data = json.load(f)

    all_series = {}
    for sensor_arr in day_data:
        grid = sensor_arr[0]
        cols = grid.get("cols", [])
        if len(cols) < 2:
            continue

        meta = cols[1].get("meta", {})
        name = meta.get("id", {}).get("dis", "unknown")
        unit = meta.get("unit", "")

        for row in grid.get("rows", []):
            ts = pd.to_datetime(row["ts"]["val"])
            val = row["v0"]
            if isinstance(val, dict):
                val = val["val"]
            all_series.setdefault(name, []).append({"timestamp": ts, "value": val})

    dfs = []
    for name, records in all_series.items():
        s = pd.DataFrame(records).set_index("timestamp")["value"].rename(name)
        dfs.append(s)

    return pd.concat(dfs, axis=1).sort_index()

df = load_day("RIoT/sites/site_A/2026-07-06.json")
print(f"Shape: {df.shape}")
print(df.head())
```

**Load a date range for a specific sensor across multiple days:**

```python
import json
import pandas as pd
from pathlib import Path

def load_sensor_range(site_dir, sensor_name, start_date, end_date):
    """Load a specific sensor's data across a date range."""
    site = Path(site_dir)
    records = []

    for f in sorted(site.glob("*.json")):
        date_str = f.stem
        if date_str < start_date or date_str > end_date:
            continue

        with open(f) as fh:
            day_data = json.load(fh)

        for sensor_arr in day_data:
            grid = sensor_arr[0]
            cols = grid.get("cols", [])
            if len(cols) < 2:
                continue

            name = cols[1].get("meta", {}).get("id", {}).get("dis", "")
            if name != sensor_name:
                continue

            for row in grid.get("rows", []):
                val = row["v0"]
                if isinstance(val, dict):
                    val = val["val"]
                records.append({
                    "timestamp": pd.to_datetime(row["ts"]["val"]),
                    "value": val,
                })
            break

    df = pd.DataFrame(records).set_index("timestamp").sort_index()
    return df

# Example: load CO2 from Site A's living room for a week
df = load_sensor_range(
    "RIoT/sites/site_A",
    "Site A Living AQ CO2",
    "2026-07-01",
    "2026-07-07",
)
print(f"CO2 readings: {len(df)}")
print(df.describe())
```

**List all available sensors for a site:**

```python
import json
from pathlib import Path

def list_sensors(site_dir):
    """List all unique sensors available at a site (from the most recent file)."""
    site = Path(site_dir)
    latest_file = sorted(site.glob("*.json"))[-1]

    with open(latest_file) as f:
        day_data = json.load(f)

    sensors = []
    for sensor_arr in day_data:
        grid = sensor_arr[0]
        cols = grid.get("cols", [])
        if len(cols) < 2:
            continue
        meta = cols[1].get("meta", {})
        sensors.append({
            "name": meta.get("id", {}).get("dis", "unknown"),
            "unit": meta.get("unit", "-"),
            "kind": meta.get("kind", ""),
        })
    return sensors

for s in list_sensors("RIoT/sites/site_A"):
    print(f"  {s['name']:<45s} [{s['unit']}]")
```

## Citation
To cite this work, feel free to use the following BibTeX entry:
```python
@inproceedings{petrescu2026riot, 
  title={RIoT: An IoT Dataset for Robust Smart Building Systems in the Wild},
  author={Petrescu, Stefan and Rellermeyer, Jan S.},
  year={2026},
  booktitle = {Proceedings of the 13th ACM International Conference on Systems for Energy-Efficient Buildings, Cities, and Transportation},
  series = {BuildSys '26},
}
```
