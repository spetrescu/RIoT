# DatasetX specifics
As mentioned in the paper, we use Haystack tagging to describe metadata for the measurements. Each data file (per day, per site) contains telemetry for all sensor points intstalled (CO2, TVOC, temperature, etc.). The point tagging enables interpreting the data in context — the actual hierarchy of devices, but also semantically w.r.t. what the measurement represents and how it is used. Particularly, each of the files contains the following global grid structure:
- `meta`: Grid-level metadata (version, display name, history start/end, timezone, history limit)
- `cols`: Definition of each returned column (e.g., 'ts' for timestamp and 'v0' for measurement value)
- `rows`: Time-stamped measurement values
For column definitions:
`ts`: timestamp of measurement, also includes timezone information
`v0`: represents the measured telemetry point

Below, we provide information about all tags used and their particular purpose:
1. `id`: unique reference to the point

2. `dis` / navName: human-friendly display name (e.g. "PM2.5")

3. `kind`: Data type (number)

4. `unit`: physical unit (e.g. temp in C)

5. `point`: marker indicating a point entity (required for any telemetry point)

6. `sensor`: marker indicating raw measured data (vs. a setpoint or command)

7. `roomRef` / `floorRef` / `siteRef`: location hierarchy for organizing data by physical spaces

8. `equipRef`: reference to the associated equipment entity

9. `lorawanPoint`: identifies LoRaWAN-connected sensor points

10. `lorawanCur`: network variable name

11. `his`: marker indicating history tracking is enabled

12. `hisStart` / `hisEnd`: date range of available historical data

13. `hisCollectInterval`: measurement sampling frequency

14. `curVal` / `curStatus` / `cur`: real-time data snapshot and status attributes

15. `amSensorRef`: reference to internal digital-twin sensor model

16. `amSensorModelId`: device type tag (sensor platform/channel)

