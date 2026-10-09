# Activity-SG-CBR

Floor Plan Recommendation Prototype

A local web app that turns a daily activity schedule into floor plan recommendations.
It generates candidate layouts with a shape grammar, lets you pick one as a query, and
retrieves the most similar real floor plans from the ResPlan dataset using case-based reasoning (CBR).

## How it works
1. **Input** — upload or paste an activity JSON (what you do, where, when, with how many people).
2. **Generate** — a shape grammar builds all valid room layouts that follow the activity sequence. You can filter them by which side of each room should stay open (N/S/E/W).
3. **Select** — click one generated layout to use as the query.
4. **Match** — the query is compared with every plan in the ResPlan DB on five features (room type, room size, adjacency, connection, direction) and the top 5 are shown. You can adjust the feature weights in the sidebar, rate each result (1–10), leave notes, and save the session as a JSON log.

## Requirements
- Python 3.9 or later
- Packages:
```
pip install flask numpy matplotlib shapely networkx
```
## Files
| File | Description |
|---|---|
| `prototype.py` | The app (backend + web UI in one file) |
| `ResPlan.zip` | ResPlan floor plan database (stored with Git LFS). Unzip to get `ResPlan.pkl` |

> `ResPlan.zip` is stored with Git LFS. If you clone the repo and only see a small pointer file,
> install [Git LFS](https://git-lfs.com) and run `git lfs pull`.
> Alternatively, download the file directly from the GitHub web page.

## How to run
1. Download `prototype.py` and `ResPlan.zip` from this repository.
2. Unzip `ResPlan.zip` → you get `ResPlan.pkl`.
3. Open a terminal and run (replace the paths with where you saved the files):
   ```
   FLOORPLAN_DB=~/Downloads/ResPlan.pkl python ~/Downloads/prototype.py
   ```
   On Windows (PowerShell):
   ```
   $env:FLOORPLAN_DB="C:\Users\you\Downloads\ResPlan.pkl"; python C:\Users\you\Downloads\prototype.py
   ```
4. A browser window opens automatically at **http://localhost:8888**. If it doesn't, open that address manually.
5. Click **Load sample** (or upload your own JSON), then **Parse & Generate**.

You can also skip `FLOORPLAN_DB` and connect the database later from the **DB Connection** box in the sidebar by entering the path to `ResPlan.pkl`.
Press `Ctrl+C` in the terminal to stop the server.

## Input JSON format

```json
{
  "prolificId": "test_user",
  "entries": [
    {
      "startTime": "08:00",
      "mainActivity": 1,
      "where": "living_room",
      "spatialPreferences": { "roomType": "living_room" },
      "peopleCount": 4
    }
  ]
}
```

- `where` / `roomType`: one of `bedroom1`–`bedroom4`, `bathroom1`–`bathroom3`, `living_room`, `kitchen`, `kitchen_dining`, `dining_room`, `storage_balcony`, `garden`, `front_door`, `garage`, or `outside` (skipped).
- `peopleCount`: number or `"6_or_more"` — determines room size.

## Session log
Clicking **Finish** saves `session_log_<prolificId>_<timestamp>.json` to the directory set in
**Log Save Directory** (sidebar). It records the query plan, weights, and your ratings and notes for each result.
