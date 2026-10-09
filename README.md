# Activity-SG-CBR

Floor Plan Recommendation Prototype
: A two-part prototype that turns a person's daily activity schedule into floor plan recommendations.

1. **Digital Time Use Survey (TUS)** — a web survey where participants record a typical weekday
   (what they do, when, where, and with whom). Each response becomes an activity JSON file.
2. **Floor Plan Recommendation App** — a local web app that takes that JSON, generates candidate
   layouts with a shape grammar, and retrieves the most similar real floor plans from the ResPlan
   dataset using case-based reasoning (CBR).

## Files
| File | Description |
|---|---|
| `Digital_TUS.html` | Digital Time Use Survey (Step 1). Open in a browser; no installation needed |
| `prototype.py` | Floor plan recommendation app (Step 2) — backend + web UI in one file |
| `ResPlan.zip` | ResPlan floor plan database (stored with Git LFS). Unzip to get `ResPlan.pkl` |

> `ResPlan.zip` is stored with Git LFS. If you clone the repo and only see a small pointer file,
> install [Git LFS](https://git-lfs.com) and run `git lfs pull`.
> Alternatively, download the file directly from the GitHub web page.

## Step 1 — Digital Time Use Survey

`Digital_TUS.html` is a standalone page. Participants enter demographic and housing information,
then record every main activity from 4:00 AM to 4:00 AM the following day, each with:

- start / end time
- main activity (77 categories) and optional secondary activity
- room type where the activity ideally takes place
- number and type of people involved

Submitted responses are posted to a Google Apps Script endpoint set in `APPS_SCRIPT_URL` at the
top of the `<script>` block. Replace it with your own endpoint (and your own Prolific completion
code) if you deploy the survey yourself. Each response, saved as JSON, is the input for Step 2.

## Step 2 — Floor Plan Recommendation App

### How it works

1. **Input** — upload or paste the activity JSON from Step 1.
2. **Generate** — a shape grammar builds all valid room layouts that follow the activity sequence. You can filter them by which side of each room should stay open (N/S/E/W).
3. **Select** — click one generated layout to use as the query.
4. **Match** — the query is compared with every plan in the ResPlan DB on five features (room type, room size, adjacency, connection, direction) and the top 5 are shown. You can adjust the feature weights in the sidebar, rate each result (1–10), leave notes, and save the session as a JSON log.

### Requirements

- Python 3.9 or later
- Packages:

```
pip install flask numpy matplotlib shapely networkx
```

### How to run

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
5. Upload a JSON file from Step 1 (or click **Load sample**), then **Parse & Generate**.

You can also skip `FLOORPLAN_DB` and connect the database later from the **DB Connection** box in the sidebar by entering the path to `ResPlan.pkl`.

Press `Ctrl+C` in the terminal to stop the server.

### Input JSON format

This is the structure produced by the survey in Step 1 (only the fields used by the app are shown):

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

- `roomType`: one of `bedroom1`–`bedroom6`, `bathroom1`–`bathroom3`, `living_room`, `kitchen_dining`, `storage_balcony`, `garage`, `garden`, `front_door`, or `outside` (skipped).
- `peopleCount`: number or `"6_or_more"` — determines room size.

### Session log

Clicking **Finish** saves `session_log_<prolificId>_<timestamp>.json` to the directory set in
**Log Save Directory** (sidebar). It records the query plan, weights, and your ratings and notes for each result.
