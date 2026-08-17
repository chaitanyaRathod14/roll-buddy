~# Roll Buddy — Face Recognition Attendance

Take class attendance by pointing a camera at the room (or uploading a class photo). Every face is detected, matched against a MongoDB-persisted roster of enrolled students, and reported as **present**, **unrecognized**, or **not seen**.

The project has two parts:

- **Backend** — a Flask microservice (`app.py`) built on [DeepFace](https://github.com/serengil/deepface) (Facenet embeddings, OpenCV/RetinaFace detectors) and MongoDB.
- **Frontend** — a single-file browser UI (`index.html`) that talks to the backend over HTTP, with live camera capture and drag-and-drop photo upload.

## Features

- **Enroll a student** — capture 3–5 reference photos via camera, or upload a single photo containing multiple students for bulk registration (each detected face is walked through the form one at a time).
- **Data augmentation at enrollment** — every accepted reference photo is expanded into 8 synthetic variants (flip, ±12° rotation, brightness/contrast jitter, zoom) and the resulting embeddings are averaged into one stored profile, making matching robust to lighting/angle.
- **Take attendance** — capture a class photo from the camera or upload one; every face is matched against the full roster stored in MongoDB.
- **Roster management** — list and delete enrolled students from the UI.
- **Persisted roster** — student profiles (name, PRN/GR, class code, embedding) live in MongoDB, so recognition always uses the current roster and clients can't spoof matches.
- **Defensive by design** — CORS for browser frontends, capped payload size, JSON error responses instead of stack-trace debug pages, and base64 decoding that fails gracefully.

## Architecture

```
index.html (browser UI)
     │  HTTP / JSON (base64 images)
     ▼
app.py (Flask microservice)
     │  DeepFace (Facenet embeddings, OpenCV/RetinaFace detection)
     ▼
MongoDB (students collection, keyed by PRN/GR number)
```

Key pipeline decisions:

- **PRN/GR is the unique key.** Many students share a class code (e.g. `TY-IT-G`), so the human-readable class code (`student_id`) is *not* unique.
- **Embeddings never leave the server.** `/detect_faces` returns only cropped thumbnails; `/recognize` fetches embeddings from MongoDB itself.
- **Different detectors per job.** OpenCV for registration (fast, fine for single well-lit reference faces) and RetinaFace for recognition (better on crowded class photos).

## Prerequisites

- Python 3.10+
- MongoDB running locally (default `mongodb://localhost:27017`)

## Setup

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv env
   .\env\Scripts\activate        # Windows
   # source env/bin/activate     # macOS/Linux
   pip install -r requirements.txt
   ```

2. Create a `.env` file (a `.env` is already present with defaults):

   ```env
   MONGO_URI=mongodb://localhost:27017
   MONGO_DB=attendance-sys
   ```

3. Start the backend:

   ```bash
   python app.py
   ```

   The service runs on `http://localhost:5001`. On first start it pre-loads the Facenet model, which takes a moment. You should see `Connected to MongoDB at ...` in the logs.

4. Open `index.html` in a browser (just double-click the file). It talks to the backend at `http://localhost:5001` (change `API_BASE` in the `<script>` block if your backend is elsewhere).

## Usage

1. **Enroll students** — switch to the *Enroll student* tab:
   - *Camera capture:* take 3–5 clear, front-facing shots of one student, fill in name / PRN / year / branch / division, and save.
   - *Upload photo (multiple faces):* upload a class or group photo, click **Detect faces**, then register each numbered face with its own details using **Save & next** or **Skip this face**.
2. **Take attendance** — switch to the *Take attendance* tab, capture or upload a class photo, and review the results:
   - **Present** — matched to an enrolled student (with confidence %).
   - **Unrecognized faces** — faces found but not in the roster.
   - **Not seen** — enrolled students who weren't matched.

## API

Base URL: `http://localhost:5001`

| Method | Route | Description |
| --- | --- | --- |
| GET | `/` | Service status message |
| GET | `/health` | Health check (`{"status": "ok"}`) |
| GET | `/students` | List all enrolled students (embeddings omitted) |
| DELETE | `/students/<prn_gr>` | Remove a student |
| POST | `/detect_faces` | Detect all faces in a photo, return numbered cropped thumbnails |
| POST | `/enroll` | Enroll (or update) a student from 1–5 reference photos |
| POST | `/recognize` | Match all faces in a class photo against the roster |

### `/enroll`

```json
{
  "prn_gr": "1252130014",
  "name": "Chaitanya Rathod",
  "year": "TY",
  "branch": "IT",
  "division": "G",
  "images": ["data:image/jpeg;base64,..."]
}
```

Returns the saved `prn_gr`, derived class code (`student_id`), number of photos used/skipped, and the count of embeddings averaged into the profile.

### `/recognize`

```json
{ "class_photo": "data:image/jpeg;base64,..." }
```

Returns matched students (with confidence), unmatched face locations, and the total face count.

### `/detect_faces`

```json
{ "photo": "data:image/jpeg;base64,..." }
```

Returns `{ "total_faces_detected", "faces": [{ "face_index", "face_location", "thumbnail" }] }`. Thumbnails are re-submitted to `/enroll` for embedding.

## Configuration

Environment variables (with defaults):

| Variable | Default | Purpose |
| --- | --- | --- |
| `MONGO_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `MONGO_DB` | — | Database name |
| `FRONTEND_ORIGIN` | `*` | Allowed CORS origin for the frontend |

Server-side constants in `app.py`:

- `MATCH_TOLERANCE = 0.40` — cosine-distance threshold for a match (lower = stricter).
- `MAX_CONTENT_LENGTH = 12MB` — caps upload size to avoid DoS.
- `AUGMENTATIONS_PER_PHOTO = 8` — synthetic variants generated per reference photo.

## Known limitations / notes

- The frontend and backend are assumed to run on the same machine; `API_BASE` is hardcoded to `localhost:5001`.
- Model warm-up happens at first request if pre-loading at startup is slow.
- Accuracy depends on photo quality; enroll with multiple well-lit, front-facing shots.
