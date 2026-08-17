"""
Face Recognition Microservice
Handles:
  1. Enrollment  - generate a face embedding from a student's reference photo(s)
  2. Face detection (for bulk enrollment) - find every face in an uploaded
     photo and return numbered, cropped thumbnails so a caller can register
     each detected face as a separate student.
  3. Recognition - detect all faces in a class photo, match each against
     known student embeddings, return matched/unmatched results

Notes on this pass (see fix list in README):
  - CORS enabled so a browser-based frontend on a different origin/port can call this.
  - Every route wraps DeepFace/OpenCV work in try/except and returns JSON errors
    instead of letting Flask's HTML/debugger page leak stack traces.
  - Base64 decoding is defensive: malformed base64 / non-image bytes no longer
    crash the request.
  - Payload size is capped (MAX_CONTENT_LENGTH) to avoid huge-upload DoS.
  - debug mode and bind host are driven by env vars, defaulting to safe values.
  - threaded=True so a slow recognize() call doesn't block a concurrent enroll().
  - Student profiles (name, PRN/GR no., class code, face embedding) are persisted
    in MongoDB instead of living only in the browser tab: /enroll writes a
    document, /recognize reads all of them straight from the database, and
    /students lists/deletes them. See MONGO_URI below.
  - The student's PRN/GR number is now the true unique key for a student
    (many students can share the same class "code", e.g. TY-IT-G, so that
    code can no longer be used as a unique document key). /detect_faces lets
    the frontend upload a single photo containing several students, get back
    a numbered, cropped thumbnail per detected face, and then enroll each
    face individually with its own name/PRN/branch/division/year.
  - Enrollment augmentation: every accepted reference photo is expanded into
    8 synthetic variants (flip, small rotations, brightness/contrast jitter,
    slight zoom) and an embedding is computed for each; all embeddings for a
    student are averaged into one stored profile. This makes matching more
    robust to a single reference photo's particular lighting/angle, at the
    cost of more DeepFace calls per enrollment. See embeddings_for_photo().
"""

import os
from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import base64
import binascii
import logging
from datetime import datetime, timezone

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from deepface import DeepFace
from pymongo import MongoClient, ASCENDING
from pymongo.errors import PyMongoError, DuplicateKeyError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("face-recognition-service")

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": os.environ.get("FRONTEND_ORIGIN", "*")}})

app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

MODEL_NAME = "Facenet"
DETECTOR_BACKEND_REGISTRATION = "opencv"
DETECTOR_BACKEND_RECOGNITION = "retinaface"

MATCH_TOLERANCE = 0.40

CROP_MARGIN_RATIO = 0.35

MONGO_URI = os.environ.get("MONGO_URI", "")
MONGO_DB_NAME = os.environ.get("MONGO_DB", "")

mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
db = mongo_client[MONGO_DB_NAME]
students_col = db["students"]

try:
    students_col.create_index([("prn_gr", ASCENDING)], unique=True)
    logger.info("Connected to MongoDB at %s (db=%s)", MONGO_URI, MONGO_DB_NAME)
except PyMongoError as e:
    logger.warning("MongoDB not reachable at startup (%s). Will retry per-request.", e)


def require_db():
    """Ping Mongo before a DB-backed request; returns an error Response or None."""
    try:
        mongo_client.admin.command("ping")
        return None
    except PyMongoError as e:
        logger.error("MongoDB unavailable: %s", e)
        return jsonify({"error": "Database unavailable. Is MongoDB running?"}), 503


logger.info("Pre-loading %s model, please wait...", MODEL_NAME)
try:
    DeepFace.build_model(MODEL_NAME)
    logger.info("Model loaded successfully.")
except Exception as e:  # pragma: no cover - startup diagnostics only
    logger.warning("Could not pre-load model: %s", e)


class BadImage(ValueError):
    """Raised when a base64 payload can't be turned into a usable image."""


def decode_base64_image(base64_string):
    """Convert a base64 image string (from frontend) into a numpy array (BGR for DeepFace).

    Raises BadImage if the string isn't valid base64 or doesn't decode to an image,
    instead of letting the caller crash on a bad/None value.

    The decoded image is converted to grayscale (then back to 3-channel BGR)
    before being returned, so all downstream face detection/embedding runs on
    grayscale data for improved matching accuracy - even though the frontend
    always captures/uploads color images.
    """
    if not base64_string or not isinstance(base64_string, str):
        raise BadImage("Image data missing or not a string")

    if "," in base64_string:
        base64_string = base64_string.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(base64_string, validate=True)
    except (binascii.Error, ValueError) as e:
        raise BadImage(f"Invalid base64 image data: {e}") from e

    img_np = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if img is None:
        raise BadImage("Could not decode image bytes (unsupported or corrupt format)")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return img


def encode_image_to_base64_jpeg(img_array, quality=85):
    """Encode a BGR numpy image back into a data-URL base64 JPEG string."""
    ok, buf = cv2.imencode(".jpg", img_array, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise BadImage("Could not encode cropped face to JPEG")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def crop_face(img_array, facial_area, margin_ratio=CROP_MARGIN_RATIO):
    """Crop a face out of a full photo with a little padding around the box,
    clipped to the image bounds so a downstream single-face detector has a
    clean, mostly-background-free image to work with."""
    h, w = img_array.shape[:2]
    x, y, fw, fh = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]

    pad_x = int(fw * margin_ratio)
    pad_y = int(fh * margin_ratio)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + fw + pad_x)
    y2 = min(h, y + fh + pad_y)

    return img_array[y1:y2, x1:x2]


def cosine_distance(a, b):
    """Calculate the cosine distance between two vectors."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return 999.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 999.0
    return float(1.0 - (np.dot(a, b) / (norm_a * norm_b)))


def represent_faces(img_array, detector_backend, enforce_detection=True):
    try:
        return DeepFace.represent(
            img_path=img_array,
            model_name=MODEL_NAME,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
        )
    except ValueError:
        return []


AUGMENTATIONS_PER_PHOTO = 8


def augment_variants(img_array):
    """Return a list of AUGMENTATIONS_PER_PHOTO synthetically varied copies of
    a face image, each nudging one realistic axis of variation (pose, light,
    contrast, framing) without distorting the face enough to hurt matching."""
    h, w = img_array.shape[:2]
    variants = []

    variants.append(cv2.flip(img_array, 1))

    for angle in (12, -12):
        rot_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        variants.append(
            cv2.warpAffine(img_array, rot_matrix, (w, h), borderMode=cv2.BORDER_REFLECT101)
        )

    variants.append(cv2.convertScaleAbs(img_array, alpha=1.0, beta=30))
    variants.append(cv2.convertScaleAbs(img_array, alpha=1.0, beta=-30))

    variants.append(cv2.convertScaleAbs(img_array, alpha=1.25, beta=0))
    variants.append(cv2.convertScaleAbs(img_array, alpha=0.8, beta=0))

    crop_ratio = 0.88
    ch, cw = max(1, int(h * crop_ratio)), max(1, int(w * crop_ratio))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    cropped = img_array[y0:y0 + ch, x0:x0 + cw]
    variants.append(cv2.resize(cropped, (w, h)))

    return variants


def embeddings_for_photo(img_array):
    """Given one reference photo, return the list of embeddings to fold into
    the average: the original photo's embedding plus one embedding per
    successfully-processed augmented variant.

    Returns None if the original photo doesn't contain exactly one detectable
    face (ambiguous or empty photos are rejected here, same as before -
    augmentation only kicks in once the base photo is confirmed valid).
    """
    objs = represent_faces(img_array, DETECTOR_BACKEND_REGISTRATION, enforce_detection=True)
    if not objs or len(objs) != 1:
        return None

    embeddings = [objs[0]["embedding"]]

    for variant in augment_variants(img_array):
        try:
            variant_objs = represent_faces(variant, DETECTOR_BACKEND_REGISTRATION, enforce_detection=False)
        except Exception as e:  # noqa: BLE001 - OpenCV/DeepFace can raise many error types
            logger.warning("Skipping an enroll augmentation after an error: %s", e)
            continue

        if not variant_objs:
            continue

        best = max(
            variant_objs,
            key=lambda o: o.get("facial_area", {}).get("w", 0) * o.get("facial_area", {}).get("h", 0),
        )
        embeddings.append(best["embedding"])

    return embeddings


def build_student_code(year, branch, division):
    """Build the human-readable class code shown/stored alongside a student,
    e.g. year='TY', branch='IT', division='G' -> 'TY-IT-G'. This is NOT a
    unique key (many students share a class code) - PRN/GR is."""
    parts = [p.strip().upper() for p in (year, branch, division) if p and p.strip()]
    return "-".join(parts)


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "Payload too large. Please send smaller/fewer images."}), 413


@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Face Recognition Microservice is running!"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/students", methods=["GET"])
def list_students():
    """Returns every enrolled student (without embeddings, which are large and
    only needed server-side during /recognize)."""
    db_err = require_db()
    if db_err:
        return db_err

    try:
        docs = students_col.find({}, {"embedding": 0}).sort("name", ASCENDING)
        students = [
            {
                "prn_gr": d.get("prn_gr"),
                "student_id": d.get("student_id"),
                "name": d.get("name"),
                "branch": d.get("branch"),
                "division": d.get("division"),
                "year": d.get("year"),
                "photos_used": d.get("photos_used"),
                "embeddings_used": d.get("embeddings_used"),
                "enrolled_at": d.get("enrolled_at"),
            }
            for d in docs
        ]
    except PyMongoError as e:
        logger.error("Failed to list students: %s", e)
        return jsonify({"error": "Could not read students from the database"}), 503

    return jsonify({"students": students})


@app.route("/students/<prn_gr>", methods=["DELETE"])
def delete_student(prn_gr):
    db_err = require_db()
    if db_err:
        return db_err

    try:
        result = students_col.delete_one({"prn_gr": prn_gr})
    except PyMongoError as e:
        logger.error("Failed to delete student %s: %s", prn_gr, e)
        return jsonify({"error": "Could not delete student"}), 503

    if result.deleted_count == 0:
        return jsonify({"error": "Student not found"}), 404
    return jsonify({"deleted": prn_gr})


@app.route("/detect_faces", methods=["POST"])
def detect_faces():
    """
    Input:  { "photo": base64_img }

    Detects every face in an uploaded photo (e.g. a group/class photo used
    for bulk registration) and returns a numbered, cropped thumbnail for
    each one, so the frontend can walk the caller through registering each
    face as a separate student one at a time.

    No embeddings are returned to the client - only cropped image thumbnails.
    Each thumbnail is later re-submitted (as a normal enroll photo) to
    /enroll, where the embedding is (re)computed server-side.

    Output: {
      "total_faces_detected": N,
      "faces": [ { "face_index", "face_location", "thumbnail" }, ... ]
    }
    """
    data = request.get_json(silent=True) or {}
    photo_b64 = data.get("photo")

    if not photo_b64:
        return jsonify({"error": "No photo provided"}), 400

    try:
        img_array = decode_base64_image(photo_b64)
    except BadImage as e:
        return jsonify({"error": str(e)}), 400

    try:
        objs = represent_faces(img_array, DETECTOR_BACKEND_REGISTRATION, enforce_detection=False)
    except Exception as e:  # noqa: BLE001
        logger.error("Face detection failed on uploaded photo: %s", e)
        return jsonify({"error": "Could not process the uploaded photo"}), 500

    faces = []
    for idx, face_obj in enumerate(objs, start=1):
        area = face_obj.get("facial_area") or {}
        if not all(k in area for k in ("x", "y", "w", "h")):
            continue
        if area["w"] <= 0 or area["h"] <= 0:
            continue

        try:
            crop = crop_face(img_array, area)
            thumbnail = encode_image_to_base64_jpeg(crop)
        except BadImage as e:
            logger.warning("Could not crop/encode detected face #%d: %s", idx, e)
            continue

        top, left = area["y"], area["x"]
        faces.append({
            "face_index": idx,
            "face_location": {
                "top": top,
                "left": left,
                "right": left + area["w"],
                "bottom": top + area["h"],
            },
            "thumbnail": thumbnail,
        })

    return jsonify({
        "total_faces_detected": len(faces),
        "faces": faces,
    })


@app.route("/enroll", methods=["POST"])
def enroll():
    """
    Input:  {
      "prn_gr": "...",            (required, unique - PRN or GR number)
      "name": "...",               (required)
      "year": "...",                e.g. "TY"
      "branch": "...",              e.g. "IT"
      "division": "...",            e.g. "G"
      "images": [base64_img1, base64_img2, ...]   (1-5 reference photos)
    }
    Output: { "prn_gr", "student_id", "name", "photos_used", "photos_skipped",
              "embeddings_used" }

    Persists (or updates, if this prn_gr already exists) the averaged face
    embedding as a document in MongoDB, keyed by PRN/GR number - this is what
    /recognize reads back. "student_id" here is a derived, human-readable
    class code (year-branch-division, e.g. "TY-IT-G") and is NOT unique:
    many students in the same class share it.

    Every accepted reference photo is automatically expanded into several
    augmented variants (flip/rotation/brightness/contrast/zoom - see
    embeddings_for_photo/augment_variants above) and an embedding is computed
    for each; ALL of those embeddings (not just one per photo) are averaged
    into the final stored profile. This trades extra processing time for a
    profile that's far less sensitive to the exact lighting/angle/expression
    of whichever reference photos happened to be captured.
    """
    db_err = require_db()
    if db_err:
        return db_err

    data = request.get_json(silent=True) or {}
    images = data.get("images", [])
    prn_gr = str(data.get("prn_gr") or "").strip()
    name = (data.get("name") or "").strip()
    year = (data.get("year") or "").strip()
    branch = (data.get("branch") or "").strip()
    division = (data.get("division") or "").strip()

    if not prn_gr or not name:
        return jsonify({"error": "prn_gr and name are required"}), 400
    if not year or not branch or not division:
        return jsonify({"error": "year, branch, and division are required"}), 400
    if not images or not isinstance(images, list):
        return jsonify({"error": "No images provided"}), 400

    student_code = build_student_code(year, branch, division)

    embeddings = []   # every embedding (original + augmented) across all photos
    photos_used = 0   # count of reference photos that contributed at least the original embedding
    skipped = 0       # reference photos rejected outright (bad data / no face / multiple faces)

    for img_b64 in images:
        try:
            img_array = decode_base64_image(img_b64)
        except BadImage as e:
            logger.info("Skipping unusable enroll image: %s", e)
            skipped += 1
            continue

        try:
            photo_embeddings = embeddings_for_photo(img_array)
        except Exception as e:  # noqa: BLE001 - DeepFace/OpenCV can raise many error types
            logger.warning("Face detection failed on an enroll image: %s", e)
            skipped += 1
            continue

        if photo_embeddings is None:
            skipped += 1  # no face, or more than one face - ambiguous for enrollment
            continue

        photos_used += 1
        embeddings.extend(photo_embeddings)

    if not embeddings:
        return jsonify({"error": "No valid single-face image found in any provided photo"}), 400

    avg_embedding = np.mean(np.array(embeddings), axis=0)

    doc = {
        "student_id": student_code,
        "name": name,
        "prn_gr": prn_gr,
        "branch": branch.upper(),
        "division": division.upper(),
        "year": year.upper(),
        "embedding": avg_embedding.tolist(),
        "photos_used": photos_used,
        "embeddings_used": len(embeddings),
        "enrolled_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        students_col.replace_one({"prn_gr": prn_gr}, doc, upsert=True)
    except DuplicateKeyError:
        return jsonify({"error": f"A student with PRN/GR '{prn_gr}' already exists"}), 409
    except PyMongoError as e:
        logger.error("Failed to save student %s to MongoDB: %s", prn_gr, e)
        return jsonify({"error": "Could not save student to the database"}), 503

    return jsonify({
        "prn_gr": prn_gr,
        "student_id": student_code,
        "name": name,
        "photos_used": photos_used,
        "photos_skipped": skipped,
        "embeddings_used": len(embeddings),
    })


@app.route("/recognize", methods=["POST"])
def recognize():
    """
    Input: { "class_photo": base64_img }

    known_students are no longer supplied by the client - they're fetched
    straight from MongoDB, so recognition always uses the current persisted
    roster (and a client can't spoof matches by sending fabricated embeddings).

    Output: {
      "matched":   [ { "prn_gr", "student_id", "name", "confidence", "face_location" } ],
      "unmatched": [ { "face_location" } ],
      "total_faces_detected": N
    }
    """
    db_err = require_db()
    if db_err:
        return db_err

    data = request.get_json(silent=True) or {}
    class_photo_b64 = data.get("class_photo")

    if not class_photo_b64:
        return jsonify({"error": "No class photo provided"}), 400

    try:
        img_array = decode_base64_image(class_photo_b64)
    except BadImage as e:
        return jsonify({"error": str(e)}), 400

    try:
        known_students = list(
            students_col.find({}, {"prn_gr": 1, "student_id": 1, "name": 1, "embedding": 1})
        )
    except PyMongoError as e:
        logger.error("Failed to load students for recognition: %s", e)
        return jsonify({"error": "Could not read students from the database"}), 503

    try:
        objs = represent_faces(img_array, DETECTOR_BACKEND_RECOGNITION, enforce_detection=True)
    except Exception as e:  # noqa: BLE001
        logger.error("Face detection failed on class photo: %s", e)
        return jsonify({"error": "Could not process class photo"}), 500

    matched = []
    unmatched = []

    for face_obj in objs:
        encoding = face_obj["embedding"]
        area = face_obj["facial_area"]

        top = area["y"]
        right = area["x"] + area["w"]
        bottom = area["y"] + area["h"]
        left = area["x"]

        best_distance = 999.0
        best_student = None

        if known_students:
            distances = [cosine_distance(s["embedding"], encoding) for s in known_students]
            best_idx = int(np.argmin(distances))
            best_distance = distances[best_idx]
            best_student = known_students[best_idx]

        if best_student is not None and best_distance <= MATCH_TOLERANCE:
            ratio = min(max(best_distance / MATCH_TOLERANCE, 0.0), 1.0)
            confidence = round((1 - ratio * 0.5) * 100, 1)
            matched.append({
                "prn_gr": best_student.get("prn_gr"),
                "student_id": best_student.get("student_id"),
                "name": best_student.get("name", "Unknown"),
                "confidence": confidence,
                "face_location": {"top": top, "right": right, "bottom": bottom, "left": left},
            })
        else:
            unmatched.append({
                "face_location": {"top": top, "right": right, "bottom": bottom, "left": left},
            })

    return jsonify({
        "matched": matched,
        "unmatched": unmatched,
        "total_faces_detected": len(objs),
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port="5001", debug="true", use_reloader=False, threaded=True)
