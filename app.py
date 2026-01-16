import base64
import hashlib
import hmac
import os
import secrets
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from starlette.status import HTTP_302_FOUND
from starlette.templating import Jinja2Templates

DB_PATH = os.getenv("PVMA_DB_PATH", "users.db")
TRAINING_DATA_DIR = Path(os.getenv("PVMA_TRAINING_DATA_DIR", "training_data"))
OUTPUT_DIR = Path(os.getenv("PVMA_OUTPUT_DIR", "output"))


def connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = connect_db()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username varchar(255) NOT NULL,
                email varchar(255) NOT NULL UNIQUE,
                password varchar(255) NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                data_bucket_id INTEGER NOT NULL UNIQUE,
                prediction_id varchar(255) NOT NULL UNIQUE,
                title varchar(255) NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                job_id varchar(255) NOT NULL UNIQUE,
                bucket_id varchar(255) NOT NULL,
                status varchar(255),
                start_date varchar(255),
                end_date varchar(255),
                criteria varchar(255)
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def _hash_password(password: str, *, salt: bytes | None = None, iterations: int = 210_000) -> str:
    salt_bytes = salt if salt is not None else secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, iterations)
    salt_b64 = base64.b64encode(salt_bytes).decode("ascii")
    dk_b64 = base64.b64encode(dk).decode("ascii")
    return f"pbkdf2_sha256${iterations}${salt_b64}${dk_b64}"


def _verify_password(password: str, stored: str) -> bool:
    if stored.startswith("pbkdf2_sha256$"):
        parts = stored.split("$")
        if len(parts) != 4:
            return False
        _, iterations_s, salt_b64, dk_b64 = parts
        try:
            iterations = int(iterations_s)
            salt = base64.b64decode(salt_b64.encode("ascii"))
            expected = base64.b64decode(dk_b64.encode("ascii"))
        except Exception:
            return False
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(actual, expected)
    return secrets.compare_digest(password, stored)


def generate_random_string() -> str:
    import random
    import string

    return "".join(random.choices(string.ascii_uppercase + string.digits, k=10))


def generate_prediction_id() -> str:
    return "pred_" + generate_random_string()


def generate_train_data_id() -> str:
    return "tr_" + generate_random_string()


def generate_job_id() -> str:
    return "job_" + generate_random_string()


def month_diff(start_date: datetime, end_date: datetime) -> int:
    return (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)


def _redirect(url: str) -> RedirectResponse:
    return RedirectResponse(url=url, status_code=HTTP_302_FOUND)


def get_current_user(request: Request) -> dict[str, Any]:
    user = request.session.get("user")
    if not user:
        raise RuntimeError("unauthorized")
    return user


app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("PVMA_SECRET_KEY", "change-me"),
    https_only=False,
    same_site="lax",
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
def _startup() -> None:
    TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    init_db()


@app.get("/login", response_class=HTMLResponse)
def login_get(request: Request, msg: str | None = None):
    if request.session.get("user"):
        return _redirect("/")
    return templates.TemplateResponse("login.html", {"request": request, "title": "Login", "msg": msg})


@app.post("/login")
def login_post(request: Request, email: str = Form(...), password: str = Form(...)):
    conn = connect_db()
    try:
        row = conn.execute("SELECT id, password FROM users WHERE email = ?", (email,)).fetchone()
        if not row:
            return _redirect("/login?msg=Invalid+Credentials.+Please+try+again.")
        stored_pwd = row["password"]
        if not _verify_password(password, stored_pwd):
            return _redirect("/login?msg=Invalid+Credentials.+Please+try+again.")
        if not stored_pwd.startswith("pbkdf2_sha256$"):
            conn.execute("UPDATE users SET password = ? WHERE id = ?", (_hash_password(password), row["id"]))
            conn.commit()
    finally:
        conn.close()

    request.session["user"] = {"email": email, "id": row["id"]}
    return _redirect("/")


@app.get("/register", response_class=HTMLResponse)
def register_get(request: Request, msg: str | None = None):
    if request.session.get("user"):
        return _redirect("/")
    return templates.TemplateResponse("register.html", {"request": request, "title": "Register", "msg": msg})


@app.post("/register")
def register_post(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirmPassword: str = Form(...),
):
    if password != confirmPassword:
        return _redirect("/register?msg=Passwords+do+not+match")

    conn = connect_db()
    try:
        existing = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if existing:
            return _redirect("/register?msg=User+already+exists")
        conn.execute(
            "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
            (name, email, _hash_password(password)),
        )
        conn.commit()
    finally:
        conn.close()
    return _redirect("/login?msg=Registration+successful")


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request, msg: str | None = None):
    try:
        user = get_current_user(request)
    except RuntimeError:
        return _redirect("/login")

    conn = connect_db()
    try:
        rows = conn.execute(
            """
            SELECT jobs.id, jobs.job_id, jobs.start_date, jobs.end_date, jobs.status, predictions.title
            FROM jobs
            INNER JOIN predictions ON jobs.bucket_id = predictions.data_bucket_id
            WHERE jobs.user_id = ?
            """,
            (user["id"],),
        ).fetchall()
        dict_result = [dict(row) for row in rows]
    finally:
        conn.close()

    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "title": "Dashboard", "predictions": dict_result, "msg": msg},
    )


@app.get("/new-training", response_class=HTMLResponse)
def new_training(request: Request, msg: str | None = None):
    try:
        user = get_current_user(request)
    except RuntimeError:
        return _redirect("/login")

    conn = connect_db()
    try:
        rows = conn.execute(
            "SELECT id, data_bucket_id, prediction_id, title FROM predictions WHERE user_id = ?",
            (user["id"],),
        ).fetchall()
        dict_result = [dict(row) for row in rows]
    finally:
        conn.close()

    return templates.TemplateResponse(
        "newTraining.html",
        {"request": request, "title": "New Training", "predictions": dict_result, "msg": msg},
    )


@app.get("/result", response_class=HTMLResponse)
def detail_page(request: Request, job_id: str, msg: str | None = None):
    try:
        user = get_current_user(request)
    except RuntimeError:
        return _redirect("/login")

    conn = connect_db()
    try:
        row = conn.execute(
            """
            SELECT jobs.start_date, jobs.end_date, jobs.status, predictions.title
            FROM jobs
            INNER JOIN predictions ON jobs.bucket_id = predictions.data_bucket_id
            WHERE jobs.user_id = ? AND predictions.user_id = jobs.user_id AND jobs.job_id = ?
            """,
            (user["id"], job_id),
        ).fetchone()
        if not row:
            return _redirect("/?msg=Result+not+found")
        dict_result = dict(row)
        dict_result["job_id"] = job_id
    finally:
        conn.close()

    return templates.TemplateResponse(
        "result.html",
        {"request": request, "title": "Result Page", "result": dict_result, "msg": msg},
    )


@app.post("/start-predict")
async def start_predict(
    request: Request,
    title: str = Form(...),
    from_time_prediction: str = Form(..., alias="from-time-prediction"),
    to_time_prediction: str = Form(..., alias="to-time-prediction"),
    climate_data: UploadFile = File(..., alias="climate-data"),
    google_trends_data: UploadFile = File(..., alias="google-trends-data"),
    statistical_data: UploadFile = File(..., alias="statistical-data"),
):
    try:
        user = get_current_user(request)
    except RuntimeError:
        return _redirect("/login")

    prediction_id = generate_prediction_id()
    data_bucket_id = generate_train_data_id()
    job_id = generate_job_id()

    bucket_dir = TRAINING_DATA_DIR / data_bucket_id
    bucket_dir.mkdir(parents=True, exist_ok=False)

    (bucket_dir / "climate_data.csv").write_text((await climate_data.read()).decode("utf-8"), encoding="utf-8")
    (bucket_dir / "google_trends_data.csv").write_text(
        (await google_trends_data.read()).decode("utf-8"),
        encoding="utf-8",
    )
    (bucket_dir / "statistical_data.csv").write_text(
        (await statistical_data.read()).decode("utf-8"),
        encoding="utf-8",
    )

    psd = datetime.strptime(from_time_prediction, "%Y-%m-%d")
    ped = datetime.strptime(to_time_prediction, "%Y-%m-%d")
    months = month_diff(psd, ped)

    conn = connect_db()
    try:
        conn.execute(
            "INSERT INTO predictions (user_id, data_bucket_id, prediction_id, title) VALUES (?, ?, ?, ?)",
            (user["id"], data_bucket_id, prediction_id, title),
        )
        conn.execute(
            "INSERT INTO jobs (user_id, job_id, bucket_id, status, start_date, criteria) VALUES (?, ?, ?, ?, ?, ?)",
            (user["id"], job_id, data_bucket_id, "waiting", datetime.now().isoformat(), str(months)),
        )
        conn.commit()
    finally:
        conn.close()

    return _redirect("/")


@app.post("/save-data")
async def save_data(
    request: Request,
    title: str = Form(...),
    climate_data: UploadFile = File(..., alias="climate-data"),
    google_trends_data: UploadFile = File(..., alias="google-trends-data"),
    statistical_data: UploadFile = File(..., alias="statistical-data"),
):
    try:
        user = get_current_user(request)
    except RuntimeError:
        return _redirect("/login")

    prediction_id = generate_prediction_id()
    data_bucket_id = generate_train_data_id()

    bucket_dir = TRAINING_DATA_DIR / data_bucket_id
    bucket_dir.mkdir(parents=True, exist_ok=False)

    (bucket_dir / "climate_data.csv").write_text((await climate_data.read()).decode("utf-8"), encoding="utf-8")
    (bucket_dir / "google_trends_data.csv").write_text(
        (await google_trends_data.read()).decode("utf-8"),
        encoding="utf-8",
    )
    (bucket_dir / "statistical_data.csv").write_text(
        (await statistical_data.read()).decode("utf-8"),
        encoding="utf-8",
    )

    conn = connect_db()
    try:
        conn.execute(
            "INSERT INTO predictions (user_id, data_bucket_id, prediction_id, title) VALUES (?, ?, ?, ?)",
            (user["id"], data_bucket_id, prediction_id, title),
        )
        conn.commit()
    finally:
        conn.close()

    return _redirect("/new-training")


@app.post("/predict-saved")
def predict_saved(
    request: Request,
    prediction_id: str = Form(..., alias="prediction-id"),
    from_time_prediction: str = Form(..., alias="from-time-prediction"),
    to_time_prediction: str = Form(..., alias="to-time-prediction"),
):
    try:
        user = get_current_user(request)
    except RuntimeError:
        return _redirect("/login")

    job_id = generate_job_id()
    psd = datetime.strptime(from_time_prediction, "%Y-%m-%d")
    ped = datetime.strptime(to_time_prediction, "%Y-%m-%d")
    months = month_diff(psd, ped)

    conn = connect_db()
    try:
        row = conn.execute(
            "SELECT user_id, data_bucket_id FROM predictions WHERE prediction_id = ?",
            (prediction_id,),
        ).fetchone()
        if not row:
            return _redirect("/new-training?msg=Prediction+not+found")

        if row["user_id"] != user["id"]:
            return _redirect("/new-training?msg=Unauthorized")

        conn.execute(
            "INSERT INTO jobs (user_id, job_id, bucket_id, status, start_date, criteria) VALUES (?, ?, ?, ?, ?, ?)",
            (row["user_id"], job_id, row["data_bucket_id"], "waiting", datetime.now().isoformat(), str(months)),
        )
        conn.commit()
    finally:
        conn.close()

    return _redirect("/")


@app.get("/output/{job_id}")
def get_result_img(job_id: str, csv: int | None = None):
    if csv:
        filepath = OUTPUT_DIR / f"{job_id}.csv"
        if not filepath.exists():
            return _redirect("/?msg=Output+not+found")
        return FileResponse(filepath)

    filepath = OUTPUT_DIR / f"{job_id}.png"
    if not filepath.exists():
        return _redirect("/?msg=Output+not+found")
    return FileResponse(filepath)


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return _redirect("/login")