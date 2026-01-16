# PVMA
Predictive Visitor Management Application - Thesis by Bhargavi Govardhanam, Advisor: Dr. Yi Liu, UMass Dartmouth

PVMA is a web application for managing and running visitor-demand predictions.

It consists of:

- a **FastAPI** web app (`app.py`) that provides the UI (Jinja templates)
- a **background worker** (`worker.py`) that processes queued prediction jobs
- an ML pipeline (`tourist_trend.py`) that generates forecast plots and CSV output

## Features

- User registration and login
- Upload and save training datasets
- Queue prediction jobs with a date range (criteria)
- View job status (waiting/running/completed/failed)
- Download outputs (plots + CSV)

## Project structure

- `app.py`: FastAPI app + HTML routes
- `worker.py`: polls `jobs` table and executes ML pipeline
- `tourist_trend.py`: data preprocessing + TensorFlow model + plotting
- `templates/`: Jinja2 templates
- `static/`: CSS and assets
- `training_data/`: per-job uploaded datasets (created at runtime)
- `output/`: generated plots and prediction CSVs (created at runtime)
- `users.db`: SQLite database (created/used at runtime)

## Requirements

- Python 3.10+ recommended
- SQLite (usually present by default on macOS/Linux)

## Setup

Create and activate a virtual environment:

`python -m venv .venv`

macOS/Linux:

`source .venv/bin/activate`

Install dependencies:

`pip install -r requirements.txt`

## Configuration

Copy `.env.example` to `.env` and set a strong session secret:

`PVMA_SECRET_KEY=change-me`

Optional overrides:

- `PVMA_DB_PATH` (default: `users.db`)
- `PVMA_TRAINING_DATA_DIR` (default: `training_data`)
- `PVMA_OUTPUT_DIR` (default: `output`)

Note: `.env` is ignored by git. Configure these values in your deployment environment.

## Run

Start the web app:

`uvicorn app:app --reload --port 5001`

Open:

- `http://127.0.0.1:5001/login`

Start the background worker (in a second terminal):

`python worker.py`

## How it works

1. You upload (or reuse) datasets via the UI.
2. The web app stores datasets under `training_data/<bucket_id>/`.
3. When you click **Predict**, the web app inserts a row into:

   - `predictions` table (dataset metadata)
   - `jobs` table (status starts as `waiting`)

4. `worker.py` polls `jobs` for `waiting` jobs, sets status to `running`, then calls `tourist_trend.main(...)`.
5. The ML pipeline writes results to `output/<job_id>.png`, `output/<job_id>_2.png` and `output/<job_id>.csv`.
6. The UI reads outputs from `/output/{job_id}` and updates job status.

## Dataset format

The UI expects **three CSV files** (Statistical, Climate, Google Trends). They should cover the same date range.

Required columns (exact names expected by the pipeline):

- Statistical data
  - `date`
  - `visitors`
- Climate data
  - `date`
  - `tavg`
- Google Trends data
  - `date`
  - `travel`

## Outputs

- Forecast plot: `output/<job_id>.png`
- Recent-window plot: `output/<job_id>_2.png`
- Forecast CSV: `output/<job_id>.csv`

## Security notes

- Authentication uses a signed cookie session. Set `PVMA_SECRET_KEY` to a strong secret in production.
- Passwords are stored hashed (PBKDF2-SHA256). If your `users.db` contained plaintext passwords previously, they will be upgraded on next successful login.

## Troubleshooting

- If you see `Output not found`, ensure `worker.py` is running and the job has reached `completed`.
- If dependencies fail to install, verify your Python version and try:

  `python -m pip install --upgrade pip`

- If TensorFlow installation is problematic on your machine, consider using a Python version compatible with your TensorFlow build.
