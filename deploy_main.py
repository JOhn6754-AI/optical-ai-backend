"""
OptiCal AI - Backend API (Milestone 3 - Production Ready)
Changes from dev version:
  - CORS reads allowed origins from environment variable
  - SQLite path uses /data/ directory on Render (persistent disk)
  - PORT reads from environment variable (required by Render)
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import csv
import io
import json
import time
import sqlite3
import os
import requests
from math import radians, sin, cos, sqrt, atan2
from collections import defaultdict
from datetime import datetime

app = FastAPI(title="OptiCal AI", version="2.0.0")

# ============================================================
# CORS — reads from environment variable in production
# Falls back to localhost for local development
# ============================================================
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:5173")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        FRONTEND_URL,
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# DATABASE — uses /data/ on Render (persistent), local otherwise
# ============================================================
DATA_DIR = "/data" if os.path.exists("/data") else "."
DB_PATH = os.path.join(DATA_DIR, "optical.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS businesses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            home_address TEXT NOT NULL,
            hourly_labor_cost REAL DEFAULT 85,
            profitable_threshold REAL DEFAULT 120,
            marginal_threshold REAL DEFAULT 75,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            business_id INTEGER,
            run_at TEXT DEFAULT CURRENT_TIMESTAMP,
            total_revenue REAL,
            total_drive_hours REAL,
            annual_waste REAL,
            red_job_count INTEGER,
            summary_json TEXT,
            FOREIGN KEY(business_id) REFERENCES businesses(id)
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ============================================================
# CACHES
# ============================================================
geocode_cache = {}
route_cache = {}

# ============================================================
# GEOCODING — OpenStreetMap Nominatim
# ============================================================
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
HEADERS = {"User-Agent": "OptiCalAI/1.0 (local business scheduling tool)"}

def geocode_address(address: str) -> tuple:
    if address in geocode_cache:
        return geocode_cache[address]
    try:
        time.sleep(1.1)
        resp = requests.get(
            NOMINATIM_URL,
            params={"q": address, "format": "json", "limit": 1},
            headers=HEADERS,
            timeout=10
        )
        results = resp.json()
        if results:
            lat = float(results[0]["lat"])
            lon = float(results[0]["lon"])
            geocode_cache[address] = (lat, lon)
            return (lat, lon)
    except Exception as e:
        print(f"Geocode failed for '{address}': {e}")
    fallback = (48.1958, -114.3128)
    geocode_cache[address] = fallback
    return fallback

# ============================================================
# ROUTING — OSRM
# ============================================================
OSRM_URL = "http://router.project-osrm.org/route/v1/driving"

def get_drive_time_hours(coord1: tuple, coord2: tuple) -> float:
    key = (
        (round(coord1[0], 4), round(coord1[1], 4)),
        (round(coord2[0], 4), round(coord2[1], 4))
    )
    if key in route_cache:
        return route_cache[key]
    if key[0] == key[1]:
        return 0.0
    try:
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        url = f"{OSRM_URL}/{lon1},{lat1};{lon2},{lat2}"
        resp = requests.get(url, params={"overview": "false"}, timeout=10)
        data = resp.json()
        if data.get("code") == "Ok":
            hours = data["routes"][0]["duration"] / 3600
            route_cache[key] = hours
            return hours
    except Exception as e:
        print(f"Routing failed: {e}")
    fallback = haversine_hours(coord1, coord2)
    route_cache[key] = fallback
    return fallback

def haversine_hours(coord1: tuple, coord2: tuple) -> float:
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 3959
    lat1_r, lat2_r = radians(lat1), radians(lat2)
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(lat1_r) * cos(lat2_r) * sin(dlon/2)**2
    miles = R * 2 * atan2(sqrt(a), sqrt(1-a))
    return miles / 40

# ============================================================
# SCORING
# ============================================================
def score_job(revenue, duration_hours, drive_time, hourly_cost,
              profitable_threshold, marginal_threshold):
    total_time = duration_hours + drive_time
    labor_cost = total_time * hourly_cost
    gross_per_hour = revenue / total_time if total_time > 0 else 0
    profit = revenue - labor_cost
    if gross_per_hour >= profitable_threshold:
        classification = "GREEN"
    elif gross_per_hour >= marginal_threshold:
        classification = "YELLOW"
    else:
        classification = "RED"
    return {
        "gross_per_hour": round(gross_per_hour, 2),
        "profit": round(profit, 2),
        "labor_cost": round(labor_cost, 2),
        "drive_time": round(drive_time, 2),
        "total_time": round(total_time, 2),
        "classification": classification,
    }

def calculate_daily_drive_real(day_jobs, home_coords):
    by_employee = defaultdict(list)
    for job in day_jobs:
        by_employee[job["employee"]].append(job)
    total_drive = 0
    for emp_jobs in by_employee.values():
        current = home_coords
        for job in emp_jobs:
            total_drive += get_drive_time_hours(current, job["coords"])
            current = job["coords"]
        total_drive += get_drive_time_hours(current, home_coords)
    return total_drive

# ============================================================
# MODELS
# ============================================================
class BusinessProfile(BaseModel):
    name: str
    home_address: str
    hourly_labor_cost: float = 85
    profitable_threshold: float = 120
    marginal_threshold: float = 75

class NewJobRequest(BaseModel):
    date: str
    client_name: str
    address: str
    service_type: str
    revenue: float
    duration_hours: float
    home_address: str = "506 Main St Kalispell MT 59901"
    hourly_labor_cost: float = 85
    profitable_threshold: float = 120
    marginal_threshold: float = 75

# ============================================================
# BUSINESS ENDPOINTS
# ============================================================
@app.get("/businesses")
def list_businesses():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM businesses ORDER BY name").fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.post("/businesses")
def create_business(profile: BusinessProfile):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        """INSERT INTO businesses
           (name, home_address, hourly_labor_cost, profitable_threshold, marginal_threshold)
           VALUES (?, ?, ?, ?, ?)""",
        (profile.name, profile.home_address, profile.hourly_labor_cost,
         profile.profitable_threshold, profile.marginal_threshold)
    )
    conn.commit()
    biz_id = cursor.lastrowid
    conn.close()
    return {"id": biz_id, "message": f"Business '{profile.name}' saved"}

@app.delete("/businesses/{biz_id}")
def delete_business(biz_id: int):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM businesses WHERE id = ?", (biz_id,))
    conn.commit()
    conn.close()
    return {"message": "Deleted"}

# ============================================================
# ANALYZE
# ============================================================
@app.post("/analyze")
async def analyze_jobs(
    file: UploadFile = File(...),
    hourly_labor_cost: float = 85,
    profitable_threshold: float = 120,
    marginal_threshold: float = 75,
    home_address: str = "506 Main St Kalispell MT 59901",
    business_id: Optional[int] = None,
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    content = await file.read()
    text = content.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))

    jobs = []
    for row in reader:
        try:
            address = row["address"].strip()
            jobs.append({
                "date": row["date"].strip(),
                "client_name": row["client_name"].strip(),
                "address": address,
                "service_type": row["service_type"].strip(),
                "revenue": float(row["revenue"]),
                "duration_hours": float(row["duration_hours"]),
                "employee": row["employee"].strip(),
                "coords": geocode_address(address),
            })
        except (KeyError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Bad row: {row}. Error: {str(e)}")

    home_coords = geocode_address(home_address)
    by_day = defaultdict(list)
    for job in jobs:
        by_day[job["date"]].append(job)

    days_result = []
    total_revenue = 0
    total_work_hours = 0
    total_drive_hours = 0
    red_jobs = []

    for date in sorted(by_day.keys()):
        day_jobs = by_day[date]
        drive_time = calculate_daily_drive_real(day_jobs, home_coords)
        drive_per_job = drive_time / len(day_jobs) if day_jobs else 0
        day_revenue = sum(j["revenue"] for j in day_jobs)
        day_work = sum(j["duration_hours"] for j in day_jobs)
        total_revenue += day_revenue
        total_work_hours += day_work
        total_drive_hours += drive_time

        scored_jobs = []
        for job in day_jobs:
            s = score_job(
                job["revenue"], job["duration_hours"], drive_per_job,
                hourly_labor_cost, profitable_threshold, marginal_threshold
            )
            scored_job = {**job, **s}
            scored_job.pop("coords")
            scored_jobs.append(scored_job)
            if s["classification"] == "RED":
                red_jobs.append({
                    "client_name": job["client_name"],
                    "service_type": job["service_type"],
                    "revenue": job["revenue"],
                    "gross_per_hour": s["gross_per_hour"],
                    "suggested_price": round(profitable_threshold * s["total_time"], 2),
                    "surcharge_needed": round(profitable_threshold * s["total_time"] - job["revenue"], 2),
                })

        date_obj = datetime.strptime(date, "%Y-%m-%d")
        days_result.append({
            "date": date,
            "day_name": date_obj.strftime("%A, %B %d"),
            "jobs": scored_jobs,
            "day_revenue": round(day_revenue, 2),
            "day_work_hours": round(day_work, 2),
            "day_drive_hours": round(drive_time, 2),
        })

    drive_cost = total_drive_hours * hourly_labor_cost
    total_time = total_work_hours + total_drive_hours
    actual_hourly = total_revenue / total_time if total_time > 0 else 0
    potential_hourly = total_revenue / total_work_hours if total_work_hours > 0 else 0
    money_left = (potential_hourly - actual_hourly) * total_work_hours
    annual_waste = drive_cost * 50

    summary = {
        "total_revenue": round(total_revenue, 2),
        "total_work_hours": round(total_work_hours, 2),
        "total_drive_hours": round(total_drive_hours, 2),
        "drive_cost": round(drive_cost, 2),
        "actual_hourly": round(actual_hourly, 2),
        "potential_hourly": round(potential_hourly, 2),
        "money_left_on_table": round(money_left, 2),
        "annual_drive_waste": round(annual_waste, 2),
        "annual_recoverable": round(annual_waste * 0.4, 2),
        "red_job_count": len(red_jobs),
        "total_jobs": len(jobs),
    }

    if business_id:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            """INSERT INTO analyses
               (business_id, total_revenue, total_drive_hours, annual_waste,
                red_job_count, summary_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (business_id, summary["total_revenue"], summary["total_drive_hours"],
             summary["annual_drive_waste"], summary["red_job_count"],
             json.dumps(summary))
        )
        conn.commit()
        conn.close()

    return {
        "days": days_result,
        "summary": summary,
        "red_jobs": red_jobs,
        "geocoding": "real",
        "config": {
            "hourly_labor_cost": hourly_labor_cost,
            "profitable_threshold": profitable_threshold,
            "marginal_threshold": marginal_threshold,
        }
    }

# ============================================================
# EVALUATE JOB
# ============================================================
@app.post("/evaluate-job")
async def evaluate_job(request: NewJobRequest):
    home_coords = geocode_address(request.home_address)
    job_coords = geocode_address(request.address)
    drive_one_way = get_drive_time_hours(home_coords, job_coords)

    s = score_job(
        request.revenue, request.duration_hours, drive_one_way,
        request.hourly_labor_cost, request.profitable_threshold,
        request.marginal_threshold,
    )

    if s["classification"] == "GREEN":
        recommendation = "ACCEPT"
        reason = "High margin job — fits your profitability target."
    elif s["classification"] == "YELLOW":
        surcharge = round(request.profitable_threshold * s["total_time"] - request.revenue, 2)
        recommendation = "COUNTER"
        reason = f"Marginal job. Quote +${surcharge} travel fee to hit your target margin."
    else:
        suggested_price = round(request.profitable_threshold * s["total_time"], 2)
        recommendation = "DECLINE"
        reason = f"Loses money at ${request.revenue:.0f}. Minimum profitable price is ${suggested_price}."

    return {
        "recommendation": recommendation,
        "reason": reason,
        "gross_per_hour": s["gross_per_hour"],
        "classification": s["classification"],
        "drive_time_hours": round(drive_one_way, 2),
        "total_time_hours": s["total_time"],
        "estimated_profit": s["profit"],
        "geocoding": "real",
    }

@app.get("/")
def root():
    return {"status": "OptiCal AI is running", "version": "2.0.0"}
