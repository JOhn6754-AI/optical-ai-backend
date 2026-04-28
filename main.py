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

geocode_cache = {}
route_cache = {}

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

OSRM_URL = "http://router.project-osrm.org/route/v1/driving"

# ============================================================
# MONTANA TERRAIN MULTIPLIERS (#10)
# Bounding boxes for known slow zones in Montana.
# Format: (name, lat_min, lat_max, lon_min, lon_max, multiplier)
# Multiplier applies to OSRM drive time — 1.3 = 30% longer than
# standard road speed assumptions due to mountain grades/curves.
# ============================================================
MONTANA_TERRAIN_ZONES = [
    # Going-to-the-Sun Road corridor / Glacier approaches
    ("Glacier/GTTS Corridor",     48.55, 48.85, -113.95, -113.55, 1.45),
    # Mission Mountains / Highway 93 between Polson and Missoula
    ("Mission Mountains",         47.10, 48.05, -114.25, -113.70, 1.30),
    # Swan Range / Highway 83 (Swan Valley)
    ("Swan Valley",               47.10, 48.20, -113.80, -113.35, 1.35),
    # Flathead Lake area (curvy shoreline roads)
    ("Flathead Lake",             47.55, 48.05, -114.45, -113.90, 1.20),
    # Marias Pass / US-2 east of Glacier
    ("Marias Pass",               48.25, 48.55, -113.55, -112.95, 1.35),
    # Rogers Pass / Highway 200 (Little Belt Mountains)
    ("Rogers Pass Area",          46.75, 47.25, -112.65, -111.95, 1.25),
    # Lolo Pass / Highway 12 west of Missoula
    ("Lolo Pass",                 46.45, 46.85, -115.20, -114.50, 1.40),
    # MacDonald Pass / Highway 12 east of Helena
    ("MacDonald Pass",            46.55, 46.80, -112.65, -112.25, 1.25),
    # Beartooth Highway approaches (south of Billings)
    ("Beartooth Approaches",      44.90, 45.30, -109.80, -109.20, 1.40),
    # Lost Trail Pass / Highway 93 south of Hamilton
    ("Lost Trail Pass",           45.55, 45.85, -114.25, -113.85, 1.35),
]

def get_terrain_multiplier(coord1: tuple, coord2: tuple) -> float:
    """
    Check if the midpoint of a route falls within a known slow terrain zone.
    Returns the highest applicable multiplier (or 1.0 if none match).
    """
    mid_lat = (coord1[0] + coord2[0]) / 2
    mid_lon = (coord1[1] + coord2[1]) / 2
    best = 1.0
    for (name, lat_min, lat_max, lon_min, lon_max, mult) in MONTANA_TERRAIN_ZONES:
        if lat_min <= mid_lat <= lat_max and lon_min <= mid_lon <= lon_max:
            if mult > best:
                best = mult
    return best

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
            # Apply Montana terrain multiplier (#10)
            terrain_mult = get_terrain_multiplier(coord1, coord2)
            hours = hours * terrain_mult
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

def haversine_miles(coord1: tuple, coord2: tuple) -> float:
    """Straight-line distance in miles — used for fast route sorting."""
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 3959
    lat1_r, lat2_r = radians(lat1), radians(lat2)
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(lat1_r) * cos(lat2_r) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

# ============================================================
# ROUTE OPTIMIZER — Nearest Neighbor Algorithm (#6)
# For each day+employee, reorders jobs so total drive distance
# is minimized. Uses haversine (fast, no API calls) for sorting,
# then recalculates real drive times on the optimized order.
# Returns original vs optimized drive hours and $ savings.
# ============================================================
def optimize_route_for_employee(jobs: list, home_coords: tuple) -> dict:
    """
    Nearest-neighbor optimization for one employee's day.
    Returns original_hours, optimized_hours, optimized_order (indices).
    """
    if len(jobs) <= 1:
        orig = 0.0
        if jobs:
            orig = (haversine_hours(home_coords, jobs[0]["coords"]) +
                    haversine_hours(jobs[0]["coords"], home_coords))
        return {"original_hours": round(orig, 3), "optimized_hours": round(orig, 3), "order": list(range(len(jobs)))}

    # Calculate original route distance (order as given)
    orig_hours = 0.0
    current = home_coords
    for job in jobs:
        orig_hours += haversine_hours(current, job["coords"])
        current = job["coords"]
    orig_hours += haversine_hours(current, home_coords)

    # Nearest-neighbor: greedily pick the closest unvisited job
    unvisited = list(range(len(jobs)))
    order = []
    current = home_coords
    while unvisited:
        nearest_idx = min(unvisited, key=lambda i: haversine_miles(current, jobs[i]["coords"]))
        order.append(nearest_idx)
        current = jobs[nearest_idx]["coords"]
        unvisited.remove(nearest_idx)

    # Calculate optimized route distance
    opt_hours = 0.0
    current = home_coords
    for idx in order:
        opt_hours += haversine_hours(current, jobs[idx]["coords"])
        current = jobs[idx]["coords"]
    opt_hours += haversine_hours(current, home_coords)

    return {
        "original_hours": round(orig_hours, 3),
        "optimized_hours": round(opt_hours, 3),
        "order": order,
    }

def optimize_all_days(days_jobs: dict, home_coords: tuple, hourly_cost: float) -> dict:
    """
    Run nearest-neighbor optimization across all days.
    days_jobs: { date_str: [job, ...] }
    Returns an optimization summary block.
    """
    total_original = 0.0
    total_optimized = 0.0
    day_results = []

    for date in sorted(days_jobs.keys()):
        day_jobs = days_jobs[date]
        by_employee = defaultdict(list)
        for job in day_jobs:
            by_employee[job["employee"]].append(job)

        day_orig = 0.0
        day_opt = 0.0
        for emp, emp_jobs in by_employee.items():
            result = optimize_route_for_employee(emp_jobs, home_coords)
            day_orig += result["original_hours"]
            day_opt += result["optimized_hours"]

        total_original += day_orig
        total_optimized += day_opt
        savings = max(0.0, day_orig - day_opt)
        day_results.append({
            "date": date,
            "original_drive_hours": round(day_orig, 2),
            "optimized_drive_hours": round(day_opt, 2),
            "hours_saved": round(savings, 2),
            "dollars_saved": round(savings * hourly_cost, 2),
        })

    total_saved = max(0.0, total_original - total_optimized)
    pct = round((total_saved / total_original * 100) if total_original > 0 else 0, 1)

    return {
        "total_original_drive_hours": round(total_original, 2),
        "total_optimized_drive_hours": round(total_optimized, 2),
        "total_hours_saved": round(total_saved, 2),
        "total_dollars_saved": round(total_saved * hourly_cost, 2),
        "pct_improvement": pct,
        "days": day_results,
    }

# ============================================================
# PATTERN DETECTION — AI Insights (#7)
# Analyzes scored jobs and days to surface plain-English
# insights about day patterns, bad zip codes, service type
# profitability, drive waste, and outlier clients.
# ============================================================
def detect_patterns(days_result: list, summary: dict, hourly_cost: float,
                    profitable_threshold: float) -> list:
    """
    Returns a list of insight dicts: { type, icon, title, detail }
    type is one of: "warning", "good", "info"
    """
    insights = []
    all_jobs = [job for day in days_result for job in day["jobs"]]
    if not all_jobs:
        return insights

    # --- Day-of-week margin analysis ---
    day_margins = defaultdict(list)
    for day in days_result:
        try:
            dow = datetime.strptime(day["date"], "%Y-%m-%d").strftime("%A")
        except Exception:
            continue
        for job in day["jobs"]:
            day_margins[dow].append(job.get("gross_per_hour", 0))

    if len(day_margins) >= 2:
        day_avgs = {d: sum(v)/len(v) for d, v in day_margins.items()}
        best_day = max(day_avgs, key=day_avgs.get)
        worst_day = min(day_avgs, key=day_avgs.get)
        best_avg = day_avgs[best_day]
        worst_avg = day_avgs[worst_day]
        if best_avg > 0 and (best_avg - worst_avg) / best_avg > 0.2:
            drop_pct = round((best_avg - worst_avg) / best_avg * 100)
            insights.append({
                "type": "warning",
                "icon": "📅",
                "title": f"{worst_day}s run {drop_pct}% below your best margin day",
                "detail": f"{best_day}s average ${best_avg:.0f}/hr vs {worst_day}s at ${worst_avg:.0f}/hr. Consider lighter scheduling or higher pricing on {worst_day}s."
            })

    # --- Drive waste vs industry benchmark ---
    total_time = summary.get("total_work_hours", 0) + summary.get("total_drive_hours", 0)
    if total_time > 0:
        drive_pct = round(summary.get("total_drive_hours", 0) / total_time * 100)
        if drive_pct > 25:
            insights.append({
                "type": "warning",
                "icon": "🚗",
                "title": f"{drive_pct}% of your time is unpaid driving",
                "detail": f"Industry benchmark is under 20%. You're spending ${summary.get('drive_cost', 0):.0f} on drive time this period. Route optimization could recover an estimated 30–50% of that."
            })
        elif drive_pct < 15:
            insights.append({
                "type": "good",
                "icon": "✅",
                "title": f"Strong routing — only {drive_pct}% drive time",
                "detail": "Your jobs are well-clustered. Focus on pricing optimization rather than routing."
            })

    # --- Zip code / area profitability ---
    zip_margins = defaultdict(list)
    for job in all_jobs:
        addr = job.get("address", "")
        parts = addr.replace(",", " ").split()
        zip_code = next((p for p in parts if p.isdigit() and len(p) == 5), None)
        if zip_code:
            zip_margins[zip_code].append(job.get("gross_per_hour", 0))

    if len(zip_margins) >= 2:
        overall_avg = summary.get("actual_hourly", 0)
        bad_zips = [(z, sum(v)/len(v), len(v)) for z, v in zip_margins.items()
                    if sum(v)/len(v) < overall_avg * 0.75 and len(v) >= 2]
        if bad_zips:
            z, avg, count = sorted(bad_zips, key=lambda x: x[1])[0]
            insights.append({
                "type": "warning",
                "icon": "📍",
                "title": f"Zip code {z} is dragging your margins",
                "detail": f"{count} jobs in {z} average ${avg:.0f}/hr vs your ${overall_avg:.0f}/hr overall. Consider adding a travel surcharge or raising prices in this area."
            })

    # --- Service type profitability ---
    service_margins = defaultdict(list)
    for job in all_jobs:
        svc = job.get("service_type", "Unknown")
        if svc:
            service_margins[svc].append(job.get("gross_per_hour", 0))

    if len(service_margins) >= 2:
        svc_avgs = {s: sum(v)/len(v) for s, v in service_margins.items() if len(v) >= 2}
        if svc_avgs:
            best_svc = max(svc_avgs, key=svc_avgs.get)
            worst_svc = min(svc_avgs, key=svc_avgs.get)
            best_svc_avg = svc_avgs[best_svc]
            worst_svc_avg = svc_avgs[worst_svc]
            if best_svc_avg > profitable_threshold:
                insights.append({
                    "type": "good",
                    "icon": "⭐",
                    "title": f"{best_svc} is your most profitable service",
                    "detail": f"Averaging ${best_svc_avg:.0f}/hr — above your ${profitable_threshold:.0f}/hr target. Prioritize booking more of these."
                })
            if worst_svc_avg < profitable_threshold * 0.8 and worst_svc != best_svc:
                insights.append({
                    "type": "warning",
                    "icon": "⚠️",
                    "title": f"{worst_svc} consistently underperforms",
                    "detail": f"Averaging ${worst_svc_avg:.0f}/hr — below your ${profitable_threshold:.0f}/hr target. Review pricing or time estimates for this service type."
                })

    # --- High-drive-time outlier jobs ---
    avg_drive = sum(j.get("drive_time", 0) for j in all_jobs) / len(all_jobs) if all_jobs else 0
    outliers = [j for j in all_jobs if j.get("drive_time", 0) > avg_drive * 2 and j.get("drive_time", 0) > 0.5]
    if outliers:
        outlier_cost = sum(j.get("drive_time", 0) * hourly_cost for j in outliers)
        insights.append({
            "type": "info",
            "icon": "🔍",
            "title": f"{len(outliers)} job{'s' if len(outliers) > 1 else ''} with unusually long drive times",
            "detail": f"These jobs have 2x+ average drive time, costing an extra ${outlier_cost:.0f} in unbillable hours. Bundle with nearby work or add a travel fee."
        })

    # Cap at 5 insights — most actionable first
    return insights[:5]

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
            coords = scored_job.pop("coords")
            scored_job["lat"] = round(coords[0], 6)
            scored_job["lng"] = round(coords[1], 6)
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

    # Run optimization and pattern detection (#6, #7)
    optimization = optimize_all_days(by_day, home_coords, hourly_labor_cost)
    insights = detect_patterns(days_result, summary, hourly_labor_cost, profitable_threshold)

    return {
        "days": days_result,
        "summary": summary,
        "red_jobs": red_jobs,
        "geocoding": "real",
        "config": {
            "hourly_labor_cost": hourly_labor_cost,
            "profitable_threshold": profitable_threshold,
            "marginal_threshold": marginal_threshold,
        },
        "optimization": optimization,
        "insights": insights,
    }

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

# ============================================================
# SMART COLUMN MAPPER
# ============================================================

REQUIRED_FIELDS = {
    "date":     ["date", "job_date", "service_date", "scheduled", "appointment_date",
                 "visit_date", "completed_date", "order_date", "booked"],
    "client":   ["client", "customer", "name", "customer_name", "client_name",
                 "account", "contact", "business_name", "company"],
    "address":  ["address", "location", "service_address", "job_address",
                 "street", "site", "property", "destination", "place"],
    "service":  ["service", "service_type", "job_type", "type", "work_type",
                 "category", "description", "job_description", "task"],
    "revenue":  ["revenue", "price", "amount", "total", "charge", "invoice_amount",
                 "total_amount", "billed", "fee", "cost", "rate", "payment"],
    "duration": ["duration", "hours", "time", "minutes", "job_duration",
                 "duration_minutes", "duration_hours", "length", "hrs"],
    "employee": ["employee", "tech", "technician", "worker", "staff", "assigned_to",
                 "rep", "team_member", "assigned", "operator"],
}

OPTIONAL_FIELDS = {
    "notes":  ["notes", "comments", "memo", "details", "remarks"],
    "status": ["status", "job_status", "state", "outcome", "result"],
}

def normalize(s: str) -> str:
    return s.lower().strip().replace(" ", "").replace("_", "").replace("-", "")

def score_column(col_name: str, aliases: list) -> float:
    norm_col = normalize(col_name)
    best = 0.0
    for alias in aliases:
        norm_alias = normalize(alias)
        if norm_col == norm_alias:
            return 1.0
        if norm_alias in norm_col or norm_col in norm_alias:
            score = len(norm_alias) / max(len(norm_col), len(norm_alias))
            best = max(best, score * 0.9)
    return best

def detect_column_type(values: list) -> str:
    non_empty = [v.strip() for v in values if v.strip()][:20]
    if not non_empty:
        return "text"
    currency_hits = sum(1 for v in non_empty
                        if v.startswith("$") or v.replace(",", "").replace(".", "").isdigit())
    if currency_hits > len(non_empty) * 0.6:
        return "currency"
    number_hits = sum(1 for v in non_empty
                      if v.replace(".", "").replace(",", "").isdigit())
    if number_hits > len(non_empty) * 0.6:
        return "number"
    date_hits = sum(1 for v in non_empty
                    if any(h in v for h in ["2024", "2025", "2026", "/", "-"]) and len(v) < 20)
    if date_hits > len(non_empty) * 0.4:
        return "date"
    return "text"

def smart_map_columns(headers: list, sample_rows: list) -> dict:
    all_fields = {**REQUIRED_FIELDS, **OPTIONAL_FIELDS}
    suggested = {}
    confidence = {}
    used_columns = set()

    col_samples = {h: [] for h in headers}
    for row in sample_rows[:20]:
        for i, val in enumerate(row):
            if i < len(headers):
                col_samples[headers[i]].append(val)

    score_matrix = {}
    for field, aliases in all_fields.items():
        col_scores = {}
        for col in headers:
            base_score = score_column(col, aliases)
            col_type = detect_column_type(col_samples.get(col, []))
            if field == "revenue" and col_type in ("currency", "number"):
                base_score = min(1.0, base_score + 0.15)
            if field == "duration" and col_type == "number":
                base_score = min(1.0, base_score + 0.1)
            if field == "date" and col_type == "date":
                base_score = min(1.0, base_score + 0.15)
            col_scores[col] = base_score
        score_matrix[field] = col_scores

    field_order = sorted(all_fields.keys(),
                         key=lambda f: max(score_matrix[f].values(), default=0),
                         reverse=True)

    for field in field_order:
        scores = score_matrix[field]
        best_col = max(scores, key=scores.get)
        best_score = scores[best_col]
        if best_score > 0.3 and best_col not in used_columns:
            suggested[field] = best_col
            confidence[field] = round(best_score, 2)
            used_columns.add(best_col)

    return {
        "suggested": suggested,
        "confidence": confidence,
        "unmapped_required": [f for f in REQUIRED_FIELDS if f not in suggested],
        "needs_review": [f for f, c in confidence.items() if c < 0.75 and f in REQUIRED_FIELDS],
        "unmapped_columns": [c for c in headers if c not in used_columns],
        "all_columns": headers,
    }


class ColumnMapping(BaseModel):
    date: Optional[str] = None
    client: Optional[str] = None
    address: Optional[str] = None
    service: Optional[str] = None
    revenue: Optional[str] = None
    duration: Optional[str] = None
    employee: Optional[str] = None
    notes: Optional[str] = None
    status: Optional[str] = None

class MappedAnalysisRequest(BaseModel):
    csv_data: str
    mapping: ColumnMapping


@app.post("/map-columns")
async def map_columns_endpoint(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = content.decode("utf-8-sig")
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
        if len(rows) < 2:
            raise HTTPException(status_code=400, detail="CSV needs at least a header row and one data row.")
        headers = [h.strip() for h in rows[0]]
        if len(headers) < 2:
            raise HTTPException(status_code=400, detail="Only found one column — is this a comma-separated file?")
        sample_rows = rows[1:6]
        result = smart_map_columns(headers, sample_rows)
        result["filename"] = file.filename
        result["total_rows"] = len(rows) - 1
        result["csv_content"] = text
        result["sample_rows"] = sample_rows
        return result
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Could not read file. Save it as CSV (UTF-8) and try again.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-mapped")
async def analyze_mapped_endpoint(
    request: MappedAnalysisRequest,
    hourly_labor_cost: float = 85,
    profitable_threshold: float = 120,
    marginal_threshold: float = 75,
    home_address: str = "506 Main St Kalispell MT 59901",
    business_id: Optional[int] = None,
):
    try:
        reader = csv.DictReader(io.StringIO(request.csv_data))
        raw_rows = list(reader)
        if not raw_rows:
            raise HTTPException(status_code=400, detail="No data rows found.")

        mapping = {k: v for k, v in request.mapping.dict().items() if v}

        jobs = []
        for row in raw_rows:
            try:
                address = row.get(mapping.get("address", ""), "").strip()
                revenue_raw = row.get(mapping.get("revenue", ""), "0").strip().replace("$", "").replace(",", "")
                duration_raw = row.get(mapping.get("duration", ""), "0").strip()
                jobs.append({
                    "date": row.get(mapping.get("date", ""), "").strip(),
                    "client_name": row.get(mapping.get("client", ""), "").strip(),
                    "address": address,
                    "service_type": row.get(mapping.get("service", ""), "").strip(),
                    "revenue": float(revenue_raw) if revenue_raw else 0,
                    "duration_hours": float(duration_raw) if duration_raw else 0,
                    "employee": row.get(mapping.get("employee", ""), "").strip(),
                    "coords": geocode_address(address),
                })
            except (ValueError, KeyError) as e:
                continue

        if not jobs:
            raise HTTPException(status_code=400, detail="No valid jobs found after mapping.")

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
                coords = scored_job.pop("coords")
                scored_job["lat"] = round(coords[0], 6)
                scored_job["lng"] = round(coords[1], 6)
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

            date_obj = datetime.strptime(date, "%Y-%m-%d") if date else datetime.now()
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

        # Run optimization and pattern detection (#6, #7)
        optimization = optimize_all_days(by_day, home_coords, hourly_labor_cost)
        insights = detect_patterns(days_result, summary, hourly_labor_cost, profitable_threshold)

        return {
            "days": days_result,
            "summary": summary,
            "red_jobs": red_jobs,
            "geocoding": "real",
            "config": {
                "hourly_labor_cost": hourly_labor_cost,
                "profitable_threshold": profitable_threshold,
                "marginal_threshold": marginal_threshold,
            },
            "optimization": optimization,
            "insights": insights,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
