import math
import time
from datetime import datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def get_weather_data(api_key: str, city: str) -> dict | None:
    """
    Fetch current weather from OpenWeatherMap and derive a physically
    correct irradiation value (0.0 at night, bell-curve during the day,
    reduced by cloud cover).

    Returns a dict or None on any failure – never raises.
    """
    if not api_key:
        return None

    try:
        # ── Resilient HTTP session ─────────────────────────────────────────
        session = requests.Session()
        retry   = Retry(
            total=2,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://",  adapter)
        session.mount("https://", adapter)

        url    = "http://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": api_key, "units": "metric"}
        resp   = session.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data   = resp.json()

        # ── Parse fields ───────────────────────────────────────────────────
        temp       = data["main"]["temp"]
        humidity   = data["main"]["humidity"]
        clouds_pct = data["clouds"]["all"]           # 0–100
        sunrise_ts = data["sys"]["sunrise"]          # unix
        sunset_ts  = data["sys"]["sunset"]           # unix
        description= data["weather"][0]["description"].capitalize()

        # ── Irradiation: physically correct ───────────────────────────────
        now_ts = time.time()
        if now_ts <= sunrise_ts or now_ts >= sunset_ts:
            irradiation = 0.0                        # nighttime
        else:
            day_len      = sunset_ts - sunrise_ts
            elapsed      = now_ts   - sunrise_ts
            solar_pos    = math.sin(math.pi * elapsed / day_len)   # 0→1→0
            cloud_factor = 1.0 - (clouds_pct / 100.0) * 0.75
            irradiation  = round(max(0.0, min(1.0, solar_pos * cloud_factor)), 3)

        return {
            "temperature":  round(temp, 2),
            "humidity":     humidity,
            "irradiation":  irradiation,
            "clouds_pct":   clouds_pct,
            "description":  description,
            "city":         data.get("name", city),
            "fetched_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    except Exception:
        return None
