# agromet_advisory.py
# Partha Pratim Ray, 12/06/2025
# parthapratimray1986@gmail.com


import gradio as gr
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
from datetime import date, timedelta
import math
import sqlite3

GEOCODE_API   = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_API   = "https://api.open-meteo.com/v1/forecast"
OLLAMA_BASE   = "http://localhost:11434"
DB_PATH       = "agromet_et.sqlite"

def get_ollama_models():
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=10)
        tags = r.json().get("models", [])
        return [m["name"] for m in tags] if tags else []
    except Exception:
        return []

def show_model_info(model):
    try:
        r = requests.post(f"{OLLAMA_BASE}/api/show", json={"model": model}, timeout=30)
        data = r.json()
        family = data.get("details", {}).get("family", "")
        param_size = data.get("details", {}).get("parameter_size", "")
        quant = data.get("details", {}).get("quantization_level", "")
        arch = data.get("model_info", {}).get("general.architecture", "")
        return f"**Family**: {family} | **Parameters**: {param_size} | **Quantization**: {quant} | **Arch**: {arch}"
    except Exception:
        return "Model info not available."

def geocode(name, count=5):
    params = {"name": name, "count": count, "language": "en", "format": "json"}
    r = requests.get(GEOCODE_API, params=params, timeout=10)
    r.raise_for_status()
    return [
        (res["name"], res["latitude"], res["longitude"], res.get("country", ""))
        for res in r.json().get("results", [])
    ]

def fetch_weather(lat, lon, days=5):
    start = date.today()
    end = start + timedelta(days=days-1)
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": (
            "temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
            "relative_humidity_2m_mean,precipitation_sum,et0_fao_evapotranspiration,"
            "shortwave_radiation_sum"
        ),
        "timezone": "auto",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    resp = requests.get(WEATHER_API, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()["daily"]

def calc_ra(lat, doy):
    lat_rad = math.radians(lat)
    dr = 1 + 0.033 * math.cos(2 * math.pi / 365 * doy)
    delta = 0.409 * math.sin(2 * math.pi / 365 * doy - 1.39)
    ws = math.acos(-math.tan(lat_rad) * math.tan(delta))
    ra = (24 * 60 / math.pi) * 0.0820 * dr * (
        ws * math.sin(lat_rad) * math.sin(delta) +
        math.cos(lat_rad) * math.cos(delta) * math.sin(ws)
    )
    return ra

def compute_et_metrics(df, lat):
    df["doy"] = pd.to_datetime(df["time"]).dt.dayofyear
    df["Ra"] = df["doy"].apply(lambda doy: calc_ra(lat, doy))
    df["Tavg"] = (df["temperature_2m_max"] + df["temperature_2m_min"]) / 2

    df["ET_Hargreaves_Samani"] = (
        0.0023 * (df["Tavg"] + 17.8)
        * (df["temperature_2m_max"] - df["temperature_2m_min"]).pow(0.5)
        * df["Ra"]
    )

    if {"temperature_2m_mean", "relative_humidity_2m_mean", "shortwave_radiation_sum"}.issubset(df.columns):
        T = df["temperature_2m_mean"]
        RH = df["relative_humidity_2m_mean"]
        Rs = df["shortwave_radiation_sum"]
        df["ET_Turc"] = (0.013 * (T + 15) / (T + 15 + 50 * (1 - RH / 100))) * Rs
    else:
        df["ET_Turc"] = np.nan

    alpha_PT = 1.26
    def delta_svp(T):
        return 4098 * (0.6108 * math.exp(17.27 * T / (T + 237.3))) / (T + 237.3) ** 2
    def gamma(P=101.3):
        return 0.665e-3 * P

    df["delta"] = df["Tavg"].apply(delta_svp)
    P = 101.3
    g = gamma(P)
    if "shortwave_radiation_sum" in df.columns:
        Rn = df["shortwave_radiation_sum"]
        df["ET_PriestleyTaylor"] = alpha_PT * df["delta"] / (df["delta"] + g) * (Rn * 0.408)
        df["ET_Makkink"] = 0.61 * df["delta"] / (df["delta"] + g) * (df["shortwave_radiation_sum"] * 0.408)
    else:
        df["ET_PriestleyTaylor"] = np.nan
        df["ET_Makkink"] = np.nan
    return df

def store_to_sqlite(df, db_path, location, lat, lon, llm_advice, llm_model, 
                    total_duration, load_duration, prompt_eval_count, 
                    prompt_eval_duration, eval_count, eval_duration, tokens_per_second):
    df.insert(1, "location", location)
    df.insert(2, "lat", lat)
    df.insert(3, "lon", lon)
    df["ollama_model"] = llm_model
    df["total_duration"] = total_duration
    df["load_duration"] = load_duration
    df["prompt_eval_count"] = prompt_eval_count
    df["prompt_eval_duration"] = prompt_eval_duration
    df["eval_count"] = eval_count
    df["eval_duration"] = eval_duration
    df["tokens_per_second"] = tokens_per_second
    df["llm_advice"] = llm_advice
    conn = sqlite3.connect(db_path)
    df.to_sql('et_metrics', conn, if_exists='append', index=False)
    conn.close()

def prepare_summary(df):
    lines = []
    for r in df.itertuples():
        lines.append(
            f"{r.time[:10]}: Tmax={r.temperature_2m_max:.1f}C, "
            f"Tmin={r.temperature_2m_min:.1f}C, "
            f"Rain={r.precipitation_sum:.1f}mm, "
            f"ET0={r.et0_fao_evapotranspiration:.2f}mm, "
            f"HS={r.ET_Hargreaves_Samani:.2f}mm, "
            f"Turc={r.ET_Turc:.2f}mm, "
            f"PT={r.ET_PriestleyTaylor:.2f}mm, "
            f"Makkink={r.ET_Makkink:.2f}mm"
        )
    return "\n".join(lines)

def plot_et_comparison(df, loc_name, lat, lon):
    labels = pd.to_datetime(df["time"]).dt.strftime('%Y-%m-%d')
    x = np.arange(len(labels))
    width = 0.16

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = []
    bars.append(ax.bar(x - 2*width, df["et0_fao_evapotranspiration"], width, label="FAO ET₀ (API)"))
    bars.append(ax.bar(x - width, df["ET_Hargreaves_Samani"], width, label="HS"))
    bars.append(ax.bar(x, df["ET_Turc"], width, label="Turc"))
    bars.append(ax.bar(x + width, df["ET_PriestleyTaylor"], width, label="Priestley-Taylor"))
    bars.append(ax.bar(x + 2*width, df["ET_Makkink"], width, label="Makkink"))

    for group in bars:
        for bar in group:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.1,
                        f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_xlabel("Date")
    ax.set_ylabel("Evapotranspiration (mm day⁻¹)")
    ax.set_title(f"Comparative ET Models Forecasting – {loc_name} ({lat:.4f}, {lon:.4f})")
    ax.legend()
    plt.tight_layout()
    return fig

def ollama_advice(loc_name, lat, lon, summary, model):
    num_days = len([line for line in summary.splitlines() if line.strip()])

    # Few-shot EXAMPLES (3, 5, 7 days)
    ex3_data = (
        "2024-06-12: Tmax=34.5C, Tmin=20.1C, Rain=5.0mm, ET0=4.2mm, HS=3.8mm, Turc=3.2mm, PT=4.0mm, Makkink=3.6mm\n"
        "2024-06-13: Tmax=36.0C, Tmin=21.0C, Rain=0.0mm, ET0=5.2mm, HS=4.1mm, Turc=3.5mm, PT=4.4mm, Makkink=3.9mm\n"
        "2024-06-14: Tmax=30.2C, Tmin=18.9C, Rain=55.0mm, ET0=3.7mm, HS=3.3mm, Turc=2.9mm, PT=3.4mm, Makkink=3.1mm"
    )
    ex3_output = """1. Irrigation plan: Irrigate 10 mm on day-1 morning. Skip irrigation if rain comes on day-3.
2. Crop / soil action: Mulch after day-1 harvest to keep soil moist.
3. Livestock / labour: Give shade to cows and goats on hot day-2.
4. Pest & disease watch: Watch for leaf spot after day-3 rain; spray neem water.
5. Input-saving tip: Use drip irrigation to save water and electricity.
Forecast from open-meteo + ET models. Moderate reliability."""

    ex5_data = (
        "2024-06-15: Tmax=28.5C, Tmin=17.2C, Rain=0.0mm, ET0=3.5mm, HS=2.8mm, Turc=2.4mm, PT=3.1mm, Makkink=2.7mm\n"
        "2024-06-16: Tmax=31.0C, Tmin=19.0C, Rain=12.0mm, ET0=4.0mm, HS=3.1mm, Turc=2.8mm, PT=3.6mm, Makkink=3.0mm\n"
        "2024-06-17: Tmax=35.5C, Tmin=21.0C, Rain=2.0mm, ET0=5.5mm, HS=4.5mm, Turc=3.6mm, PT=4.8mm, Makkink=4.2mm\n"
        "2024-06-18: Tmax=26.5C, Tmin=16.0C, Rain=40.0mm, ET0=3.2mm, HS=2.5mm, Turc=2.0mm, PT=2.8mm, Makkink=2.3mm\n"
        "2024-06-19: Tmax=29.0C, Tmin=18.0C, Rain=0.0mm, ET0=4.1mm, HS=3.2mm, Turc=2.7mm, PT=3.5mm, Makkink=3.0mm"
    )
    ex5_output = """1. Irrigation plan: Give 12 mm irrigation on day-2 morning. No irrigation needed on day-4 due to rain.
2. Crop / soil action: Mulch after harvest on day-1 to reduce water loss.
3. Livestock / labour: Keep cattle under shade on hot day-3 afternoon.
4. Pest & disease watch: Watch for fungal disease after day-4 rain; apply organic spray if needed.
5. Input-saving tip: Use sprinkler to save water on dry days.
Forecast from open-meteo + ET models. Moderate reliability."""

    ex7_data = (
        "2024-06-20: Tmax=33.0C, Tmin=21.2C, Rain=2.0mm, ET0=4.8mm, HS=4.0mm, Turc=3.1mm, PT=4.3mm, Makkink=3.7mm\n"
        "2024-06-21: Tmax=35.2C, Tmin=22.5C, Rain=0.0mm, ET0=5.3mm, HS=4.5mm, Turc=3.7mm, PT=4.9mm, Makkink=4.2mm\n"
        "2024-06-22: Tmax=29.5C, Tmin=19.0C, Rain=50.0mm, ET0=3.4mm, HS=2.8mm, Turc=2.1mm, PT=3.2mm, Makkink=2.5mm\n"
        "2024-06-23: Tmax=27.0C, Tmin=18.1C, Rain=60.0mm, ET0=3.0mm, HS=2.4mm, Turc=1.7mm, PT=2.8mm, Makkink=2.1mm\n"
        "2024-06-24: Tmax=31.1C, Tmin=20.8C, Rain=10.0mm, ET0=4.1mm, HS=3.3mm, Turc=2.6mm, PT=3.6mm, Makkink=3.0mm\n"
        "2024-06-25: Tmax=32.5C, Tmin=21.0C, Rain=0.0mm, ET0=4.7mm, HS=3.9mm, Turc=3.0mm, PT=4.2mm, Makkink=3.6mm\n"
        "2024-06-26: Tmax=28.8C, Tmin=19.2C, Rain=35.0mm, ET0=3.6mm, HS=2.9mm, Turc=2.3mm, PT=3.3mm, Makkink=2.7mm"
    )
    ex7_output = """1. Irrigation plan: Give 10 mm irrigation on day-1 and day-6 if no rain. No irrigation needed on day-3 and day-4 due to heavy rain.
2. Crop / soil action: Mulch after day-2 to keep soil moist; ensure good drainage before heavy rain on day-3 and day-4.
3. Livestock / labour: Provide shade and cool water for cattle on day-2. Shelter poultry from rain on day-3 and day-4.
4. Pest & disease watch: Watch for stem rot after continuous rain on day-3 and day-4; use organic fungicide on day-5.
5. Input-saving tip: Use rainwater harvesting for irrigation on dry days. Check pump for leaks to save energy.
Forecast from open-meteo + ET models. Moderate reliability."""

    prompt = f"""
You are an agrometeorology advisor for rural Indian farmers.
Follow this structure for your answer, using the sample categories and numbers below.

Always give advice in exactly this order:
1. Irrigation plan – quote how many mm and which days to irrigate (from ET and rain)
2. Crop / soil action – mulch, drainage, sow/harvest shift, etc.
3. Livestock / labour – heat/cold precautions
4. Pest & disease watch – link humidity/rain to likely outbreaks, give one preventive tip
5. Input-saving tip – fertigation, water/energy/labour saving

Use short, simple sentences and numbers, as in the examples.

=== EXAMPLE 1: 3 days ===
DATA:
{ex3_data}

ADVICE:
{ex3_output}

=== EXAMPLE 2: 5 days ===
DATA:
{ex5_data}

ADVICE:
{ex5_output}

=== EXAMPLE 3: 7 days ===
DATA:
{ex7_data}

ADVICE:
{ex7_output}

=== NOW YOUR TURN ===
DATA:
{summary}

Write exactly 5 points, each starting with the category as above and finish with:
Forecast from open-meteo + ET models. Moderate reliability.
"""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "20m"
    }
    try:
        r = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=900)
        r.raise_for_status()
        resp = r.json()
        model_name = resp.get("model", model)
        message = resp.get("response", "").strip()
        total_duration = resp.get("total_duration", None)
        load_duration = resp.get("load_duration", None)
        prompt_eval_count = resp.get("prompt_eval_count", None)
        prompt_eval_duration = resp.get("prompt_eval_duration", None)
        eval_count = resp.get("eval_count", None)
        eval_duration = resp.get("eval_duration", None)
        try:
            tokens_per_second = (eval_count / eval_duration * 1e9) if (eval_count and eval_duration and eval_duration > 0) else None
        except Exception:
            tokens_per_second = None
        return dict(
            advice=message,
            ollama_model=model_name,
            total_duration=total_duration,
            load_duration=load_duration,
            prompt_eval_count=prompt_eval_count,
            prompt_eval_duration=prompt_eval_duration,
            eval_count=eval_count,
            eval_duration=eval_duration,
            tokens_per_second=tokens_per_second
        )
    except requests.exceptions.ReadTimeout:
        return dict(advice="⚠️ Ollama request timed out. Please try again in a moment.")
    except Exception as e:
        return dict(advice=f"⚠️ Error generating advice: {e}")

def preload_model(model):
    """Preload Ollama model using a null prompt (no database logging)."""
    try:
        requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": model, "prompt": " ", "keep_alive": "10m", "stream": False},
            timeout=60
        )
    except Exception:
        pass

def unload_previous_model(new_model, last_model):
    if last_model and last_model != new_model:
        try:
            requests.post(
                f"{OLLAMA_BASE}/api/generate",
                json={"model": last_model, "prompt": " ", "keep_alive": 0, "stream": False},
                timeout=30
            )
        except Exception:
            pass
    preload_model(new_model)  # Only preloads, never stores to SQLite
    return show_model_info(new_model), new_model

def step_geocode(location):
    try:
        matches = geocode(location)
        if not matches:
            return (gr.update(choices=[], visible=True), 
                    gr.update(visible=False), 
                    None, 
                    gr.update(value="❌ Location not found. Please check spelling or try another location.", visible=True))
        labels = [f"{i+1}. {nm}, {co} ({la:.4f}, {lo:.4f})" for i, (nm, la, lo, co) in enumerate(matches)]
        return (gr.update(choices=labels, visible=True), 
                gr.update(visible=False), 
                matches, 
                gr.update(value="", visible=False))
    except Exception as e:
        return (gr.update(choices=[], visible=True), 
                gr.update(visible=False), 
                None, 
                gr.update(value="❌ Error searching location. Please try again.", visible=True))

def step_pick(idx):
    return gr.update(visible=True)

def step_process(selected_idx, matches, days, ollama_model):
    idx = int(selected_idx.split('.', 1)[0].strip()) - 1
    loc_name, lat, lon, _ = matches[idx]
    daily = fetch_weather(lat, lon, days=days)
    df = pd.DataFrame(daily)
    df = compute_et_metrics(df, lat)
    summary = prepare_summary(df)
    ollama_result = ollama_advice(loc_name, lat, lon, summary, ollama_model)
    advice = ollama_result.get("advice", "")
    # Only store real prompt runs, never from model preloads or unloads
    store_to_sqlite(
        df, DB_PATH, loc_name, lat, lon, advice,
        ollama_result.get("ollama_model", ollama_model),
        ollama_result.get("total_duration", None),
        ollama_result.get("load_duration", None),
        ollama_result.get("prompt_eval_count", None),
        ollama_result.get("prompt_eval_duration", None),
        ollama_result.get("eval_count", None),
        ollama_result.get("eval_duration", None),
        ollama_result.get("tokens_per_second", None)
    )
    info = show_model_info(ollama_model)
    fig = plot_et_comparison(df, loc_name, lat, lon)
    return (
        gr.update(value=df[[
            "location", "lat", "lon", "time", "temperature_2m_max", "temperature_2m_min",
            "precipitation_sum", "et0_fao_evapotranspiration", "ET_Hargreaves_Samani",
            "ET_Turc", "ET_PriestleyTaylor", "ET_Makkink"
        ]], visible=True),
        fig,
        info,
        advice
    )

with gr.Blocks() as demo:
    gr.Markdown("# AgroMetLLM: Evapotranspiration & Agro-Advisory in Raspberry Pi 4B by Using Ollama, Local LLMs, and Open-Meteo API")
    with gr.Row():
        models = get_ollama_models()
        ollama_model = gr.Dropdown(label="Select LLM Model", choices=models, value=(models[0] if models else None))
        location = gr.Textbox(label="Location (district/city)", placeholder="e.g. Gangtok")
        days = gr.Slider(3, 7, value=5, step=1, label="Forecast days")
    model_info = gr.Markdown("Model info will appear here.")
    search_btn = gr.Button("Search Location")
    location_pick = gr.Dropdown(label="Select exact location (if multiple)", visible=False)
    submit_btn = gr.Button("Generate ET Forecast & Local LLM Advisory", visible=False)
    out_table = gr.Dataframe(headers=None, label="Evapotranspiration Metrics (mm/day)", visible=False)
    out_plot = gr.Plot(label="Comparative ET Models Forecasting Chart")
    out_advice = gr.Textbox(label="Farmer Advisory (LLM Output)", interactive=False)
    matches_state = gr.State([])
    last_model_state = gr.State(None)
    err_msg = gr.Markdown("", visible=False)

    ollama_model.change(
        unload_previous_model,
        inputs=[ollama_model, last_model_state],
        outputs=[model_info, last_model_state]
    )

    search_btn.click(
        fn=step_geocode,
        inputs=[location],
        outputs=[location_pick, submit_btn, matches_state, err_msg],
    )
    location_pick.change(
        step_pick, location_pick, submit_btn
    )
    submit_btn.click(
        fn=step_process,
        inputs=[location_pick, matches_state, days, ollama_model],
        outputs=[out_table, out_plot, model_info, out_advice]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
