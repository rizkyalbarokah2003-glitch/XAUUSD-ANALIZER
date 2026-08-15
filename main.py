import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from openai import OpenAI

APP_DIR = Path(__file__).resolve().parent
FRONTEND = APP_DIR / "XAU_ANALYZER_AI_v4_LIVE_HP.html"

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
MAX_BYTES = 10 * 1024 * 1024

app = FastAPI(title="XAU Analyzer AI v4")

SCHEMA = {
    "type": "object",
    "properties": {
        "signal": {"type": "string", "enum": ["BUY", "SELL", "WAIT"]},
        "trend": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 100},
        "entry_zone": {"type": "string"},
        "stop_loss": {"type": "string"},
        "tp1": {"type": "string"},
        "tp2": {"type": "string"},
        "risk_reward": {"type": "string"},
        "reason": {"type": "string"},
        "invalidation": {"type": "string"},
        "readable": {"type": "boolean"},
        "confirmations": {
            "type": "object",
            "properties": {
                "structure": {"type": "boolean"},
                "sr": {"type": "boolean"},
                "supply_demand": {"type": "boolean"},
                "liquidity": {"type": "boolean"},
                "momentum": {"type": "boolean"},
                "candle": {"type": "boolean"},
                "mtf": {"type": "boolean"},
            },
            "required": [
                "structure", "sr", "supply_demand", "liquidity",
                "momentum", "candle", "mtf"
            ],
            "additionalProperties": False,
        },
    },
    "required": [
        "signal", "trend", "confidence", "entry_zone", "stop_loss",
        "tp1", "tp2", "risk_reward", "reason", "invalidation",
        "readable", "confirmations"
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """
You are XAU/USD Chart Analyst AI. Analyze only the supplied chart screenshot.
Do not invent prices, levels, candles, indicators, or timeframe information that
cannot be read from the image.

Your job is to identify a probable setup, not to guarantee profit. Be conservative.
If the chart is blurry, cropped, contradictory, missing price scale, or does not
provide enough evidence for a responsible setup, set readable=false and signal=WAIT.

Evaluate these seven independent confirmations:
1) market structure
2) support/resistance
3) supply/demand
4) liquidity / sweep / stop-hunt context
5) momentum
6) candle confirmation
7) multi-timeframe context (only mark true if the screenshot actually provides
   enough MTF evidence; otherwise false)

Important:
- XAU/USD only.
- Distinguish BUY, SELL, and WAIT.
- A trade idea should have a clear invalidation.
- Never manufacture exact entry/SL/TP numbers. If a level is not readable, say
  "N/A - level tidak terbaca".
- If evidence is mixed, prefer WAIT.
- Return concise Indonesian text.
- The final score is calculated by the backend from the seven confirmations,
  so do not try to manipulate the score.
"""

WEIGHTS = {
    "structure": 20,
    "sr": 15,
    "supply_demand": 15,
    "liquidity": 15,
    "momentum": 10,
    "candle": 10,
    "mtf": 15,
}


def score_confirmations(c: dict[str, Any]) -> int:
    return sum(weight for key, weight in WEIGHTS.items() if c.get(key) is True)


def normalize_result(data: dict[str, Any]) -> dict[str, Any]:
    c = data.get("confirmations") or {}
    score = score_confirmations(c)

    if not data.get("readable", False):
        signal = "WAIT"
    else:
        signal = str(data.get("signal", "WAIT")).upper()
        if signal not in {"BUY", "SELL", "WAIT"}:
            signal = "WAIT"

    # Conservative gate: a directional call needs enough independent evidence.
    if score < 70:
        signal = "WAIT"

    confidence = float(data.get("confidence", 0))
    confidence = max(0, min(100, round(confidence)))
    confidence = min(confidence, score)

    if signal == "WAIT":
        entry = "N/A - tunggu konfirmasi"
        sl = "N/A"
        tp1 = "N/A"
        tp2 = "N/A"
        rr = "N/A"
    else:
        entry = data.get("entry_zone", "N/A")
        sl = data.get("stop_loss", "N/A")
        tp1 = data.get("tp1", "N/A")
        tp2 = data.get("tp2", "N/A")
        rr = data.get("risk_reward", "N/A")

    return {
        "signal": signal,
        "score": score,
        "trend": data.get("trend", "Tidak jelas"),
        "confidence": confidence,
        "entry_zone": entry,
        "stop_loss": sl,
        "tp1": tp1,
        "tp2": tp2,
        "risk_reward": rr,
        "reason": data.get("reason", "Bukti belum cukup."),
        "invalidation": data.get("invalidation", "Setup batal jika struktur utama ditembus."),
        "confirmations": c,
    }


@app.get("/health")
def health():
    return {
        "ok": bool(os.getenv("OPENAI_API_KEY")),
        "service": "xau-analyzer-ai",
        "model": MODEL,
    }


@app.get("/")
def home():
    if not FRONTEND.exists():
        raise HTTPException(404, "Frontend HTML belum ada di repository.")
    return FileResponse(FRONTEND, media_type="text/html")


@app.post("/api/analyze")
async def analyze(
    chart: UploadFile = File(...),
    timeframe: str = Form("M5"),
):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(503, "OPENAI_API_KEY belum dikonfigurasi di server.")

    content_type = chart.content_type or ""
    if content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(400, "Format gambar harus JPG, PNG, atau WEBP.")

    raw = await chart.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "Ukuran gambar maksimal 10 MB.")

    b64 = base64.b64encode(raw).decode("ascii")
    data_url = f"data:{content_type};base64,{b64}"

    client = OpenAI(api_key=api_key)

    user_prompt = f"""
Timeframe yang dipilih pengguna: {timeframe}

Analisis screenshot XAU/USD ini. Fokus pada price action yang benar-benar terlihat.
Jika timeframe atau level tidak terbaca dari gambar, jangan menebak.
"""

    try:

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "xau_analysis",
                    "strict": True,
                    "schema": SCHEMA,
                },
            },
        )
    except Exception as exc:
        raise HTTPException(502, f"AI request gagal: {exc}") from exc

    try:
        parsed = json.loads(response.choices[0].message.content)
    except Exception as exc:
        raise HTTPException(502, "AI mengembalikan format yang tidak valid.") from exc

    return normalize_result(parsed)
