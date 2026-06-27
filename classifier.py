import os
import json
from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

CLASSIFIER_MODEL = "gemini-1.5-flash"  # Small/fast model for low-latency classification

CLASSIFIER_PROMPT = """
You are a fast failure classifier for data center GPU telemetry (DCGM-style metrics).

Telemetry batch (JSON):
{telemetry}

Classify if there is a failure and its primary type.
Output ONLY valid JSON:
{
  "failure": boolean,
  "type": "normal | overheating | voltage_instability | ecc_error | power_quality | dust_proxy | humidity | combined | other",
  "confidence": float (0-1),
  "reason": "short explanation",
  "suggested_next": "brief recommendation for analyzer"
}
"""

def classify_telemetry(telemetry: dict) -> dict:
    """Classify a telemetry batch using small Gemini model."""
    prompt = CLASSIFIER_PROMPT.format(telemetry=json.dumps(telemetry, indent=2))
    response = client.models.generate_content(
        model=CLASSIFIER_MODEL,
        contents=prompt,
        generation_config={"temperature": 0.1, "response_mime_type": "application/json"}
    )
    try:
        return json.loads(response.text)
    except:
        return {"failure": False, "type": "normal", "confidence": 0.5, "reason": "parse error", "suggested_next": "monitor"}

if __name__ == "__main__":
    # Test with sample
    sample = {"time": 10, "servers": [{"id": 0, "gpu_util": 95, "gpu_temp": 85, "power_draw": 450, "ecc_errors": 5}]}
    print(classify_telemetry(sample))