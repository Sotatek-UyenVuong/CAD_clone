import requests
import time
import json
import os
from dotenv import load_dotenv

load_dotenv()
url = "https://www.datalab.to/api/v1/marker"

payload = {
    "file_url": "https://vna-minio.dev.sotaagents.ai/vna-public/3.jpeg",
    "page_range": "0-1",
    "langs": "ja",
    "force_ocr": "true",
    "format_lines": "false",
    "paginate": "false",
    "strip_existing_ocr": "false",
    "disable_image_extraction": "true",
    "disable_ocr_math": "true",
    "use_llm": "false",
    "mode": "fast",
    "output_format": "markdown",
    "skip_cache": "false",
    "save_checkpoint": "false",
}
headers = {"X-API-Key": os.getenv("MARKER_API_KEY")}

response = requests.post(url, data=payload, headers=headers, timeout=60)

try:
    resp_json = response.json()
except Exception:
    resp_json = {"detail": "Non-JSON response", "status_code": response.status_code, "text": response.text[:500]}

print(resp_json)

if not response.ok or "request_check_url" not in resp_json:
    raise SystemExit(f"Marker request failed or missing check URL: {resp_json}")

max_polls = 300
check_url = resp_json["request_check_url"]

for i in range(max_polls):
    time.sleep(2)
    response = requests.get(check_url, headers=headers) # Don't forget to send the auth headers
    data = response.json()
    print(data)
    save_dir = "/Users/uyenvuong/Downloads/chunks_offline/data"
    os.makedirs(save_dir, exist_ok=True)

    # Save each poll as readable JSON to avoid escaped Unicode
    json_path = f"{save_dir}/marker_{i}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    if data.get("status") == "complete":
        # When complete and output is markdown, save raw markdown to .md
        if data.get("output_format") == "markdown" and data.get("markdown") is not None:
            md_path = f"{save_dir}/marker_{i}.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(data["markdown"])
        break