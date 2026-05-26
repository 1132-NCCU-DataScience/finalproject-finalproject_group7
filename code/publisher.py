import json
import subprocess
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from loguru import logger
import pandas as pd

TZ_TPE = timezone(timedelta(hours=8))

def get_current_iso_time() -> str:
    return datetime.now(TZ_TPE).isoformat()

def save_latest_json_local(
    pred_df: pd.DataFrame = None, 
    model_version: str = "unknown", 
    output_dir: str | Path = "predictions",
    is_success: bool = True
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "latest.json"
    
    current_time = get_current_iso_time()
    
    old_data = {}
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                old_data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read old latest.json: {e}")

    if is_success and pred_df is not None and not pred_df.empty:
        health_status = "ok"
        last_success_at = current_time
        update_time = str(pred_df["timestamp"].max()) if "timestamp" in pred_df.columns else current_time
        
        predictions_list = []
        for _, row in pred_df.iterrows():
            item = {
                "sno": str(row["sno"]),
                "lat": float(row.get("lat", row.get("latitude", 0.0))), 
                "lng": float(row.get("lng", row.get("longitude", 0.0))),
                "shortage_rate": float(row["shortage_rate"]),
                "available_bikes": int(row["available_bikes"]),
                "total_capacity": int(row["total_capacity"]),
                "moran_type": str(row.get("moran_type", "NS")),
                "moran_p_value": float(row.get("moran_p_value", 1.0)),
                "pred_prob": float(row["pred_prob"]),
            }
            predictions_list.append(item)
    else:
        last_success_at = old_data.get("last_success_at", current_time)
        update_time = old_data.get("update_time", last_success_at)
        predictions_list = old_data.get("predictions", [])
        model_version = old_data.get("model_version", model_version)
        
        try:
            last_success_dt = datetime.fromisoformat(last_success_at)
            delay_mins = (datetime.now(TZ_TPE) - last_success_dt).total_seconds() / 60
            if delay_mins < 15:
                health_status = "ok"
            elif delay_mins < 30:
                health_status = "stale"
            else:
                health_status = "degraded"
        except ValueError:
            health_status = "degraded"
            logger.error("Timestamp parsing error for fallback.")

    output_payload = {
        "model_version": model_version,
        "update_time": update_time,
        "last_attempt_at": current_time,
        "last_success_at": last_success_at,
        "health_status": health_status,
        "n_stations": len(predictions_list),
        "predictions": predictions_list
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)
    logger.info(f"[Publisher] Successfully saved local JSON to {json_path} (Status: {health_status})")

    _git_commit_and_push(output_path.parent)
    return json_path

def _git_commit_and_push(repo_path: str, max_retries: int = 2):
    commands = [
        ["git", "add", "predictions/latest.json"],
        ["git", "commit", "-m", f"chore: auto-update predictions {get_current_iso_time()}"],
        ["git", "push", "origin", "main"]
    ]
    
    for cmd in commands:
        if cmd[1] == "push":
            for attempt in range(max_retries + 1):
                result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True)
                if result.returncode == 0:
                    logger.success("Git Push successful!")
                    break
                else:
                    logger.warning(f"Git Push failed (Attempt {attempt+1}/{max_retries+1}): {result.stderr}")
                    if attempt < max_retries:
                        time.sleep(2 ** attempt)
            else:
                logger.error("Git Push ultimately failed. Retaining local changes for the next update cycle.")
        else:
            subprocess.run(cmd, cwd=repo_path, capture_output=True)