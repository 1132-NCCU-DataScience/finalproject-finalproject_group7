import json
import subprocess
import time
from datetime import datetime, timezone, timedelta
from loguru import logger
import pandas as pd

TZ_TPE = timezone(timedelta(hours=8))

def get_current_iso_time() -> str:
    return datetime.now(TZ_TPE).isoformat()

def publish_predictions(
    df_predictions: pd.DataFrame, 
    model_version: str, 
    repo_path: str = ".",
    is_success: bool = True
):
    json_path = f"{repo_path}/predictions/latest.json"
    current_time = get_current_iso_time()
    
    old_data = {}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            old_data = json.load(f)
    except FileNotFoundError:
        logger.warning("latest.json not found. A new file will be created.")

    if is_success:
        health_status = "ok"
        last_success_at = current_time
        predictions_list = df_predictions.to_dict(orient="records")
    else:
        last_success_at = old_data.get("last_success_at", current_time)
        predictions_list = old_data.get("predictions", [])
        
        last_success_dt = datetime.fromisoformat(last_success_at)
        delay_mins = (datetime.now(TZ_TPE) - last_success_dt).total_seconds() / 60
        health_status = "stale" if delay_mins < 30 else "degraded"

    output_payload = {
        "model_version": model_version,
        "update_time": last_success_at,
        "last_attempt_at": current_time,
        "last_success_at": last_success_at,
        "health_status": health_status,
        "n_stations": len(predictions_list),
        "predictions": predictions_list
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)
    logger.info(f"Successfully serialized data to {json_path}")

    _git_commit_and_push(repo_path)


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