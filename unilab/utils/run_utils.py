import os

def get_latest_run(log_dir: str) -> str | None:
    """Find the latest run in the log directory that contains a model.
    
    Args:
        log_dir: Path to the base log directory (e.g., logs/fast_sac_Go2LocoFlatTerrain)
        
    Returns:
        Path to the latest run directory containing a model, or None if none found.
    """
    if not os.path.exists(log_dir):
        return None
    runs = sorted([d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d)) and d != "git"])
    
    # Iterate backwards to find first run with models
    for run_id in reversed(runs):
        run_path = os.path.join(log_dir, run_id)
        # Check if any .pt file exists
        if any(f.endswith(".pt") for f in os.listdir(run_path)):
            return run_path
            
    # Fallback to just the latest directory if no models found
    if len(runs) > 0:
        return os.path.join(log_dir, runs[-1])
    return None
