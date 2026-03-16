import json, logging, sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

def setup_logging(log_dir: str, run_name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    log_dir = Path(log_dir); log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{run_name or 'uagmf'}_{ts}.log"
    fmt = "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s"
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)]
    logging.basicConfig(level=level, format=fmt, datefmt="%Y-%m-%d %H:%M:%S", handlers=handlers)
    return logging.getLogger("uagmf")

class ExperimentLogger:
    def __init__(self, output_dir: str, run_name: str = "uagmf") -> None:
        self.output_dir = Path(output_dir); self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self._h: Dict[str, Any] = {"run_name": run_name, "start_time": datetime.now().isoformat(), "epochs": [], "final": {}}
    def log_epoch(self, epoch: int, metrics: Dict) -> None: self._h["epochs"].append({"epoch": epoch, **metrics})
    def log_final(self, metrics: Dict) -> None: self._h["final"] = metrics
    def save(self) -> None:
        p = self.output_dir / f"{self.run_name}_results.json"
        with open(p, "w") as f: json.dump(self._h, f, indent=2)
