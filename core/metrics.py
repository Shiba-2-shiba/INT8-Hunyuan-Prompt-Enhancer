import json
import os
import time
from typing import Any, Dict

class MetricsCollector:
    """
    Lightweight metrics collector.
    Designed to be fail-open: errors in logging should not crash the application.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MetricsCollector, cls).__new__(cls)
            cls._instance.enabled = False
            cls._instance.log_file = None
        return cls._instance

    def configure(self, config: Dict[str, Any]):
        """
        Configure the collector from policy/config dict.
        Expects config to have 'metrics' key with 'enabled' and 'log_file'.
        """
        metrics_conf = config.get("metrics", {})
        self.enabled = metrics_conf.get("enabled", False)
        
        if self.enabled:
            log_path = metrics_conf.get("log_file", "logs/metrics.jsonl")
            # Ensure directory exists
            log_dir = os.path.dirname(os.path.abspath(log_path))
            if log_dir and not os.path.exists(log_dir):
                try:
                    os.makedirs(log_dir, exist_ok=True)
                except Exception as e:
                    print(f"[Metrics] Failed to create log directory: {e}")
                    self.enabled = False
                    return

            self.log_file = log_path

    def log(self, event_type: str, data: Dict[str, Any]):
        """
        Log an event.
        """
        if not self.enabled or not self.log_file:
            return

        try:
            entry = {
                "timestamp": time.time(),
                "event": event_type,
                **data
            }
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            # Fail-open: Do not crash app, just print warning
            print(f"[Metrics] Failed to log event {event_type}: {e}")

# Global instance
collector = MetricsCollector()
