# test_config_and_schema.py
import yaml
from schemas import DetectionEvent

cfg = yaml.safe_load(open("config.yaml"))
print("✅ Config loaded:", cfg["detection"])

event = DetectionEvent(type="choking", confidence=0.20, coords=(120, 200))
print("✅ Sample event:", event.json())