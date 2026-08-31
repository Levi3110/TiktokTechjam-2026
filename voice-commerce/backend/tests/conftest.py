import os


# Tests must never depend on the LAN-hosted model server.
os.environ["QWEN_MODE"] = "demo"
os.environ["ENABLE_SBERT"] = "false"
