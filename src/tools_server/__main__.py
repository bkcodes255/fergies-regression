"""python -m src.tools_server - serves on 0.0.0.0:$PORT (default 8080)."""
import os

import uvicorn

from src.tools_server.server import build_app

if __name__ == "__main__":
    uvicorn.run(build_app(), host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), proxy_headers=True)
