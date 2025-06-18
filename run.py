from __future__ import annotations

"""Application entrypoint."""

import uvicorn


if __name__ == "__main__":
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000)
