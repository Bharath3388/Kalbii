from __future__ import annotations
from fastapi import Header, HTTPException, status
from .config import get_settings


async def api_key_auth(x_api_key: str | None = Header(default=None)) -> str:
    keys = get_settings().api_keys
    if not keys:                                          # auth disabled
        return "anonymous"
    if x_api_key is None or x_api_key not in keys:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="invalid api key")
    return x_api_key
