"""Kalshi API authentication and a thin signed-request client.

Kalshi key-pair auth: every request carries three headers --
  KALSHI-ACCESS-KEY        the API key id
  KALSHI-ACCESS-TIMESTAMP  unix milliseconds
  KALSHI-ACCESS-SIGNATURE  base64( RSA-PSS-SHA256( timestamp + METHOD + PATH ) )

The signed string uses the *path only* (no query string, no host).
"""

from __future__ import annotations

import base64
import logging
import time
from pathlib import Path
from typing import Any, Optional

import aiohttp
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from config import KalshiConfig

log = logging.getLogger("kalshi")


class KalshiAuth:
    """Loads the RSA private key and produces request signatures."""

    def __init__(self, cfg: KalshiConfig) -> None:
        self._key_id = cfg.api_key_id
        key_bytes = Path(cfg.private_key_path).read_bytes()
        loaded = serialization.load_pem_private_key(key_bytes, password=None)
        if not isinstance(loaded, rsa.RSAPrivateKey):
            raise TypeError("Kalshi private key must be an RSA key")
        self._private_key: rsa.RSAPrivateKey = loaded

    @property
    def key_id(self) -> str:
        return self._key_id

    def _sign(self, message: str) -> str:
        signature = self._private_key.sign(
            message.encode("utf-8"),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def headers(self, method: str, path: str) -> dict[str, str]:
        """Auth headers for a REST call or a WS handshake.

        `path` must be the exact path component the server sees, no query.
        """
        timestamp_ms = str(int(time.time() * 1000))
        message = timestamp_ms + method.upper() + path
        return {
            "KALSHI-ACCESS-KEY": self._key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": self._sign(message),
        }


class KalshiRestClient:
    """Minimal async REST client. Open with `async with`."""

    def __init__(self, cfg: KalshiConfig, auth: KalshiAuth) -> None:
        self._cfg = cfg
        self._auth = auth
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "KalshiRestClient":
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._session is not None:
            await self._session.close()

    async def _request(
        self,
        method: str,
        endpoint: str,
        *,
        json_body: Optional[dict[str, Any]] = None,
        timeout_s: float = 5.0,
    ) -> dict[str, Any]:
        if self._session is None:
            raise RuntimeError("KalshiRestClient used outside its context manager")

        path = self._cfg.rest_prefix + endpoint
        url = self._cfg.rest_base + path
        headers = self._auth.headers(method, path)
        headers["Content-Type"] = "application/json"

        async with self._session.request(
            method,
            url,
            json=json_body,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout_s),
        ) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise KalshiAPIError(resp.status, text)
            return await resp.json() if text else {}

    async def get_balance(self) -> dict[str, Any]:
        return await self._request("GET", "/portfolio/balance")

    async def get_market(self, ticker: str) -> dict[str, Any]:
        return await self._request("GET", f"/markets/{ticker}")

    async def get_positions(self) -> dict[str, Any]:
        return await self._request("GET", "/portfolio/positions")

    async def create_order(self, order: dict[str, Any], timeout_s: float = 5.0) -> dict[str, Any]:
        """POST /portfolio/orders -- transmits a real order.

        UNVERIFIED -- /portfolio/orders is the LEGACY endpoint (deprecation
        began ~2026-05-06). The current path is /portfolio/events/orders (V2)
        with a different body. Confirm against the docs before live use.
        """
        return await self._request(
            "POST", "/portfolio/orders", json_body=order, timeout_s=timeout_s
        )


class KalshiAPIError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"Kalshi API error {status}: {body}")
        self.status = status
        self.body = body
