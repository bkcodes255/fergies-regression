"""One server, two faces, same tools (contract v0.1):

- MCP (streamable HTTP) at /mcp - for Claude, and for Muse if its custom connectors take an MCP URL.
- REST at POST /v1/tools/<name> with a JSON body of arguments, and GET /v1/tools for the catalog
  (names, descriptions and the same input schemas MCP publishes) - for Muse's API-based connectors.

Every route except /healthz needs `Authorization: Bearer <key>`, checked against
TOOLS_SERVER_API_KEYS (comma-separated, so each client gets its own key and can be revoked alone).
The server refuses to start without at least one key.

Run locally:  python -m src.tools_server   (PORT defaults to 8080)
"""
from __future__ import annotations

import hmac
import json
import logging
import os

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from pydantic import ValidationError, validate_call
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from src.tools_server.tools import TOOLS, ToolError

VERSION = "0.1.0"
MIN_KEY_LENGTH = 32
log = logging.getLogger("tools_server")

INSTRUCTIONS = (
    "Read-only tools for Fergie's Regression, a Fantasy Premier League decision-support app. "
    "Start with get_status and mention its warnings when numbers are stale. Plans here are the "
    "same ones the app's Telegram approval flow proposes and auto-submits; nothing here can "
    "change the team. Predictions are model estimates, not facts: quote floor and ceiling when "
    "the spread matters, and say when a question needs information these tools don't have "
    "(team news beyond the status/news fields, press conferences, penalty takers)."
)

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)


def _ok(data: dict) -> JSONResponse:
    return JSONResponse({"ok": True, "data": data})


def _error(code: str, message: str, status: int) -> JSONResponse:
    return JSONResponse({"ok": False, "error": {"code": code, "message": message}}, status_code=status)


def _api_keys() -> list[str]:
    keys = [k.strip() for k in os.environ.get("TOOLS_SERVER_API_KEYS", "").split(",") if k.strip()]
    if not keys:
        raise RuntimeError("TOOLS_SERVER_API_KEYS is not set - refusing to start an unauthenticated server.")
    short = [k for k in keys if len(k) < MIN_KEY_LENGTH]
    if short:
        raise RuntimeError(f"Every key in TOOLS_SERVER_API_KEYS must be at least {MIN_KEY_LENGTH} characters.")
    return keys


def _transport_security() -> TransportSecuritySettings:
    """Host-header check against DNS rebinding. TOOLS_SERVER_ALLOWED_HOSTS lists the public
    hostname(s) once deployed (e.g. 'fergies-tools-xyz.a.run.app'); localhost is always allowed."""
    hosts = [h.strip() for h in os.environ.get("TOOLS_SERVER_ALLOWED_HOSTS", "").split(",") if h.strip()]
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=hosts + ["127.0.0.1:*", "localhost:*", "127.0.0.1", "localhost"],
        allowed_origins=[],
    )


class BearerKeyAuth:
    """Plain ASGI middleware: a valid bearer key or a 401, before any route runs."""

    def __init__(self, app, keys: list[str], open_paths: frozenset[str] = frozenset({"/healthz"})):
        self.app, self.keys, self.open_paths = app, [k.encode() for k in keys], open_paths

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] in self.open_paths:
            return await self.app(scope, receive, send)
        auth = dict(scope.get("headers") or []).get(b"authorization", b"")
        token = auth[7:].strip() if auth[:7].lower() == b"bearer " else b""
        if token and any(hmac.compare_digest(token, k) for k in self.keys):
            return await self.app(scope, receive, send)
        response = _error("unauthorized", "Missing or invalid API key (Authorization: Bearer <key>).", 401)
        response.headers["WWW-Authenticate"] = "Bearer"
        return await response(scope, receive, send)


def build_server() -> MCPServer:
    server = MCPServer("fergies-regression", title="Fergie's Regression", version=VERSION, instructions=INSTRUCTIONS)
    validated = {}
    for fn in TOOLS:
        server.add_tool(fn, annotations=READ_ONLY)
        validated[fn.__name__] = validate_call(fn)

    @server.custom_route("/healthz", methods=["GET"])
    async def healthz(request: Request) -> Response:
        return JSONResponse({"ok": True, "version": VERSION})

    @server.custom_route("/v1/tools", methods=["GET"])
    async def catalog(request: Request) -> Response:
        tools = await server.list_tools()
        return _ok({"version": VERSION, "tools": [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in tools
        ]})

    @server.custom_route("/v1/tools/{name}", methods=["POST"])
    async def call(request: Request) -> Response:
        name = request.path_params["name"]
        fn = validated.get(name)
        if fn is None:
            return _error("unknown_tool", f"No tool named {name!r}. GET /v1/tools lists them.", 404)
        body = await request.body()
        try:
            args = json.loads(body) if body.strip() else {}
        except json.JSONDecodeError:
            return _error("invalid_json", "The request body must be a JSON object of arguments.", 400)
        if not isinstance(args, dict):
            return _error("invalid_arguments", "The request body must be a JSON object of arguments.", 400)
        try:
            return _ok(await run_in_threadpool(fn, **args))  # DB work stays off the event loop
        except ValidationError as exc:
            detail = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
            return _error("invalid_arguments", detail, 400)
        except ToolError as exc:
            return _error(exc.code, exc.message, 422)
        except Exception:  # noqa: BLE001 - never leak internals (connection strings, SQL) to the caller
            log.exception("Tool %s failed", name)
            return _error("internal_error", "The tool failed unexpectedly; the server log has details.", 500)

    return server


def build_app():
    keys = _api_keys()
    server = build_server()
    app = server.streamable_http_app(
        streamable_http_path="/mcp", stateless_http=True, json_response=True,
        transport_security=_transport_security(), host="0.0.0.0",
    )
    return BearerKeyAuth(app, keys)
