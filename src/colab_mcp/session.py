# Copyright 2026 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from collections.abc import AsyncIterator
import contextlib
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass
from fastmcp import FastMCP, Client
from fastmcp.client.transports import ClientTransport
from fastmcp.dependencies import CurrentContext
from fastmcp.server.context import Context
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.middleware.tool_injection import ToolInjectionMiddleware
from fastmcp.server.proxy import FastMCPProxy
from fastmcp.tools.tool import Tool, ToolResult
from mcp.client.session import ClientSession
from mcp.types import TextContent
from typing import Any, cast
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
import webbrowser

from colab_mcp.websocket_server import (
    COLAB,
    COLAB_ALT_DOMAIN,
    ColabWebSocketServer,
    SCRATCH_PATH,
)

UI_CONNECTION_TIMEOUT = 60.0  # secs

FE_CONNECTED_KEY = "fe_connected"
PROXY_TOKEN_KEY = "proxy_token"
PROXY_PORT_KEY = "proxy_port"
AUTHUSER_KEY = "authuser"
INJECTED_TOOL_NAME = "open_colab_browser_connection"
CONNECTION_INFO_TOOL_NAME = "get_colab_browser_connection_info"


@dataclass
class ColabConnectionInfo:
    url: str
    token: str
    port: int
    authuser: str
    connected: bool


def _fastmcp_context(context: MiddlewareContext) -> Any:
    return context.fastmcp_context


def _resolve_authuser(authuser: str, ctx: Context) -> str:
    if authuser:
        return authuser
    return ctx.get_state(AUTHUSER_KEY) or "0"


def _build_colab_connection_url(
    token: str, port: int, authuser: str, url: str = ""
) -> str:
    target_url = url or f"{COLAB}{SCRATCH_PATH}"
    parsed = urlparse(target_url)
    if parsed.scheme != "https" or parsed.netloc not in {
        urlparse(COLAB).netloc,
        urlparse(COLAB_ALT_DOMAIN).netloc,
    }:
        raise ValueError("Colab URL must use https and point to a Google Colab host")

    query_params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query_params["authuser"] = authuser

    fragment_parts = [part for part in parsed.fragment.split("&") if part]
    fragment_parts = [
        part
        for part in fragment_parts
        if not part.startswith("mcpProxyToken=")
        and not part.startswith("mcpProxyPort=")
    ]
    fragment_parts.extend([f"mcpProxyToken={token}", f"mcpProxyPort={port}"])

    return urlunparse(
        parsed._replace(
            query=urlencode(query_params),
            fragment="&".join(fragment_parts),
        )
    )


def _get_connection_info(
    ctx: Context, authuser: str = "", url: str = ""
) -> ColabConnectionInfo:
    token = ctx.get_state(PROXY_TOKEN_KEY)
    port = ctx.get_state(PROXY_PORT_KEY)
    resolved_authuser = _resolve_authuser(authuser, ctx)
    connection_url = _build_colab_connection_url(
        token=token,
        port=port,
        authuser=resolved_authuser,
        url=url,
    )
    return ColabConnectionInfo(
        url=connection_url,
        token=token,
        port=port,
        authuser=resolved_authuser,
        connected=bool(ctx.get_state(FE_CONNECTED_KEY)),
    )


class ColabTransport(ClientTransport):
    def __init__(self, wss: ColabWebSocketServer):
        self.wss = wss

    @contextlib.asynccontextmanager
    async def connect_session(self, **session_kwargs) -> AsyncIterator[ClientSession]:
        async with ClientSession(
            self.wss.read_stream, self.wss.write_stream, **session_kwargs
        ) as session:
            yield session

    def __repr__(self) -> str:
        return "<ColabSessionProxyTransport>"


class ColabProxyClient:
    def __init__(self, wss: ColabWebSocketServer):
        self.wss = wss
        self.stubbed_mcp_client = Client(FastMCP())
        self.proxy_mcp_client: Client | None = None
        self._exit_stack = AsyncExitStack()
        self._start_task = None

    def is_connected(self):
        return self.wss.connection_live.is_set() and self.proxy_mcp_client is not None

    async def await_proxy_connection(self):
        if self._start_task is None:
            return
        start_task = self._start_task
        with contextlib.suppress(asyncio.TimeoutError):
            # wait for the connection to be live and for the proxy client to fully initialize
            connection_tasks = asyncio.gather(
                self.wss.connection_live.wait(), start_task
            )
            await asyncio.wait_for(
                connection_tasks,
                timeout=UI_CONNECTION_TIMEOUT,
            )

    def client_factory(self):
        if self.is_connected():
            return cast(Client, self.proxy_mcp_client)
        # return a client mapped to a stubbed mcp server if there is no session proxy
        return self.stubbed_mcp_client

    async def _start_proxy_client(self):
        # blocks until a websocket connection is made successfully
        self.proxy_mcp_client = await self._exit_stack.enter_async_context(
            Client(ColabTransport(self.wss))
        )

    async def __aenter__(self):
        self._start_task = asyncio.create_task(self._start_proxy_client())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._start_task:
            self._start_task.cancel()
        await self._exit_stack.aclose()


class ColabProxyMiddleware(Middleware):
    def __init__(self, proxy_client: ColabProxyClient, authuser: str = "0"):
        self.proxy_client = proxy_client
        self.authuser = authuser
        self.last_message_connected = self.proxy_client.is_connected()

    async def on_message(self, context: MiddlewareContext, call_next):
        """
        Check for a change to Colab session connectivity on any communication with this MCP server and
        notify the client when the connectivity status has changed.
        """
        fastmcp_context = _fastmcp_context(context)
        fastmcp_context.set_state(FE_CONNECTED_KEY, self.proxy_client.is_connected())
        fastmcp_context.set_state(PROXY_TOKEN_KEY, self.proxy_client.wss.token)
        fastmcp_context.set_state(PROXY_PORT_KEY, self.proxy_client.wss.port)
        fastmcp_context.set_state(AUTHUSER_KEY, self.authuser)

        result = await call_next(context)

        connected = self.proxy_client.is_connected()
        connection_state_changed = connected != self.last_message_connected
        self.last_message_connected = connected
        if connection_state_changed:
            await fastmcp_context.send_tool_list_changed()

        return result

    async def on_call_tool(self, context, call_next):
        fastmcp_context = _fastmcp_context(context)
        result = await call_next(context)
        if context.message.name != INJECTED_TOOL_NAME:
            return result
        if self.proxy_client.is_connected():
            return result
        # if the tool call was for open_colab_browser_connection and there is no existing connection, try to await full connection
        await fastmcp_context.report_progress(
            progress=1, total=3, message="The user is not connected to the Colab UI"
        )
        await fastmcp_context.report_progress(
            progress=2,
            total=3,
            message="Waiting for user to connect in Colab - will wait for 60s",
        )
        await self.proxy_client.await_proxy_connection()
        if self.proxy_client.is_connected():
            await fastmcp_context.report_progress(
                progress=3, total=3, message="The Colab UI is successfully connected!"
            )
            return ToolResult(
                content=[TextContent(type="text", text="true")],
                structured_content={"result": True},
            )
        else:
            await fastmcp_context.report_progress(
                progress=3,
                total=3,
                message="Timeout while waiting for the user to connect.",
            )
            return ToolResult(
                content=[TextContent(type="text", text="false")],
                structured_content={"result": False},
            )


async def check_session_proxy_tool_fn(
    authuser: str = "", url: str = "", ctx: Context = CurrentContext()
) -> bool:
    """Opens a connection to a Google Colab browser session and unlocks notebook editing tools.

    Args:
        authuser: Google account index to open Colab with (e.g. "0" for default,
            "1" for second account, "2" for third). If not provided, falls back
            to the --authuser CLI flag or defaults to "0".
        url: Optional full Google Colab notebook URL to open. If omitted, opens
            the default empty Colab notebook.

    Returns:
        True if the connection attempt succeeded, False otherwise.
    """
    fe_connected = ctx.get_state(FE_CONNECTED_KEY)
    if fe_connected:
        return True
    connection_info = _get_connection_info(ctx, authuser=authuser, url=url)
    webbrowser.open_new(connection_info.url)
    return False


async def get_session_proxy_connection_info_tool_fn(
    authuser: str = "", url: str = "", ctx: Context = CurrentContext()
) -> dict:
    """Returns Colab browser connection details for manual connection flows.

    Args:
        authuser: Google account index to use when building the Colab URL.
        url: Optional full Google Colab notebook URL to target.

    Returns:
        A dict containing the final URL, token, port, authuser, and current
        connection status.
    """
    return asdict(_get_connection_info(ctx, authuser=authuser, url=url))


check_session_proxy_tool = Tool.from_function(
    fn=check_session_proxy_tool_fn,
    name=INJECTED_TOOL_NAME,
    description="Opens a connection to a Google Colab browser session and unlocks notebook editing tools. Pass authuser to select which Google account to use and url to open a specific Colab notebook. Returns a boolean representing whether the connection attempt succeeded",
)


get_session_proxy_connection_info_tool = Tool.from_function(
    fn=get_session_proxy_connection_info_tool_fn,
    name=CONNECTION_INFO_TOOL_NAME,
    description="Returns Colab browser connection details for manual connection flows, including the final URL, proxy token, proxy port, selected authuser, and whether the browser is already connected",
)


class ColabSessionProxy:
    def __init__(self, authuser: str = "0"):
        self._exit_stack = AsyncExitStack()
        self.proxy_server: FastMCPProxy | None = None
        # list order matters, see: https://gofastmcp.com/servers/middleware#multiple-middleware
        self.middleware: list[Middleware] = []
        self.wss: ColabWebSocketServer | None = None
        self.authuser = authuser

    async def start_proxy_server(self):
        self.wss = await self._exit_stack.enter_async_context(ColabWebSocketServer())
        proxy_client = await self._exit_stack.enter_async_context(
            ColabProxyClient(self.wss)
        )
        self.proxy_server = FastMCPProxy(
            client_factory=cast(Any, proxy_client.client_factory),
            instructions="Connects to a user's Google Colab session in a browser and allows for interactions with their Google Colab notebook",
        )
        # ColabProxyMiddleware must be first because it sets the fe_connected state
        self.middleware.append(
            ColabProxyMiddleware(proxy_client, authuser=self.authuser)
        )
        self.middleware.append(
            ToolInjectionMiddleware(
                tools=[check_session_proxy_tool, get_session_proxy_connection_info_tool]
            )
        )

    async def cleanup(self):
        await self._exit_stack.aclose()
