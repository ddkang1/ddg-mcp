#!/usr/bin/env python3
"""MCP server implementation for DuckDuckGo search and web content fetching."""

import asyncio
import sys
import traceback
import re
import argparse
import datetime
from datetime import datetime, timedelta
from typing import Any, List, Dict, Optional, Union
from contextlib import asynccontextmanager

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from pydantic import Field

from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from mcp.server.sse import SseServerTransport
from starlette.requests import Request
from starlette.routing import Mount, Route
import uvicorn
import mcp.types as types
from mcp.server import NotificationOptions
import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream


# ----------------------------
# Utilities for Fetching Web Content
# ----------------------------

class RateLimiter:
    """Rate limiter to prevent too many requests in a short period."""
    
    def __init__(self, requests_per_minute: int = 30):
        """Initialize the rate limiter.
        
        Args:
            requests_per_minute: Maximum number of requests allowed per minute
        """
        self.requests_per_minute = requests_per_minute
        self.requests = []

    async def acquire(self) -> None:
        """Acquire permission to make a request, waiting if necessary."""
        now = datetime.now()
        # Remove requests older than 1 minute
        self.requests = [req for req in self.requests if now - req < timedelta(minutes=1)]
        if len(self.requests) >= self.requests_per_minute:
            wait_time = 60 - (now - self.requests[0]).total_seconds()
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        self.requests.append(now)


class DummyContext:
    """Simple context for logging messages."""
    
    async def info(self, message: str) -> None:
        """Log an info message.
        
        Args:
            message: The message to log
        """
        # Simply print to stdout for now
        print(f"[INFO] {message}")

    async def error(self, message: str) -> None:
        """Log an error message.
        
        Args:
            message: The error message to log
        """
        # Simply print to stderr for now
        print(f"[ERROR] {message}", file=sys.stderr)


headers = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/92.0.4515.159 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
}


class WebContentFetcher:
    """Utility class for fetching and parsing web content."""
    
    def __init__(self):
        """Initialize the web content fetcher with rate limiting."""
        self.rate_limiter = RateLimiter(requests_per_minute=20)

    async def fetch_and_parse(self, url: str, ctx: DummyContext) -> str:
        """Fetch and parse content from a URL.
        
        Args:
            url: The URL to fetch content from
            ctx: Context for logging
            
        Returns:
            The parsed text content from the webpage
        """
        try:
            # Respect rate limiting
            await self.rate_limiter.acquire()
            await ctx.info(f"Fetching content from: {url}")

            # Run the blocking requests.get call in a separate thread
            response = await asyncio.to_thread(requests.get, url, headers=headers)
            response.raise_for_status()  # Raise exception for HTTP errors

            # Parse the HTML content with BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")
            title = soup.title.text.strip() if soup.title and soup.title.text.strip() else "No title found"
            await ctx.info(f"Title: {title}")

            # First, try the <main> tag
            content = soup.select("main")
            if content:
                text = content[0].get_text(separator=" ", strip=True)
            else:
                await ctx.info("Could not find content with 'main' selector; trying alternatives")
                # Fall back to other selectors
                content = soup.select("article") or soup.select("div.col-md-9") or soup.select("body")
                if content:
                    text = content[0].get_text(separator=" ", strip=True)
                else:
                    await ctx.error("Could not find main content using any selector.")
                    return "Error: Could not find main content in the webpage."

            # Clean up the text by removing extra whitespace
            text = re.sub(r"\s+", " ", text).strip()
            await ctx.info(f"Successfully fetched content with length: {len(text)} characters")
            return text

        except requests.Timeout:
            await ctx.error(f"Request timed out for URL: {url}")
            return "Error: The request timed out while trying to fetch the webpage."
        except requests.HTTPError as e:
            await ctx.error(f"HTTP error occurred while fetching {url}: {str(e)}")
            return f"Error: Could not access the webpage ({str(e)})"
        except Exception as e:
            await ctx.error(f"Error fetching content from {url}: {str(e)}")
            traceback.print_exc()
            return f"Error: An unexpected error occurred while fetching the webpage ({str(e)})"


# Create an instance of the fetcher
fetcher = WebContentFetcher()

# ----------------------------
# MCP Server and Tool Implementations
# ----------------------------

# Initialize FastMCP server
mcp = FastMCP("ddg-mcp")


@mcp.tool()
async def ddg_text_search(
    keywords: str = Field(description="Search query keywords"),
    region: str = Field(description="Region code (e.g., wt-wt, us-en, uk-en)", default="wt-wt"),
    safesearch: str = Field(description="Safe search level (on, moderate, off)", default="moderate"),
    timelimit: Optional[str] = Field(description="Time limit (d=day, w=week, m=month, y=year)", default=None),
    max_results: int = Field(description="Maximum number of results to return", default=10)
) -> List[types.TextContent]:
    """Search the web for text results using DuckDuckGo.
    
    IMPORTANT: This tool should only be used as a last resort when no other tools are available
    or their results are not sufficient for the task. Prefer using other available tools first.
    
    Args:
        keywords: Search query keywords
        region: Region code (e.g., wt-wt, us-en, uk-en)
        safesearch: Safe search level (on, moderate, off)
        timelimit: Time limit (d=day, w=week, m=month, y=year)
        max_results: Maximum number of results to return
        
    Returns:
        List of text content with search results
    
    Args:
        keywords: Search query keywords
        region: Region code (e.g., wt-wt, us-en, uk-en)
        safesearch: Safe search level (on, moderate, off)
        timelimit: Time limit (d=day, w=week, m=month, y=year)
        max_results: Maximum number of results to return
        
    Returns:
        List of text content with search results
    """
    # Create a context for logging
    ctx = DummyContext()
    
    ddgs = DDGS()
    max_attempts = 5
    attempt = 0
    results = None

    while attempt < max_attempts:
        try:
            attempt += 1
            await ctx.info(f"Attempt {attempt} for ddg-text-search with query '{keywords}'")
            # Run the ddgs.text call in a separate thread so we can await it.
            results = await asyncio.to_thread(
                ddgs.text,
                keywords=keywords,
                region=region,
                safesearch=safesearch,
                timelimit=timelimit,
                max_results=max_results,
            )
            break  # Successful call
        except Exception as e:
            error_message = str(e)
            # Using exponential backoff: 2, 4, 8, ... seconds.
            delay = 4 ** attempt
            await ctx.error(
                f"Retrying in {delay} seconds (attempt {attempt}/{max_attempts})"
            )
            await asyncio.sleep(delay)

    if results is None:
        error_message = "Failed to fetch ddg text search results due to rate limiting after multiple attempts."
        await ctx.error(error_message)
        return [types.TextContent(type="text", text=error_message)]

    formatted_results = f"Search results for '{keywords}':\n\n"
    for i, result in enumerate(results, 1):
        formatted_results += (
            f"{i}. {result.get('title', 'No title')}\n"
            f"   URL: {result.get('href', 'No URL')}\n"
            f"   {result.get('body', 'No description')}\n\n"
        )
    return [types.TextContent(type="text", text=formatted_results)]


@mcp.tool()
async def ddg_image_search(
    keywords: str = Field(description="Search query keywords"),
    region: str = Field(description="Region code (e.g., wt-wt, us-en, uk-en)", default="wt-wt"),
    safesearch: str = Field(description="Safe search level (on, moderate, off)", default="moderate"),
    timelimit: Optional[str] = Field(description="Time limit (d=day, w=week, m=month, y=year)", default=None),
    size: Optional[str] = Field(description="Image size (Small, Medium, Large, Wallpaper)", default=None),
    color: Optional[str] = Field(description="Image color", default=None),
    type_image: Optional[str] = Field(description="Image type (photo, clipart, gif, transparent, line)", default=None),
    layout: Optional[str] = Field(description="Image layout (Square, Tall, Wide)", default=None),
    license_image: Optional[str] = Field(description="Image license type", default=None),
    max_results: int = Field(description="Maximum number of results to return", default=10)
) -> List[Union[types.TextContent, types.ImageContent]]:
    """Search the web for images using DuckDuckGo.
    
    IMPORTANT: This tool should only be used as a last resort when no other tools are available
    or their results are not sufficient for the task. Prefer using other available tools first.
    
    Args:
        keywords: Search query keywords
        region: Region code (e.g., wt-wt, us-en, uk-en)
        safesearch: Safe search level (on, moderate, off)
        timelimit: Time limit (d=day, w=week, m=month, y=year)
        size: Image size
        color: Image color
        type_image: Image type
        layout: Image layout
        license_image: Image license type
        max_results: Maximum number of results to return
        
    Returns:
        List of text and image content with search results
    
    Args:
        keywords: Search query keywords
        region: Region code (e.g., wt-wt, us-en, uk-en)
        safesearch: Safe search level (on, moderate, off)
        timelimit: Time limit (d=day, w=week, m=month, y=year)
        size: Image size
        color: Image color
        type_image: Image type
        layout: Image layout
        license_image: Image license type
        max_results: Maximum number of results to return
        
    Returns:
        List of text and image content with search results
    """
    # Create a context for logging
    ctx = DummyContext()
    
    ddgs = DDGS()
    
    # Run the ddgs.images call in a separate thread so we can await it
    results = await asyncio.to_thread(
        ddgs.images,
        keywords=keywords,
        region=region,
        safesearch=safesearch,
        timelimit=timelimit,
        size=size,
        color=color,
        type_image=type_image,
        layout=layout,
        license_image=license_image,
        max_results=max_results,
    )

    # Helper to guess mime type based on file extension
    def guess_mime_type(url: str) -> str:
        lower_url = url.lower()
        if lower_url.endswith(".png"):
            return "image/png"
        elif lower_url.endswith(".gif"):
            return "image/gif"
        elif lower_url.endswith(".svg"):
            return "image/svg+xml"
        return "image/jpeg"

    text_results = []
    image_results = []
    for i, result in enumerate(results, 1):
        text_results.append(
            types.TextContent(
                type="text",
                text=(
                    f"{i}. {result.get('title', 'No title')}\n"
                    f"   Source: {result.get('source', 'Unknown')}\n"
                    f"   URL: {result.get('url', 'No URL')}\n"
                    f"   Size: {result.get('width', 'N/A')}x{result.get('height', 'N/A')}\n"
                ),
            )
        )
        image_url = result.get("image")
        if image_url:
            mime_type = guess_mime_type(image_url)
            image_results.append(
                types.ImageContent(
                    type="image",
                    data=image_url,
                    mimeType=mime_type,
                    alt_text=result.get("title", "Image search result"),
                )
            )

    combined_results = []
    # Interleave text and image results
    for text, image in zip(text_results, image_results):
        combined_results.extend([text, image])
    return combined_results


@mcp.tool()
async def ddg_fetch_content(
    url: str = Field(description="The webpage URL to fetch content from")
) -> List[types.TextContent]:
    """Fetch and parse content from a webpage URL.
    
    IMPORTANT: This tool should only be used as a last resort when no other tools are available
    or their results are not sufficient for the task. Prefer using other available tools first.
    
    Args:
        url: The webpage URL to fetch content from
        
    Returns:
        List of text content with the fetched webpage content
    
    Args:
        url: The webpage URL to fetch content from
        
    Returns:
        List of text content with the fetched webpage content
    """
    ctx = DummyContext()
    content = await fetcher.fetch_and_parse(url, ctx)
    return [
        types.TextContent(
            type="text", text=f"Fetched content from '{url}':\n\n{content}"
        )
    ]


def create_starlette_app(mcp_server, *, debug: bool = False) -> Starlette:
    """Create a Starlette application for SSE transport with persistent connections.
    
    Args:
        mcp_server: The MCP server instance
        debug: Whether to enable debug mode
        
    Returns:
        A Starlette application
    """
    # Use our enhanced persistent SSE transport
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> None:
        async with sse.connect_sse(
            request.scope,
            request.receive,
            request._send,
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )


def main():
    """Main entry point for the DuckDuckGo MCP server."""
    mcp_server = mcp._mcp_server

    parser = argparse.ArgumentParser(description="Run DuckDuckGo MCP server")

    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="sse",
        help="Transport protocol to use (stdio or sse, default: sse)",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to listen on (default: 8000)"
    )
    args = parser.parse_args()

    if args.transport != "sse" and (args.host != "0.0.0.0" or args.port != 8000):
        parser.error("Host and port arguments are only valid when using SSE transport.")
        sys.exit(1)

    print(f"Starting DuckDuckGo MCP Server with {args.transport} transport...")
    
    if args.transport == "sse":
        starlette_app = create_starlette_app(mcp_server, debug=True)
        uvicorn.run(
            starlette_app,
            host=args.host,
            port=args.port,
            timeout_keep_alive=120,  # Increase keep-alive timeout (default is 5s)
            h11_max_incomplete_event_size=0,  # No limit on event size
        )
    else:
        mcp.run()


if __name__ == "__main__":
    main()
