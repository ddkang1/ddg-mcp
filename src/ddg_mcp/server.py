import asyncio
import sys
import traceback
import re
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio


# ----------------------------
# Utilities for Fetching Web Content
# ----------------------------

class RateLimiter:
    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self.requests = []

    async def acquire(self) -> None:
        now = datetime.now()
        # Remove requests older than 1 minute
        self.requests = [req for req in self.requests if now - req < timedelta(minutes=1)]
        if len(self.requests) >= self.requests_per_minute:
            wait_time = 60 - (now - self.requests[0]).total_seconds()
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        self.requests.append(now)


class DummyContext:
    async def info(self, message: str) -> None:
        # Simply print to stdout for now
        print(f"[INFO] {message}")

    async def error(self, message: str) -> None:
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
    def __init__(self):
        self.rate_limiter = RateLimiter(requests_per_minute=20)

    async def fetch_and_parse(self, url: str, ctx: DummyContext) -> str:
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

server = Server("ddg-mcp")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """
    List available tools with their input schemas.
    """
    return [
        types.Tool(
            name="ddg-text-search",
            description="Search the web for text results using DuckDuckGo",
            inputSchema={
                "type": "object",
                "properties": {
                    "keywords": {"type": "string", "description": "Search query keywords"},
                    "region": {
                        "type": "string",
                        "description": "Region code (e.g., wt-wt, us-en, uk-en)",
                        "default": "wt-wt",
                    },
                    "safesearch": {
                        "type": "string",
                        "enum": ["on", "moderate", "off"],
                        "description": "Safe search level",
                        "default": "moderate",
                    },
                    "timelimit": {
                        "type": "string",
                        "enum": ["d", "w", "m", "y"],
                        "description": "Time limit (d=day, w=week, m=month, y=year)",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10,
                    },
                },
                "required": ["keywords"],
            },
        ),
        types.Tool(
            name="ddg-image-search",
            description="Search the web for images using DuckDuckGo",
            inputSchema={
                "type": "object",
                "properties": {
                    "keywords": {"type": "string", "description": "Search query keywords"},
                    "region": {
                        "type": "string",
                        "description": "Region code (e.g., wt-wt, us-en, uk-en)",
                        "default": "wt-wt",
                    },
                    "safesearch": {
                        "type": "string",
                        "enum": ["on", "moderate", "off"],
                        "description": "Safe search level",
                        "default": "moderate",
                    },
                    "timelimit": {
                        "type": "string",
                        "enum": ["d", "w", "m", "y"],
                        "description": "Time limit (d=day, w=week, m=month, y=year)",
                    },
                    "size": {
                        "type": "string",
                        "enum": ["Small", "Medium", "Large", "Wallpaper"],
                        "description": "Image size",
                    },
                    "color": {
                        "type": "string",
                        "enum": [
                            "color",
                            "Monochrome",
                            "Red",
                            "Orange",
                            "Yellow",
                            "Green",
                            "Blue",
                            "Purple",
                            "Pink",
                            "Brown",
                            "Black",
                            "Gray",
                            "Teal",
                            "White",
                        ],
                        "description": "Image color",
                    },
                    "type_image": {
                        "type": "string",
                        "enum": ["photo", "clipart", "gif", "transparent", "line"],
                        "description": "Image type",
                    },
                    "layout": {
                        "type": "string",
                        "enum": ["Square", "Tall", "Wide"],
                        "description": "Image layout",
                    },
                    "license_image": {
                        "type": "string",
                        "enum": ["any", "Public", "Share", "ShareCommercially", "Modify", "ModifyCommercially"],
                        "description": "Image license type",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10,
                    },
                },
                "required": ["keywords"],
            },
        ),
        types.Tool(
            name="ddg-fetch-content",
            description="Fetch and parse content from a webpage URL",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The webpage URL to fetch content from"},
                },
                "required": ["url"],
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[
    types.TextContent | types.ImageContent | types.EmbeddedResource
]:
    if not arguments:
        raise ValueError("Missing arguments")

    if name == "ddg-text-search":
        keywords = arguments.get("keywords")
        if not keywords:
            raise ValueError("Missing keywords")
        region = arguments.get("region", "wt-wt")
        safesearch = arguments.get("safesearch", "moderate")
        timelimit = arguments.get("timelimit")
        max_results = arguments.get("max_results", 10)

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
                # Check if error message hints at rate limit (429 or "rate")
                # if "429" in error_message or "rate" in error_message.lower():
                    # using exponential backoff: 2, 4, 8, ... seconds.
                delay = 2 ** attempt
                await ctx.error(f"Error during ddg-text-search: {error_message}")
                await ctx.info(
                    f"Retrying in {delay} seconds (attempt {attempt}/{max_attempts})"
                )
                await asyncio.sleep(delay)
                # else:
                #     await ctx.error(f"Error during ddg-text-search: {error_message}")
                #     raise  # re-raise unexpected exceptions

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

    elif name == "ddg-image-search":
        keywords = arguments.get("keywords")
        if not keywords:
            raise ValueError("Missing keywords")
        region = arguments.get("region", "wt-wt")
        safesearch = arguments.get("safesearch", "moderate")
        timelimit = arguments.get("timelimit")
        size = arguments.get("size")
        color = arguments.get("color")
        type_image = arguments.get("type_image")
        layout = arguments.get("layout")
        license_image = arguments.get("license_image")
        max_results = arguments.get("max_results", 10)

        ddgs = DDGS()
        results = ddgs.images(
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

    elif name == "ddg-fetch-content":
        url = arguments.get("url")
        if not url:
            raise ValueError("Missing url")
        ctx = DummyContext()
        content = await fetcher.fetch_and_parse(url, ctx)
        return [
            types.TextContent(
                type="text", text=f"Fetched content from '{url}':\n\n{content}"
            )
        ]
    else:
        raise ValueError(f"Unknown tool: {name}")


async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="ddg-mcp",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(), experimental_capabilities={}
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
