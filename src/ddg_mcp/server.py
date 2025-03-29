import asyncio
import sys
import traceback
import re
import urllib.parse
from datetime import datetime, timedelta

import httpx
from bs4 import BeautifulSoup, Comment
from fake_useragent import UserAgent
from readability import Document 

from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
from pydantic import AnyUrl
import mcp.server.stdio
from duckduckgo_search import DDGS

# ----------------------------
# Utilities for fetching web content
# ----------------------------

# RateLimiter to ensure we don't exceed request limits.
class RateLimiter:
    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self.requests = []

    async def acquire(self):
        now = datetime.now()
        # Remove requests older than 1 minute
        self.requests = [req for req in self.requests if now - req < timedelta(minutes=1)]
        if len(self.requests) >= self.requests_per_minute:
            wait_time = 60 - (now - self.requests[0]).total_seconds()
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        self.requests.append(now)

# A minimal context for logging used by the content fetcher.
class DummyContext:
    async def info(self, message: str) -> None:
        # For now, simply print to stdout (or integrate with your logging)
        print(f"[INFO] {message}")

    async def error(self, message: str) -> None:
        # For now, simply print to stderr (or integrate with your logging)
        print(f"[ERROR] {message}", file=sys.stderr)

class WebContentFetcher:
    def __init__(self):
        self.rate_limiter = RateLimiter(requests_per_minute=20)
        self.ua = UserAgent()

    async def fetch_and_parse(self, url: str, ctx: DummyContext) -> str:
        """Fetches content from the given URL and extracts the main visible text using Readability."""
        try:
            await self.rate_limiter.acquire()
            await ctx.info(f"Fetching content from: {url}")

            # Generate a random User-Agent for every request.
            headers = {
                "User-Agent": self.ua.random,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Referer": "https://pmc.ncbi.nlm.nih.gov/",
            }

            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, follow_redirects=True, timeout=30.0)
                response.raise_for_status()

            # Use Readability to extract the main article content.
            doc = Document(response.text)
            # The summary() method returns the HTML of the main content.
            summary_html = doc.summary()

            # Use BeautifulSoup to extract clean text from the summary HTML.
            soup = BeautifulSoup(summary_html, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            text = re.sub(r"\s+", " ", text).strip()

            await ctx.info(f"Successfully fetched and parsed main content ({len(text)} characters)")
            return text

        except httpx.TimeoutException:
            await ctx.error(f"Request timed out for URL: {url}")
            return "Error: The request timed out while trying to fetch the webpage."
        except httpx.HTTPError as e:
            await ctx.error(f"HTTP error occurred while fetching {url}: {str(e)}")
            return f"Error: Could not access the webpage ({str(e)})"
        except Exception as e:
            await ctx.error(f"Error fetching content from {url}: {str(e)}")
            traceback.print_exc()
            return f"Error: An unexpected error occurred while fetching the webpage ({str(e)})"

# Create an instance of WebContentFetcher for use in the tool handler.
fetcher = WebContentFetcher()

# ----------------------------
# MCP Server and Tool Implementations
# ----------------------------

server = Server("ddg-mcp")

@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """
    List available resources.
    Currently, no resources are exposed.
    """
    return []

@server.list_prompts()
async def handle_list_prompts() -> list[types.Prompt]:
    """
    List available prompts.
    Each prompt can have optional arguments to customize its behavior.
    """
    return [
        types.Prompt(
            name="search-results-summary",
            description="Creates a summary of search results",
            arguments=[
                types.PromptArgument(
                    name="query",
                    description="Search query to summarize results for",
                    required=True,
                ),
                types.PromptArgument(
                    name="style",
                    description="Style of the summary (brief/detailed)",
                    required=False,
                )
            ],
        )
    ]

@server.get_prompt()
async def handle_get_prompt(
    name: str, arguments: dict[str, str] | None
) -> types.GetPromptResult:
    """
    Generate a prompt by combining arguments with server state.
    """
    if name == "search-results-summary":
        if not arguments or "query" not in arguments:
            raise ValueError("Missing required 'query' argument")
        
        query = arguments.get("query")
        style = arguments.get("style", "brief")
        detail_prompt = " Give extensive details." if style == "detailed" else ""
        
        # Perform search and get results
        ddgs = DDGS()
        results = ddgs.text(query, max_results=10)
        
        results_text = "\n\n".join([
            f"Title: {result.get('title', 'No title')}\n"
            f"URL: {result.get('href', 'No URL')}\n"
            f"Description: {result.get('body', 'No description')}"
            for result in results
        ])
        
        return types.GetPromptResult(
            description=f"Summarize search results for '{query}'",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"Here are the search results for '{query}'. Please summarize them{detail_prompt}:\n\n{results_text}",
                    ),
                )
            ],
        )
    else:
        raise ValueError(f"Unknown prompt: {name}")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """
    List available tools.
    Each tool specifies its arguments using JSON Schema validation.
    """
    return [
        types.Tool(
            name="ddg-text-search",
            description="Search the web for text results using DuckDuckGo",
            inputSchema={
                "type": "object",
                "properties": {
                    "keywords": {"type": "string", "description": "Search query keywords"},
                    "region": {"type": "string", "description": "Region code (e.g., wt-wt, us-en, uk-en)", "default": "wt-wt"},
                    "safesearch": {"type": "string", "enum": ["on", "moderate", "off"], "description": "Safe search level", "default": "moderate"},
                    "timelimit": {"type": "string", "enum": ["d", "w", "m", "y"], "description": "Time limit (d=day, w=week, m=month, y=year)"},
                    "max_results": {"type": "integer", "description": "Maximum number of results to return", "default": 10},
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
                    "region": {"type": "string", "description": "Region code (e.g., wt-wt, us-en, uk-en)", "default": "wt-wt"},
                    "safesearch": {"type": "string", "enum": ["on", "moderate", "off"], "description": "Safe search level", "default": "moderate"},
                    "timelimit": {"type": "string", "enum": ["d", "w", "m", "y"], "description": "Time limit (d=day, w=week, m=month, y=year)"},
                    "size": {"type": "string", "enum": ["Small", "Medium", "Large", "Wallpaper"], "description": "Image size"},
                    "color": {"type": "string", "enum": ["color", "Monochrome", "Red", "Orange", "Yellow", "Green", "Blue", "Purple", "Pink", "Brown", "Black", "Gray", "Teal", "White"], "description": "Image color"},
                    "type_image": {"type": "string", "enum": ["photo", "clipart", "gif", "transparent", "line"], "description": "Image type"},
                    "layout": {"type": "string", "enum": ["Square", "Tall", "Wide"], "description": "Image layout"},
                    "license_image": {"type": "string", "enum": ["any", "Public", "Share", "ShareCommercially", "Modify", "ModifyCommercially"], "description": "Image license type"},
                    "max_results": {"type": "integer", "description": "Maximum number of results to return", "default": 10},
                },
                "required": ["keywords"],
            },
        ),
        types.Tool(
            name="ddg-news-search",
            description="Search for news articles using DuckDuckGo",
            inputSchema={
                "type": "object",
                "properties": {
                    "keywords": {"type": "string", "description": "Search query keywords"},
                    "region": {"type": "string", "description": "Region code (e.g., wt-wt, us-en, uk-en)", "default": "wt-wt"},
                    "safesearch": {"type": "string", "enum": ["on", "moderate", "off"], "description": "Safe search level", "default": "moderate"},
                    "timelimit": {"type": "string", "enum": ["d", "w", "m"], "description": "Time limit (d=day, w=week, m=month)"},
                    "max_results": {"type": "integer", "description": "Maximum number of results to return", "default": 10},
                },
                "required": ["keywords"],
            },
        ),
        types.Tool(
            name="ddg-video-search",
            description="Search for videos using DuckDuckGo",
            inputSchema={
                "type": "object",
                "properties": {
                    "keywords": {"type": "string", "description": "Search query keywords"},
                    "region": {"type": "string", "description": "Region code (e.g., wt-wt, us-en, uk-en)", "default": "wt-wt"},
                    "safesearch": {"type": "string", "enum": ["on", "moderate", "off"], "description": "Safe search level", "default": "moderate"},
                    "timelimit": {"type": "string", "enum": ["d", "w", "m"], "description": "Time limit (d=day, w=week, m=month)"},
                    "resolution": {"type": "string", "enum": ["high", "standard"], "description": "Video resolution"},
                    "duration": {"type": "string", "enum": ["short", "medium", "long"], "description": "Video duration"},
                    "license_videos": {"type": "string", "enum": ["creativeCommon", "youtube"], "description": "Video license type"},
                    "max_results": {"type": "integer", "description": "Maximum number of results to return", "default": 10},
                },
                "required": ["keywords"],
            },
        ),
        # types.Tool(
        #     name="ddg-ai-chat",
        #     description="Chat with DuckDuckGo AI",
        #     inputSchema={
        #         "type": "object",
        #         "properties": {
        #             "keywords": {"type": "string", "description": "Message or question to send to the AI"},
        #             "model": {"type": "string", "enum": ["gpt-4o-mini", "llama-3.3-70b", "claude-3-haiku", "o3-mini", "mistral-small-3"], "description": "AI model to use", "default": "gpt-4o-mini"},
        #         },
        #         "required": ["keywords"],
        #     },
        # ),
        # New tool for fetching webpage content
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
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Handle tool execution requests.
    """
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
        
        # Perform search
        ddgs = DDGS()
        results = ddgs.text(
            keywords=keywords,
            region=region,
            safesearch=safesearch,
            timelimit=timelimit,
            max_results=max_results
        )
        
        # Format results
        formatted_results = f"Search results for '{keywords}':\n\n"
        for i, result in enumerate(results, 1):
            formatted_results += (
                f"{i}. {result.get('title', 'No title')}\n"
                f"   URL: {result.get('href', 'No URL')}\n"
                f"   {result.get('body', 'No description')}\n\n"
            )
        
        return [
            types.TextContent(
                type="text",
                text=formatted_results,
            )
        ]
    
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
        
        # Perform search
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
            max_results=max_results
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
            # Default to jpeg if uncertain
            return "image/jpeg"

        # Format results
        formatted_results = f"Image search results for '{keywords}':\n\n"
        
        text_results = []
        image_results = []
        
        for i, result in enumerate(results, 1):
            text_results.append(
                types.TextContent(
                    type="text",
                    text=f"{i}. {result.get('title', 'No title')}\n"
                         f"   Source: {result.get('source', 'Unknown')}\n"
                         f"   URL: {result.get('url', 'No URL')}\n"
                         f"   Size: {result.get('width', 'N/A')}x{result.get('height', 'N/A')}\n"
                )
            )
            
            image_url = result.get('image')
            if image_url:
                mime_type = guess_mime_type(image_url)
                image_results.append(
                    types.ImageContent(
                        type="image",
                        data=image_url,             # Instead of "url", use "data"
                        mimeType=mime_type,         # Provide the mimeType field
                        alt_text=result.get('title', 'Image search result')
                    )
                )
        
        # Interleave text and image results
        combined_results = []
        for text, image in zip(text_results, image_results):
            combined_results.extend([text, image])
        
        return combined_results
    
    elif name == "ddg-news-search":
        keywords = arguments.get("keywords")
        if not keywords:
            raise ValueError("Missing keywords")
        
        region = arguments.get("region", "wt-wt")
        safesearch = arguments.get("safesearch", "moderate")
        timelimit = arguments.get("timelimit")
        max_results = arguments.get("max_results", 10)
        
        # Perform search
        ddgs = DDGS()
        results = ddgs.news(
            keywords=keywords,
            region=region,
            safesearch=safesearch,
            timelimit=timelimit,
            max_results=max_results
        )
        
        # Format results
        formatted_results = f"News search results for '{keywords}':\n\n"
        for i, result in enumerate(results, 1):
            formatted_results += (
                f"{i}. {result.get('title', 'No title')}\n"
                f"   Source: {result.get('source', 'Unknown')}\n"
                f"   Date: {result.get('date', 'No date')}\n"
                f"   URL: {result.get('url', 'No URL')}\n"
                f"   {result.get('body', 'No description')}\n\n"
            )
        
        return [
            types.TextContent(
                type="text",
                text=formatted_results,
            )
        ]
    
    elif name == "ddg-video-search":
        keywords = arguments.get("keywords")
        if not keywords:
            raise ValueError("Missing keywords")
        
        region = arguments.get("region", "wt-wt")
        safesearch = arguments.get("safesearch", "moderate")
        timelimit = arguments.get("timelimit")
        resolution = arguments.get("resolution")
        duration = arguments.get("duration")
        license_videos = arguments.get("license_videos")
        max_results = arguments.get("max_results", 10)
        
        # Perform search
        ddgs = DDGS()
        results = ddgs.videos(
            keywords=keywords,
            region=region,
            safesearch=safesearch,
            timelimit=timelimit,
            resolution=resolution,
            duration=duration,
            license_videos=license_videos,
            max_results=max_results
        )
        
        # Format results
        formatted_results = f"Video search results for '{keywords}':\n\n"
        for i, result in enumerate(results, 1):
            formatted_results += (
                f"{i}. {result.get('title', 'No title')}\n"
                f"   Publisher: {result.get('publisher', 'Unknown')}\n"
                f"   Duration: {result.get('duration', 'Unknown')}\n"
                f"   URL: {result.get('content', 'No URL')}\n"
                f"   Published: {result.get('published', 'No date')}\n"
                f"   {result.get('description', 'No description')}\n\n"
            )
        
        return [
            types.TextContent(
                type="text",
                text=formatted_results,
            )
        ]
    
    # elif name == "ddg-ai-chat":
    #     keywords = arguments.get("keywords")
    #     if not keywords:
    #         raise ValueError("Missing keywords")
        
    #     model = arguments.get("model", "gpt-4o-mini")
        
    #     # Perform AI chat
    #     ddgs = DDGS()
    #     result = ddgs.chat(
    #         keywords=keywords,
    #         model=model
    #     )
        
    #     return [
    #         types.TextContent(
    #             type="text",
    #             text=f"DuckDuckGo AI ({model}) response:\n\n{result}",
    #         )
    #     ]

    elif name == "ddg-fetch-content":
        url = arguments.get("url")
        if not url:
            raise ValueError("Missing url")
        
        # Create a dummy context to enable logging in the fetcher.
        ctx = DummyContext()
        content = await fetcher.fetch_and_parse(url, ctx)
        return [
            types.TextContent(
                type="text",
                text=f"Fetched content from '{url}':\n\n{content}",
            )
        ]

    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    # Run the server using stdin/stdout streams
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="ddg-mcp",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )