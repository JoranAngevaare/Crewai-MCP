from typing import Any, Dict, List
import requests
from mcp.server.fastmcp import FastMCP
import os
import logging
import json
from pathlib import Path
from dotenv import load_dotenv


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("search_server")

# Initialize FastMCP server
# mcp = FastMCP("search_server")

# Get current directory
current_dir = Path(__file__).parent
results_dir = current_dir / "search_results"
os.makedirs(results_dir, exist_ok=True)


from mcp.server.fastmcp import FastMCP
import glob
import os


import logging
import pathlib

logging.basicConfig(level=logging.INFO)

mcp = FastMCP("search_server")


@mcp.tool()
async def list_text_files() -> str:
    """Returns a list of txt files available to open with read_txt"""
    return "\n".join(
        glob.glob(f"{pathlib.Path(__file__).parent.resolve()}/data/txt/*.txt"),
    )


@mcp.tool()
async def read_txt(doc_path: str) -> str:
    """Returns the contect of the txt file"""
    pdf_doc = ingest_txt(doc_path)
    return pdf_doc


# TODO TYPING
def ingest_txt(doc_path: str) -> str:
    """Load txt documents."""
    if not os.path.exists(doc_path):
        return f"no file at {doc_path}"
    with open(doc_path) as f:
        content = f.read()
    return content


if __name__ == "__main__":
    logger.info("Starting Brave Search MCP Server")
    try:
        mcp.run(transport="stdio")
    except Exception as e:
        logger.exception("Search server crashed")
        # Add pause to see error in Windows
        input("Press Enter to exit...")
        raise
