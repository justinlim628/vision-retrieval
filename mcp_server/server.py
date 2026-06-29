import os

import httpx
from mcp.server.fastmcp import FastMCP

API_BASE = os.getenv('API_BASE_URL', 'http://localhost:8000')

mcp = FastMCP('CLIP Image Search')


@mcp.tool()
def search_images(query: str, k: int = 5) -> list[dict]:
    '''Search for images similar to a text query. Returns file names and similarity scores.'''
    response = httpx.post(f'{API_BASE}/search', json={'query': query, 'k': k})
    response.raise_for_status()
    return response.json()['results']


@mcp.tool()
def get_favorites() -> list[dict]:
    '''Get all saved favorite images with their timestamps.'''
    response = httpx.get(f'{API_BASE}/favorites')
    response.raise_for_status()
    return response.json()['favorites']


@mcp.tool()
def add_favorite(file_name: str) -> dict:
    '''Add an image to favorites by file name. Safe to call multiple times for the same image.'''
    response = httpx.post(f'{API_BASE}/favorites/{file_name}')
    response.raise_for_status()
    return response.json()


if __name__ == '__main__':
    mcp.run()
