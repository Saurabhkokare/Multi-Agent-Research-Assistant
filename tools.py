from tavily import TavilyAPI
#from crossref.restful import Works
import os
from dotenv import load_dotenv

load_dotenv()

Tavily_api_key=os.environ['TAVILY_API_KEY']

def fetch_related_papers(query: str) -> str:
    """Fetches related papers using APIs."""
    tavily = TavilyAPI(api_key=Tavily_api_key)
    results = tavily.search(query)
    return results
