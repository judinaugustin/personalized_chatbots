import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

async def search_web(query):

    res = client.search(
        query=query,
        search_depth="advanced",
        max_results=5
    )

    texts = []

    for r in res["results"]:
        texts.append(r["content"])

    return "\n\n".join(texts)