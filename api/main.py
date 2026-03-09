import os
import json
import base64
from io import BytesIO
from uuid import uuid4
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from dotenv import load_dotenv
from pypdf import PdfReader
from openai import AsyncOpenAI

from rag import rag_manager
from web_search import search_web

# -----------------------
# ENV
# -----------------------

load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_KEY:
    raise RuntimeError("OPENAI_API_KEY not found")

client = AsyncOpenAI(api_key=OPENAI_KEY)

# -----------------------
# PATH FIX (IMPORTANT FOR VERCEL)
# -----------------------

BASE_DIR = Path(__file__).resolve().parent.parent

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# -----------------------
# FASTAPI
# -----------------------

app = FastAPI()

# -----------------------
# MEMORY STORAGE
# -----------------------

conversations = {}

# -----------------------
# SMART WEB SEARCH
# -----------------------

def needs_web_search(query, rag_results):

    keywords = [
        "latest",
        "today",
        "news",
        "current",
        "2024",
        "2025",
        "2026",
        "weather",
        "price",
        "stock"
    ]

    q = query.lower()

    if not rag_results:
        return True

    return any(k in q for k in keywords)

# -----------------------
# HOME
# -----------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

# -----------------------
# CONVERSATIONS
# -----------------------

@app.post("/conversation/new")
async def new_conversation():

    cid = str(uuid4())

    conversations[cid] = {
        "title": "New Chat",
        "messages": []
    }

    return {"id": cid}

@app.get("/conversations")
async def list_conversations():

    return [
        {"id": cid, "title": c["title"]}
        for cid, c in conversations.items()
    ]

@app.get("/conversation/{cid}")
async def get_conversation(cid: str):

    return conversations.get(cid, {"messages": []})

@app.delete("/conversation/{cid}")
async def delete_conversation(cid: str):

    if cid in conversations:
        del conversations[cid]

    return {"success": True}

# -----------------------
# KNOWLEDGE BASE
# -----------------------

@app.get("/knowledge")
async def list_knowledge():

    return rag_manager.list_knowledge()

@app.delete("/knowledge/{kid}")
async def delete_knowledge(kid: str):

    rag_manager.delete_knowledge(kid)

    return {"success": True}

# -----------------------
# FILE UPLOAD
# -----------------------

@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    content = await file.read()
    text = ""

    if file.content_type == "application/pdf":

        reader = PdfReader(BytesIO(content))

        text = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    elif file.content_type.startswith("image/"):

        b64 = base64.b64encode(content).decode()

        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in detail for knowledge storage."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{file.content_type};base64,{b64}"
                            }
                        }
                    ]
                }
            ]
        )

        text = response.choices[0].message.content or ""

    else:
        text = content.decode()

    await rag_manager.add_knowledge(text)

    return {"success": True}

# -----------------------
# PERSONA
# -----------------------

@app.get("/persona")
async def get_persona():

    return {"persona": await rag_manager.get_persona()}

@app.post("/persona")
async def update_persona(persona: str = Form(...)):

    await rag_manager.set_persona(persona)

    return {"success": True}

# -----------------------
# CHAT
# -----------------------

@app.post("/chat")
async def chat(request: Request):

    data = await request.json()

    cid = data["conversation_id"]
    messages = data["messages"][-6:]

    last_query = messages[-1]["content"]

    if cid not in conversations:
        conversations[cid] = {
            "title": "New Chat",
            "messages": []
        }

    conversations[cid]["messages"] = messages

    # -----------------------
    # GENERATE TITLE
    # -----------------------

    if conversations[cid]["title"] == "New Chat" and len(messages) == 1:

        title_resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Create a short conversation title (max 4 words)"},
                {"role": "user", "content": messages[0]["content"]}
            ]
        )

        conversations[cid]["title"] = title_resp.choices[0].message.content

    # -----------------------
    # RAG
    # -----------------------

    rag_results = await rag_manager.retrieve_relevant(last_query)

    web_results = ""

    if needs_web_search(last_query, rag_results):
        web_results = await search_web(last_query)

    persona = await rag_manager.get_persona()

    rag_context = "\n\n---\n\n".join(rag_results)

    system_prompt = f"""
{persona}

Respond only to the latest user question.
Explain when needed.

Latest web information:
{web_results}

Relevant knowledge:
{rag_context}
"""

    # -----------------------
    # STREAM
    # -----------------------

    async def stream():

        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_prompt}] + messages,
            stream=True
        )

        async for chunk in response:

            if chunk.choices and chunk.choices[0].delta.content:

                data_line = json.dumps(
                    {"content": chunk.choices[0].delta.content}
                )

                yield f"data: {data_line}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream"
    )

# -----------------------
# VERCEL HANDLER
# -----------------------

handler = app