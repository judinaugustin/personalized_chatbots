import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from typing import List
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DEFAULT_PERSONA = "You are Riya's personal assistant. Be warm, supportive and helpful. Riya is 21 years old,studying nursing ausbildung in germany. she is from ponthenpuzha , kerala. she has someone named judin who is like a father but a big brother to her. she has parents , younger brother royal,young sisters rona and roma."


class KnowledgeItem:

    def __init__(self, id: str, text: str, embedding: List[float]):
        self.id = id
        self.text = text
        self.embedding = embedding


class RAGManager:

    def __init__(self):
        self.knowledge: List[KnowledgeItem] = []
        self.persona = DEFAULT_PERSONA

    # -----------------------
    # PERSONA
    # -----------------------

    async def set_persona(self, persona: str):
        self.persona = persona

    async def get_persona(self):
        return self.persona

    # -----------------------
    # ADD KNOWLEDGE
    # -----------------------

    async def add_knowledge(self, text: str):

        if not text or len(text.strip()) < 10:
            return

        emb = await client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )

        self.knowledge.append(
            KnowledgeItem(
                str(len(self.knowledge)),
                text,
                emb.data[0].embedding
            )
        )

    # -----------------------
    # LIST KNOWLEDGE
    # -----------------------

    def list_knowledge(self):

        return [
            {
                "id": k.id,
                "preview": k.text[:120]
            }
            for k in self.knowledge
        ]

    # -----------------------
    # DELETE KNOWLEDGE
    # -----------------------

    def delete_knowledge(self, kid: str):

        self.knowledge = [
            k for k in self.knowledge
            if k.id != kid
        ]

    # -----------------------
    # COSINE SIMILARITY
    # -----------------------

    def cosine(self, a, b):

        dot = sum(x * y for x, y in zip(a, b))

        na = sum(x * x for x in a) ** 0.5

        nb = sum(x * x for x in b) ** 0.5

        return dot / (na * nb + 1e-10)

    # -----------------------
    # RETRIEVE
    # -----------------------

    async def retrieve_relevant(self, query: str, top_k: int = 4):

        if not self.knowledge:
            return []

        emb = await client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )

        q_emb = emb.data[0].embedding

        scored = [
            (self.cosine(q_emb, k.embedding), k.text)
            for k in self.knowledge
        ]

        scored.sort(reverse=True)

        return [text for _, text in scored[:top_k]]


rag_manager = RAGManager()