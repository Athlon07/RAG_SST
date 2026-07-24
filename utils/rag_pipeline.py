"""Turns retrieved chunks + a question into a grounded answer via Groq's free LLM API.

The system prompt explicitly forbids outside knowledge and gives the model
a fixed fallback sentence to use when the answer isn't in the retrieved
context, which is the main lever against hallucination here (the other
lever is simply not passing in anything the model could hallucinate from).

Conversation memory works in two places:
1. condense_question() rewrites a follow-up like "elaborate on that" into a
   standalone question using recent history, BEFORE it's used for vector
   retrieval (embedding search can't resolve pronouns/references on its own).
2. answer_question() includes recent conversation turns as prior chat
   messages sent to the model, so the final answer can naturally refer back
   to what was already discussed.
"""
from groq import Groq

SYSTEM_PROMPT = """You are a careful document assistant. Answer the user's question using \
ONLY the context excerpts provided below, which come from one or more documents the user \
provided. You may also use the earlier conversation turns to understand what the user is \
referring to, but the actual facts in your answer must still come only from the context \
excerpts, never from outside knowledge.

Rules:
- Do not use any outside knowledge, even if you are confident about the answer.
- If the answer is not contained in the context, respond with exactly this sentence \
and nothing else: "I could not find the answer to this in the document."
- When you do answer, cite the document name and location shown in brackets for each \
excerpt you used, e.g. (resume.pdf, Page 2) or (report.xlsx, Sheet: Revenue) or \
(notes.docx, Section: Introduction) - whatever label is given.
- If relevant information appears in more than one document, mention each source you used.
- Be concise and direct. Do not speculate, guess, or fabricate information.
"""

# how many prior turns (user+assistant messages) to keep as memory
MAX_HISTORY_MESSAGES = 6


def build_context_block(retrieved):
    blocks = []
    for doc, meta, dist in retrieved:
        location = meta.get("location", "Unknown")
        source = meta.get("source", "document")
        blocks.append(f"[{source} - {location}]\n{doc}")
    return "\n\n---\n\n".join(blocks)


def condense_question(api_key: str, model: str, question: str, history=None, max_tokens: int = 200) -> str:
    """Rewrite a follow-up question into a standalone one using recent chat
    history, so vector retrieval has something concrete to search for.

    If there's no history yet, or the rewrite fails for any reason, falls
    back to the original question unchanged (retrieval still runs either way).
    """
    if not history:
        return question

    recent = history[-MAX_HISTORY_MESSAGES:]
    history_text = "\n".join(f"{h['role'].capitalize()}: {h['content']}" for h in recent)

    prompt = f"""Conversation history:
{history_text}

Follow-up question: {question}

Rewrite the follow-up question into a standalone question that includes any context \
needed to understand it on its own (e.g. replace "that", "it", "those" with what they \
refer to). If it's already standalone, just repeat it unchanged. Reply with ONLY the \
rewritten question, nothing else."""

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        rewritten = response.choices[0].message.content.strip()
        return rewritten if rewritten else question
    except Exception:
        return question


def answer_question(
    api_key: str, model: str, question: str, retrieved, history=None, max_tokens: int = 1000
):
    """
    Args:
        api_key: Groq API key (free — get one at console.groq.com)
        model: model name, e.g. "llama-3.3-70b-versatile"
        question: user's original question (as typed, not the rewritten one)
        retrieved: output of VectorStore.query()
        history: list of {"role": "user"|"assistant", "content": str} prior turns

    Returns:
        (answer_text: str, sources: list[dict])
    """
    context = build_context_block(retrieved)

    if not context.strip():
        return "I could not find the answer to this in the document.", []

    user_prompt = f"""Context excerpts from the document:

{context}

Question: {question}

Answer using only the context above."""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-MAX_HISTORY_MESSAGES:])
    messages.append({"role": "user", "content": user_prompt})

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0.1,  # low temperature keeps answers grounded rather than creative
        messages=messages,
    )

    answer_text = response.choices[0].message.content
    sources = [
        {
            "source": meta.get("source", "document"),
            "location": meta.get("location", "Unknown"),
            "excerpt": doc[:220],
        }
        for doc, meta, dist in retrieved
    ]
    return answer_text, sources
