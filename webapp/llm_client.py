from __future__ import annotations

import json
import logging
import os
from typing import Dict, List

import requests


def _get_config():
    api_key = os.getenv("LLM_API_KEY", "")
    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1/chat/completions")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    fallback = os.getenv("LLM_FALLBACK_MODELS", "")
    fallback_models = [m.strip() for m in fallback.split(",") if m.strip()]
    return api_key, base_url, model, fallback_models


def _request(messages: List[Dict[str, str]], model: str) -> str:
    api_key, base_url, _model, _fallback = _get_config()
    if not api_key:
        raise RuntimeError("LLM_API_KEY is not set.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
    }
    resp = requests.post(base_url, headers=headers, json=payload, timeout=60)
    try:
        resp.raise_for_status()
    except Exception as exc:
        logging.error("LLM request failed: status=%s body=%s", resp.status_code, resp.text)
        raise
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _model_chain() -> List[str]:
    api_key, base_url, model, fallback_models = _get_config()
    chain = [model]
    for m in fallback_models:
        if m not in chain:
            chain.append(m)
    return chain


def _request_json(messages: List[Dict[str, str]]) -> Dict[str, object]:
    last_error = None
    for model in _model_chain():
        try:
            content = _request(messages, model=model)
            return json.loads(content)
        except json.JSONDecodeError as exc:
            logging.error(
                "LLM response parse failed for model %s: %s | content=%r",
                model,
                str(exc),
                content,
            )
            last_error = exc
            continue
        except Exception as exc:
            logging.error("LLM response failed for model %s: %s", model, str(exc))
            last_error = exc
            continue
    if last_error:
        raise last_error
    raise RuntimeError("No models available")


def map_answers_to_fields(
    pdf_context: Dict[str, object],
    qa_pairs: List[Dict[str, str]],
    target_index: int | None = None,
    target_question: str | None = None,
) -> Dict[str, str]:
    system = (
        "You map answers to PDF form fields. "
        "Return JSON with an 'answers' object where keys are numeric field IDs "
        "and values are the text to fill. Return ONLY JSON. "
        "Only use provided field IDs. "
        "Be aware of user-provided spellings, as Whisper can mis-transcribe foreign names. "
        "If the user spells a name or term letter-by-letter (e.g., A-N-K-I-T), "
        "combine the letters into the final word (e.g., ANKIT). "
        "If the answer contains both a normal word and an explicit spelling, "
        "the spelling overrides the normal word. "
        "Example: input 'Ankit Bundewad. that is spelled A-N-K-I-T space B-H-A-N-D-I-W-A-D' "
        "must output 'Ankit Bhandiwad'. "
        "This is a conversational flow; follow user instructions that clarify or correct answers. "
        "If target_index is provided, ONLY map fields for that single question and do not shift "
        "answers to neighboring fields."
    )
    user = {
        "pdf_context": pdf_context,
        "qa_pairs": qa_pairs,
        "target_index": target_index,
        "target_question": target_question,
    }
    data = _request_json(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user)},
        ]
    )
    answers = data.get("answers", {})
    if not isinstance(answers, dict):
        return {}
    return {str(k): str(v) for k, v in answers.items() if str(v).strip()}


def normalize_answer(question: str, answer: str) -> str:
    system = (
        "You normalize a single user's answer for a form field. "
        "Return ONLY JSON with a single key 'value'. "
        "Be aware of user-provided spellings, as Whisper can mis-transcribe foreign names. "
        "If the user spells a name or term letter-by-letter (e.g., A-N-K-I-T), "
        "combine the letters into the final word (e.g., ANKIT). "
        "If the answer contains both a normal word and an explicit spelling, "
        "the spelling overrides the normal word. "
        "Example: input 'Ankit Bundewad. that is spelled A-N-K-I-T space B-H-A-N-D-I-W-A-D' "
        "must output 'Ankit Bhandiwad'. "
        "Follow user instructions that clarify or correct answers."
    )
    user = {"question": question, "answer": answer}
    data = _request_json(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user)},
        ]
    )
    value = data.get("value", "")
    return str(value).strip()


def generate_questions_for_targets(
    pdf_context: Dict[str, object], targets: List[Dict[str, object]]
) -> List[str]:
    system = (
        "You generate one question per target for a PDF form. "
        "Return ONLY JSON with a 'questions' array of strings. "
        "The questions array length MUST equal the number of targets and stay in the same order. "
        "Use the document text and field metadata together. "
        "If a target label is missing or not useful, infer the question from the document text. "
        "For target.type='text', ask for the value of that field. "
        "For target.type='yesno' and target.type='checkbox', ask a clear yes/no question. "
        "Keep questions short, specific, and user-facing."
    )
    user = {"pdf_context": pdf_context, "targets": targets}
    data = _request_json(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user)},
        ]
    )
    questions = data.get("questions", [])
    if not isinstance(questions, list):
        return []
    return [str(question).strip() for question in questions if str(question).strip()]


# Legacy dead/stale code moved here for reference.
#
# def generate_questions(pdf_context: Dict[str, object]) -> List[str]:
#     system = (
#         "You generate a short, ordered list of questions for a user to fill a PDF form. "
#         "Use the field labels and PDF text to ask clear, simple questions. "
#         "Return ONLY JSON with a 'questions' array of strings."
#     )
#     user = {
#         "pdf_context": pdf_context,
#     }
#     data = _request_json(
#         [
#             {"role": "system", "content": system},
#             {"role": "user", "content": json.dumps(user)},
#         ]
#     )
#     return [str(q).strip() for q in data.get("questions", []) if str(q).strip()]
#
#
