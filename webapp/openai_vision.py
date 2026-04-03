from __future__ import annotations

import base64
import json
import os
import time
from io import BytesIO
from typing import Dict, List, Optional

import requests
from PIL import Image


OPENAI_API_URL = "https://api.openai.com/v1/responses"


class OpenAIQuestionGenerationError(RuntimeError):
    pass


def _encode_image_bytes(image_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _error_message_from_response(response: requests.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        text = response.text.strip()
        if text:
            return text
        return f"OpenAI request failed with status {response.status_code}."
    error = data.get("error")
    if isinstance(error, dict):
        message = str(error.get("message", "")).strip()
        if message:
            return message
    return f"OpenAI request failed with status {response.status_code}."


def _build_missing_schema() -> Dict[str, object]:
    return {
        "name": "missing_fields",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "missing_fields": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "type": {"type": "string", "enum": ["text"]},
                            "label": {"type": "string"},
                            "value_bbox": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "x0": {"type": "number", "minimum": 0, "maximum": 1},
                                    "y0": {"type": "number", "minimum": 0, "maximum": 1},
                                    "x1": {"type": "number", "minimum": 0, "maximum": 1},
                                    "y1": {"type": "number", "minimum": 0, "maximum": 1},
                                },
                                "required": ["x0", "y0", "x1", "y1"],
                            },
                        },
                        "required": ["type", "label", "value_bbox"],
                    },
                }
            },
            "required": ["missing_fields"],
        },
    }


def _build_questions_schema() -> Dict[str, object]:
    return {
        "name": "qa_questions",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "id": {"type": "integer", "minimum": 0},
                            "question": {"type": "string"},
                        },
                        "required": ["id", "question"],
                    },
                }
            },
            "required": ["questions"],
        },
    }


def questions_from_image(
    image_bytes: bytes,
    mime_type: str,
    targets: List[Dict[str, object]],
    page_index: int = 0,
    model: str = "gpt-4o",
) -> Optional[List[str]]:
    print(f"[openai] questions_from_image start page={page_index} targets={len(targets)}")
    api_key = os.getenv("OPENAI_API_KEY", "") or os.getenv("LLM_API_KEY", "")
    if not api_key:
        raise OpenAIQuestionGenerationError(
            "OpenAI API key is not set for question generation."
        )
    model = os.getenv("QA_QUESTION_MODEL", model)
    if not targets:
        return []

    if mime_type.startswith("image/"):
        image = Image.open(BytesIO(image_bytes))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        mime_type = "image/png"

    type_by_id: Dict[int, str] = {}
    required_ids = []
    for target in targets:
        try:
            tid = int(target.get("id", -1))
        except Exception:
            continue
        if tid < 0:
            continue
        ttype = str(target.get("type", "text"))
        type_by_id[tid] = ttype
        required_ids.append(tid)
    required_ids = sorted(set(required_ids))
    if len(required_ids) != len(targets):
        return None

    prompt = (
        "You are given a PDF form with AcroForm fields already placed. "
        "I am also providing a JSON list of the fields with ids, types, and their page locations.\n\n"
        "Create a list of questions, one for each textbox, checkbox, or yes/no checkbox pair.\n"
        "This is for a program, so return ONLY JSON.\n\n"
        "Rules:\n"
        "- 1 question per field id (1:1, no merging).\n"
        "- Keep the questions short and clear.\n"
        "- For text fields, ask for the value.\n"
        "- For checkbox / yes-no pairs, ask a yes/no question.\n\n"
        "If context_bbox is provided for a field, use ONLY the text inside that box. "
        "If the label spans multiple lines within that box (including lines starting with "
        "'OR'), combine all lines into one question instead of picking just one line.\n\n"
        "Output JSON format:\n"
        "{\n"
        '  "questions": [\n'
        '    {"id": 0, "question": "..." },\n'
        '    {"id": 1, "question": "..." }\n'
        "  ]\n"
        "}\n\n"
        "Use the ids exactly as provided."
    )

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload_base = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_text",
                        "text": json.dumps(
                            {"targets": targets, "target_ids": required_ids}, ensure_ascii=True
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": _encode_image_bytes(image_bytes, mime_type),
                    },
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "qa_questions",
                "schema": _build_questions_schema()["schema"],
            }
        },
    }

    for attempt in range(2):
        payload = payload_base
        if attempt == 1:
            extra = (
                "Your last output missed some ids. You must return one entry per id "
                f"in this exact list: {required_ids}. Include each id exactly once."
            )
            payload = json.loads(json.dumps(payload_base))
            payload["input"][0]["content"][0]["text"] = prompt + " " + extra
        print("[openai] sending questions request")
        response = None
        for retry in range(4):
            try:
                response = requests.post(
                    OPENAI_API_URL, headers=headers, data=json.dumps(payload), timeout=90
                )
            except requests.RequestException as exc:
                raise OpenAIQuestionGenerationError(
                    f"OpenAI question generation request failed: {exc}"
                ) from exc
            if response.status_code != 429:
                break
            wait_s = 0.5
            try:
                wait_s = float(response.headers.get("Retry-After", wait_s))
            except Exception:
                wait_s = 0.5
            print(f"[openai] rate limited, retrying in {wait_s:.2f}s")
            time.sleep(wait_s)
        if response is None:
            return None
        try:
            response.raise_for_status()
        except requests.RequestException:
            print("OpenAI vision questions error status:", response.status_code)
            print("OpenAI vision questions error body:", response.text)
            raise OpenAIQuestionGenerationError(
                _error_message_from_response(response)
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise OpenAIQuestionGenerationError(
                f"OpenAI question generation returned invalid JSON: {exc}"
            ) from exc
        print("OpenAI vision questions response:", json.dumps(data, ensure_ascii=True))
        text = data.get("output_text")
        if not text:
            try:
                text = data["output"][0]["content"][0]["text"]
            except Exception:
                text = ""
        if not text:
            print("[openai] no questions output text found")
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            print(f"OpenAI vision questions parse error: {exc}")
            continue
        questions = parsed.get("questions", [])
        if not isinstance(questions, list):
            continue
        question_map: Dict[int, str] = {}
        for item in questions:
            try:
                qid = int(item.get("id"))
            except Exception:
                continue
            if qid in question_map:
                continue
            qtext = str(item.get("question", "")).strip()
            if not qtext:
                ttype = type_by_id.get(qid, "text")
                if ttype in ("checkbox", "yesno"):
                    qtext = "Please answer yes or no for this item."
                else:
                    qtext = "Please provide the value for this field."
            question_map[qid] = qtext
        if all(qid in question_map for qid in required_ids):
            return [question_map[qid] for qid in required_ids]
    return None


def detect_missing_fields(
    image_bytes: bytes,
    mime_type: str,
    existing_fields: List[Dict[str, object]],
    page_index: int = 0,
    model: str = "gpt-4o-mini",
) -> Optional[List[Dict[str, object]]]:
    print(
        f"[openai] detect_missing_fields start page={page_index} existing={len(existing_fields)}"
    )
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("[openai] OPENAI_API_KEY missing")
        return None

    if mime_type.startswith("image/"):
        image = Image.open(BytesIO(image_bytes))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        mime_type = "image/png"

    existing_json = json.dumps(existing_fields, ensure_ascii=True)
    prompt = (
        "You are given a form image and a list of existing input boxes that were "
        "detected by another model. Your task is to return only the missing text "
        "input areas. Do NOT return checkboxes or yes/no squares. "
        "Return only empty areas meant for user text input. "
        "Use value_bbox for the empty input area and provide a short label. "
        "Do NOT return anything that overlaps an existing box. "
        f"Page index is {page_index}.\n"
        f"Existing boxes (normalized): {existing_json}"
    )

    payload = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": _encode_image_bytes(image_bytes, mime_type),
                    },
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "missing_fields",
                "schema": _build_missing_schema()["schema"],
            }
        },
    }

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    print("[openai] sending missing-fields request")
    try:
        response = requests.post(
            OPENAI_API_URL, headers=headers, data=json.dumps(payload), timeout=90
        )
    except requests.RequestException as exc:
        print(f"OpenAI vision missing-fields request failed: {exc}")
        return None
    try:
        response.raise_for_status()
    except requests.RequestException:
        print("OpenAI vision missing-fields error status:", response.status_code)
        print("OpenAI vision missing-fields error body:", response.text)
        return None
    try:
        data = response.json()
    except ValueError as exc:
        print(f"OpenAI vision missing-fields invalid JSON response: {exc}")
        return None
    print("OpenAI vision missing-fields response:", json.dumps(data, ensure_ascii=True))
    text = data.get("output_text")
    if not text:
        try:
            text = data["output"][0]["content"][0]["text"]
        except Exception:
            text = ""
    if not text:
        print("[openai] no missing-fields output text found")
        return None

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        print(f"OpenAI vision missing-fields parse error: {exc}")
        return None
    fields = parsed.get("missing_fields", [])
    if not isinstance(fields, list):
        return None
    return fields


# Legacy dead/stale code moved here for reference.
#
# def _build_schema() -> Dict[str, object]:
#     return {
#         "name": "form_fields",
#         "schema": {
#             "type": "object",
#             "additionalProperties": False,
#             "properties": {
#                 "fields": {
#                     "type": "array",
#                     "items": {
#                         "type": "object",
#                         "additionalProperties": False,
#                         "properties": {
#                             "page": {"type": "integer", "minimum": 0},
#                             "type": {
#                                 "type": "string",
#                                 "enum": ["text", "checkbox"],
#                             },
#                             "label": {"type": "string"},
#                             "label_bbox": {
#                                 "type": "object",
#                                 "additionalProperties": False,
#                                 "properties": {
#                                     "x0": {"type": "number", "minimum": 0, "maximum": 1},
#                                     "y0": {"type": "number", "minimum": 0, "maximum": 1},
#                                     "x1": {"type": "number", "minimum": 0, "maximum": 1},
#                                     "y1": {"type": "number", "minimum": 0, "maximum": 1},
#                                 },
#                                 "required": ["x0", "y0", "x1", "y1"],
#                             },
#                             "value_bbox": {
#                                 "type": "object",
#                                 "additionalProperties": False,
#                                 "properties": {
#                                     "x0": {"type": "number", "minimum": 0, "maximum": 1},
#                                     "y0": {"type": "number", "minimum": 0, "maximum": 1},
#                                     "x1": {"type": "number", "minimum": 0, "maximum": 1},
#                                     "y1": {"type": "number", "minimum": 0, "maximum": 1},
#                                 },
#                                 "required": ["x0", "y0", "x1", "y1"],
#                             },
#                         },
#                         "required": [
#                             "page",
#                             "type",
#                             "label",
#                             "label_bbox",
#                             "value_bbox",
#                         ],
#                     },
#                 }
#             },
#             "required": ["fields"],
#         },
#     }
#
#
# def _build_label_schema() -> Dict[str, object]:
#     return {
#         "name": "box_labels",
#         "schema": {
#             "type": "object",
#             "additionalProperties": False,
#             "properties": {
#                 "labels": {
#                     "type": "array",
#                     "items": {
#                         "type": "object",
#                         "additionalProperties": False,
#                         "properties": {
#                             "id": {"type": "integer", "minimum": 0},
#                             "label": {"type": "string"},
#                         },
#                         "required": ["id", "label"],
#                     },
#                 }
#             },
#             "required": ["labels"],
#         },
#     }
#
#
# def label_boxes_from_image(
#     image_bytes: bytes,
#     mime_type: str,
#     boxes: List[Dict[str, object]],
#     page_index: int = 0,
#     model: str = "gpt-4o-mini",
# ) -> Optional[Dict[int, str]]:
#     print(f"[openai] label_boxes_from_image start page={page_index} boxes={len(boxes)}")
#     api_key = os.getenv("OPENAI_API_KEY", "") or os.getenv("LLM_API_KEY", "")
#     if not api_key:
#         print("[openai] OPENAI_API_KEY missing")
#         return None
#
#     if mime_type.startswith("image/"):
#         image = Image.open(BytesIO(image_bytes))
#         buffer = BytesIO()
#         image.save(buffer, format="PNG")
#         image_bytes = buffer.getvalue()
#         mime_type = "image/png"
#
#     prompt = (
#         "You are given a form image and a list of input boxes with normalized "
#         "bounding boxes (0-1, origin top-left). For each box, read the question "
#         "text or label that corresponds to that input area. Treat the box as the "
#         "anchor: prefer the nearest text on the same row immediately to the left. "
#         "If none, use the nearest block directly above within 2 line heights. "
#         "Do NOT use section headers. Ignore parenthetical examples (e.g., lines "
#         "starting with 'e.g.') unless there is no other label. For yes/no pairs, "
#         "return the row/service label, not the words 'Yes' or 'No'. "
#         "If no clear label is found, return an empty string for that box id. "
#         "Return ONLY JSON per the schema."
#         f" Page index is {page_index}."
#     )
#
#     payload = {
#         "model": model,
#         "input": [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "input_text", "text": prompt},
#                     {
#                         "type": "input_text",
#                         "text": json.dumps({"boxes": boxes}, ensure_ascii=True),
#                     },
#                     {
#                         "type": "input_image",
#                         "image_url": _encode_image_bytes(image_bytes, mime_type),
#                     },
#                 ],
#             }
#         ],
#         "text": {
#             "format": {
#                 "type": "json_schema",
#                 "name": "box_labels",
#                 "schema": _build_label_schema()["schema"],
#             }
#         },
#     }
#
#     headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
#     print("[openai] sending label request")
#     response = requests.post(
#         OPENAI_API_URL, headers=headers, data=json.dumps(payload), timeout=90
#     )
#     try:
#         response.raise_for_status()
#     except Exception:
#         print("OpenAI vision label error status:", response.status_code)
#         print("OpenAI vision label error body:", response.text)
#         raise
#     data = response.json()
#     print("OpenAI vision label response:", json.dumps(data, ensure_ascii=True))
#     text = data.get("output_text")
#     if not text:
#         try:
#             text = data["output"][0]["content"][0]["text"]
#         except Exception:
#             text = ""
#     if not text:
#         print("[openai] no label output text found")
#         return None
#     parsed = json.loads(text)
#     labels = parsed.get("labels", [])
#     if not isinstance(labels, list):
#         return None
#     out: Dict[int, str] = {}
#     for item in labels:
#         try:
#             idx = int(item.get("id"))
#         except Exception:
#             continue
#         label = str(item.get("label", "")).strip()
#         out[idx] = label
#     return out
#
#
# def detect_fields_from_image(
#     image_bytes: bytes,
#     mime_type: str,
#     page_index: int = 0,
#     model: str = "gpt-4o-mini",
# ) -> Optional[List[Dict[str, object]]]:
#     print(f"[openai] detect_fields_from_image start page={page_index} mime={mime_type} bytes={len(image_bytes)}")
#     api_key = os.getenv("OPENAI_API_KEY", "")
#     if not api_key:
#         print("[openai] OPENAI_API_KEY missing")
#         return None
#
#     prompt = (
#         "You detect form input fields in a document image. "
#         "Return bounding boxes for empty text inputs (lines/boxes) and checkboxes. "
#         "Provide label_bbox for the label text, and value_bbox for the empty input area. "
#         "The value_bbox must be the actual input area to be filled, not the label text. "
#         "Include small square boxes in numbered row columns and checkbox-sized squares "
#         "next to text (e.g. 'No middle name'). "
#         "Only include fields clearly intended for user input. "
#         "If the value area is not visible, omit the field entirely. "
#         "If you cannot find an empty input area, omit that field. "
#         "Use normalized coordinates (0-1) with origin at top-left. "
#         "If unsure, omit the field. "
#         f"The page index is {page_index}."
#     )
#
#     if mime_type.startswith("image/"):
#         image = Image.open(BytesIO(image_bytes))
#         buffer = BytesIO()
#         image.save(buffer, format="PNG")
#         image_bytes = buffer.getvalue()
#         mime_type = "image/png"
#
#     payload = {
#         "model": model,
#         "input": [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "input_text", "text": prompt},
#                     {
#                         "type": "input_image",
#                         "image_url": _encode_image_bytes(image_bytes, mime_type),
#                     },
#                 ],
#             }
#         ],
#         "text": {
#             "format": {"type": "json_schema", "name": "form_fields", "schema": _build_schema()["schema"]}
#         },
#     }
#
#     headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
#     print("[openai] sending request")
#     response = requests.post(
#         OPENAI_API_URL, headers=headers, data=json.dumps(payload), timeout=90
#     )
#     try:
#         response.raise_for_status()
#     except Exception:
#         print("OpenAI vision error status:", response.status_code)
#         print("OpenAI vision error body:", response.text)
#         raise
#     data = response.json()
#     print("OpenAI vision response:", json.dumps(data, ensure_ascii=True))
#     text = data.get("output_text")
#     if not text:
#         try:
#             text = data["output"][0]["content"][0]["text"]
#         except Exception:
#             text = ""
#     if not text:
#         print("[openai] no output text found")
#         return None
#     parsed = json.loads(text)
#     fields = parsed.get("fields", [])
#     if not isinstance(fields, list):
#         return None
#     return fields
