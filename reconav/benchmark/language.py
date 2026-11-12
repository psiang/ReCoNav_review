"""Verbalize symbolic benchmark tasks and check instruction surface forms."""

from __future__ import annotations

import json
from copy import deepcopy
import os
import re
import time
from importlib.metadata import version
from typing import Any


GEMINI_MODEL = "gemini-3.7-flash"
OPENAI_MODEL = "gpt-5.6-sol"


_REFERENCE_LANGUAGE_RULES = (
    "Distance FROM START always refers to the original task starting position, not a later VIA. "
    "Distance-relation scopes are benchmark-wide defaults: NEAR is within the reference's "
    "room (and floor); NEAREST TO and FARTHEST FROM compare objects on the reference's "
    "floor. Keep these defaults implicit. Do not add phrases such as 'on the same floor', "
    "'on your starting floor', 'on the floor where you started', or 'in the same room' "
    "to explain a distance relation. Preserve explicitly supplied room and named-floor "
    "qualifiers, such as 'in the kitchen' or 'on the upper floor'. "
    "An existential NEAR reference means at least one member of the named object class; "
    "express this as objects with a <reference> nearby, never as an arbitrary goal choice. "
    "Do not replace it with a singular definite reference when several references exist. "
    "Open scope excludes only exact boundary occurrences; earlier encounters are retained, "
    "even when the encountered object later becomes the endpoint. A destination room may "
    "be entered before an object GOAL inside it; do not silently remove that room from COUNT. "
    "ROOMS exclude hallway, staircase, garage, balcony and porch regions. AREAS/SPACES include "
    "every region. PORTALS exclude stairs. These words are not interchangeable: preserve the "
    "canonical collective noun in every count and avoidance clause, including singular forms. "
)


class LanguageResponseError(ValueError):
    """A provider returned a response that failed local contract validation."""

    def __init__(self, message: str, call_record: dict[str, Any]) -> None:
        super().__init__(message)
        # These are provider response statistics, not credentials. Retain
        # usage even when the generated instruction is rejected.
        self.call_record = deepcopy(call_record)


def generate_instructions(
    tasks: list[dict[str, Any]],
    *,
    provider: str,
    api_key: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate instructions with exactly one provider and validate locally."""

    if not tasks:
        raise ValueError("no symbolic tasks to verbalize")
    if provider == "gemini":
        generated, call_record = _gemini_verbalize(tasks, api_key)
    elif provider == "openai":
        generated, call_record = _gpt_verbalize(tasks, api_key)
    else:
        raise ValueError(f"unsupported language provider: {provider}")
    try:
        generated_ids = [str(item["task_id"]) for item in generated]
        if len(set(generated_ids)) != len(generated_ids):
            raise ValueError(f"{provider} returned duplicate task ids")
        by_id = {item["task_id"]: item["instruction"] for item in generated}
        expected_ids = {task["task_id"] for task in tasks}
        if set(by_id) != expected_ids:
            raise ValueError(
                f"{provider} task ids differ: "
                f"missing={sorted(expected_ids - set(by_id))}, "
                f"extra={sorted(set(by_id) - expected_ids)}"
            )
        enriched = [
            {
                **task,
                "instruction": _clean_instruction(by_id[task["task_id"]]),
            }
            for task in tasks
        ]
        _deterministic_surface_checks(enriched)
    except (KeyError, TypeError, ValueError) as error:
        raise LanguageResponseError(str(error), call_record) from error
    return enriched, call_record


def _gemini_verbalize(
    tasks: list[dict[str, Any]],
    api_key: str,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    from google import genai
    from google.genai import types

    schema = _instruction_schema(len(tasks))
    system = _instruction_system_prompt()
    payload = _instruction_payload(tasks)
    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=1.0,
        candidate_count=1,
        max_output_tokens=32768,
        response_mime_type="application/json",
        response_json_schema=schema,
    )
    http_options = types.HttpOptions(
        api_version="v1beta",
        timeout=300000,
        extra_body={"generationConfig": {"thinkingConfig": {"thinkingLevel": "HIGH"}}},
    )
    started = time.perf_counter()
    with genai.Client(api_key=api_key, http_options=http_options) as client:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=json.dumps(payload, ensure_ascii=False),
            config=config,
        )
    record = {
        "system_prompt": system,
        "user_prompt": json.dumps(payload, ensure_ascii=False),
        "raw_response": response.text,
        "provider": "google_gemini",
        "model": GEMINI_MODEL,
        "sdk_version": version("google-genai"),
        "api": "generateContent",
        "api_version": "v1beta",
        "thinking_level": "high",
        "temperature": 1.0,
        "response_id": response.response_id,
        "model_version": response.model_version,
        "latency_seconds": time.perf_counter() - started,
    }
    usage = getattr(response, "usage_metadata", None)
    if usage is not None:
        record.update(
            input_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
            output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
            reasoning_tokens=int(getattr(usage, "thoughts_token_count", 0) or 0),
            cached_input_tokens=int(getattr(usage, "cached_content_token_count", 0) or 0),
            total_tokens=int(getattr(usage, "total_token_count", 0) or 0),
        )
        record["billable_output_tokens"] = record["output_tokens"] + record["reasoning_tokens"]
    try:
        data = json.loads(response.text or "")
        return list(data["items"]), record
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise LanguageResponseError(
            f"Gemini returned invalid structured output: {error}",
            record,
        ) from error


def _instruction_schema(task_count: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string"},
                        "instruction": {"type": "string"},
                    },
                    "required": ["task_id", "instruction"],
                    "additionalProperties": False,
                },
                "minItems": task_count,
                "maxItems": task_count,
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }


def _instruction_system_prompt() -> str:
    return (_REFERENCE_LANGUAGE_RULES +
        "Write clear, natural English indoor-navigation instructions from exact "
        "symbolic intents. Preserve every goal qualifier, VIA, order, hard AVOID, "
        "COUNT operator/value/scope, and soft preference. A task may contain multiple "
        "AVOID, COUNT, or PREFER clauses; verbalize every clause separately and do "
        "not merge or drop clauses. With VIA stops, explicitly mark START-to-GOAL "
        "constraints as applying throughout the entire trip; do not attach them "
        "only to the final leg. State the endpoints of each local scope "
        "unambiguously. A named scope boundary is arrival: say 'after reaching X' "
        "or 'between reaching X and reaching Y', not 'after leaving X'. Leaving "
        "a room is a different boundary from arrival in that room. Do not add facts. Never "
        "use the word 'any' anywhere in an instruction. GOAL is never a portal. "
        "The word portal means only door, "
        "opening, or untyped connection and never stairs. Prefer 'door' when the "
        "source says door. Use 'stairs' for stair entities. Use the supplied natural "
        "floor names such as lower floor, upper floor, ground floor, or top floor; "
        "never rewrite them as numbered levels. Never mention door frames: an opening "
        "description is selection metadata, not language. Do not add a floor qualifier "
        "to a uniquely named stair. Ordering uses explicit earlier_event and "
        "later_event fields: the earlier event must happen first, irrespective "
        "of VIA declaration order. Never reverse an AFTER relation. 'then' "
        "means precedence. A VIA list alone does not impose an order: for a "
        "partial order, list the required stops with 'via' and state only the "
        "specified precedence separately; do not turn unrelated stops into a "
        "'first ... then ...' sequence. For between_without, forbidden events "
        "must not occur inside the stated interval; they remain mandatory VIA "
        "stops elsewhere in the trip. For between_only, the interval must contain "
        "all and only its listed intermediate events, with no implied order "
        "among those events. For a consecutive "
        "order, use a natural direct-transition construction such as 'go directly "
        "from X to Y'; never write 'immediately then'. A description applies "
        "to the whole named region. Preserve location qualifiers attached to VIA "
        "objects and to avoid/count/prefer object groups. When an object has both "
        "a room/floor qualifier and NEAR, "
        "make their shared attachment explicit, for example 'the lamp that is "
        "in the bedroom and has a vase nearby'; do not leave NEAR ambiguously "
        "attached to the room or floor. Keep walk-in closet as a "
        "region phrase and wardrobe as an object phrase. Use rooms only when the "
        "canonical phrase says rooms; keep areas or spaces when it says areas or "
        "spaces. Write COUNT values as English words rather than digits, for example "
        "'at least two' instead of 'at least 2'. For a hard COUNT of exactly or at "
        "most zero, prefer a natural 'without ...' construction; within the same "
        "scope this is equivalent to AVOID. For a soft PREFER COUNT of exactly or "
        "at most zero, use a natural 'preferably without ...' construction, such as "
        "'preferably without entering other areas'. Never say 'at most zero' or "
        "'exactly zero'. Return exactly one instruction per task."
        " Preserve NEAR, NEAREST TO, and FARTHEST FROM qualifiers exactly; NEAR has "
        "a fixed internal two-metre meaning, but do not expose that numeric radius."
    )


def _instruction_payload(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "task_id": task["task_id"],
            "family": task["family"],
            "canonical_intent": language_intent(task["canonical_intent"]),
        }
        for task in tasks
    ]


def language_intent(intent: dict[str, Any]) -> dict[str, Any]:
    """Expose temporal direction without the ambiguous first/second names.

    This is a lossless presentation for verbalization, not a change to the
    stored DSL, proof or task. AFTER(A,B) means B occurs before A.
    """
    def order_view(order):
        item = deepcopy(order)
        kind = item.get("kind")
        if kind in {"before", "after", "next"}:
            first, second = item.pop("first"), item.pop("second")
            earlier, later = (second, first) if kind == "after" else (first, second)
            item.update(kind="consecutive" if kind == "next" else "precedence",
                        earlier_event=earlier, later_event=later)
        elif kind in {"between_only", "between_without"}:
            item["earlier_event"] = item.pop("first")
            item["later_event"] = item.pop("second")
            member_key = ("required_and_only_intermediate_events" if kind == "between_only"
                          else "forbidden_intermediate_events")
            item[member_key] = item.pop("names")
        return item

    result = deepcopy(intent)
    result["orders"] = [order_view(order) for order in result.get("orders", [])]
    for constraint in result.get("constraints", []):
        if isinstance(constraint.get("order"), dict):
            constraint["order"] = order_view(constraint["order"])
    return result


def _gpt_verbalize(
    tasks: list[dict[str, Any]],
    api_key: str,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    from reconav.llm.shared import SharedLLMClient
    from reconav.method.runtime.model_calls import RecordedClient

    os.environ["OPENAI_API_KEY"] = api_key
    client = RecordedClient(SharedLLMClient("openai"), max_input_characters=512000)
    with client.at("benchmark_instruction_generation"):
        text = client.generate(
            _instruction_system_prompt(),
            json.dumps(_instruction_payload(tasks), ensure_ascii=False),
            max_tokens=8192,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "benchmark_instruction_generation",
                    "strict": True,
                    "schema": _instruction_schema(len(tasks)),
                },
            },
        )
    call = client.calls[0]
    record = {
        "provider": "openai",
        "model": call["provider_response"].get("model") or OPENAI_MODEL,
        "model_calls": client.calls,
        **call["usage"],
    }
    try:
        data = json.loads(text)
        return list(data["items"]), record
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise LanguageResponseError(
            f"GPT returned invalid instruction output: {error}",
            record,
        ) from error


def validate_collective_count_language(task: dict[str, Any]) -> None:
    """Reject ontology-changing count paraphrases, without inspecting model scores."""
    symbolic = task.get("symbolic_ir")
    if not isinstance(symbolic, dict):
        return
    groups = {"room": "room", "rooms": "room", "area": "area", "areas": "area",
              "space": "area", "spaces": "area", "portal": "portal", "portals": "portal",
              "door": "door", "doors": "door", "stair": "stair", "stairs": "stair",
              "floor": "level", "floors": "level", "level": "level", "levels": "level"}
    numbers = {word: n for n, word in enumerate("zero one two three four five six seven eight nine ten".split())}
    operators = {"at least": "at_least", "at most": "at_most", "exactly": "exactly",
                 "no more than": "at_most", "no fewer than": "at_least"}
    expected = {(c.get("op"), c.get("count"), groups[c["phrase"]])
                for c in symbolic.get("constraints", [])
                if c.get("kind") in {"count", "prefer_count"} and c.get("phrase") in groups}
    pattern = r"\b(at least|at most|exactly|no more than|no fewer than)\s+(zero|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:(?:distinct|different)\s+)?(rooms?|areas?|spaces?|portals?|doors?|stairs?|floors?|levels?)\b"
    for operator, count, noun in re.findall(pattern, task.get("instruction", "").lower()):
        value = (operators[operator], numbers[count], groups[noun])
        if value not in expected:
            raise ValueError(f"{task['task_id']} changes a collective count denotation: {operator} {count} {noun}")


def validate_order_language(task: dict[str, Any]) -> None:
    """Reject an explicit reversed BEFORE/AFTER between exact named phrases.

    This is a conservative local guard, not a complete natural-language parser.
    Unrecognized paraphrases still require the separate fidelity audit.
    """
    intent = language_intent(task.get("canonical_intent", {}))
    phrases = {v["name"]: v["phrase"] for v in intent.get("vias", [])}
    phrases["GOAL"] = intent.get("goal", "")
    text = task.get("instruction", "")
    bridge = re.compile(
        r"\s*,?\s*(before|after)\s+(?:(?:visiting|reaching|entering|"
        r"passing(?:\s+(?:by|through))?|going\s+to)\s+)?",
        re.IGNORECASE,
    )
    for order in intent.get("orders", []):
        if order.get("kind") not in {"precedence", "consecutive"}:
            continue
        earlier, later = order["earlier_event"], order["later_event"]
        first, second = phrases.get(earlier), phrases.get(later)
        if not first or not second or first.lower() == second.lower():
            continue
        for a, b in ((earlier, later), (later, earlier)):
            for left in re.finditer(re.escape(phrases[a]), text, re.IGNORECASE):
                for right in re.finditer(re.escape(phrases[b]), text, re.IGNORECASE):
                    if right.start() < left.end():
                        continue
                    link = bridge.fullmatch(text[left.end():right.start()])
                    if link is None:
                        continue
                    observed = (a, b) if link.group(1).lower() == "before" else (b, a)
                    if observed != (earlier, later):
                        raise ValueError(f"{task['task_id']} reverses temporal order: expected {earlier} before {later}")


def _deterministic_surface_checks(tasks: list[dict[str, Any]]) -> None:
    instructions = [task["instruction"] for task in tasks]
    if len(set(instructions)) != len(instructions):
        raise ValueError("language provider produced duplicate instructions")
    for task in tasks:
        text = task["instruction"].lower()
        validate_collective_count_language(task)
        validate_order_language(task)
        if re.search(r"\bany\s+[a-z]", text):
            raise ValueError(f"{task['task_id']} uses forbidden any-target language")
        if re.search(r"\bwithin\s+\d", text):
            raise ValueError(f"{task['task_id']} uses an unsupported distance qualifier")
        if re.search(
            r"\b(?:same\s+(?:floor|level)|(?:your|the)\s+(?:starting|start)\s+(?:floor|level)|"
            r"(?:floor|level)\s+(?:where|that)\s+you\s+(?:started|began))\b",
            text,
        ):
            raise ValueError(
                f"{task['task_id']} verbalizes an implicit distance-relation floor scope"
            )
        if re.search(r"\b(?:weight|priority)\b|\brobot\s+width\b", text):
            raise ValueError(f"{task['task_id']} exposes an internal construction parameter")
        if re.search(r"\b(?:levels?|floors?)\s+\d|\bdoor[- ]?frame", text):
            raise ValueError(
                f"{task['task_id']} uses unnatural level or door-frame language"
            )
        if re.search(r"\b(?:at least|at most|exactly)\s+\d+\b", text):
            raise ValueError(f"{task['task_id']} writes a count value as digits")
        if re.search(r"\b(?:at most|exactly)\s+zero\b", text):
            raise ValueError(
                f"{task['task_id']} uses an unnatural zero-count construction"
            )
        if "immediately then" in text:
            raise ValueError(
                f"{task['task_id']} uses an unnatural consecutive-order phrase"
            )
        if re.search(
            r"\b(?:stair|stairs|staircase)\s+portal\b|"
            r"\bportal\s+(?:stair|stairs|staircase)\b",
            text,
        ):
            raise ValueError(f"{task['task_id']} calls stairs a portal")
        if "immediately" in text and not any(
            order["kind"] == "next" for order in task["canonical_intent"]["orders"]
        ):
            raise ValueError(f"{task['task_id']} adds unsupported consecutive ordering")


def _clean_instruction(value: str) -> str:
    text = " ".join(str(value).strip().split())
    if not text:
        raise ValueError("empty generated instruction")
    number_words = {
        "0": "zero",
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
        "10": "ten",
    }

    def replace_count_value(match: re.Match[str]) -> str:
        operator, value = match.groups()
        return f"{operator} {number_words.get(value, value)}"

    return re.sub(
        r"\b(at least|at most|exactly)\s+(\d+)\b",
        replace_count_value,
        text,
        flags=re.IGNORECASE,
    )


def with_canonical_intent(task: dict[str, Any]) -> dict[str, Any]:
    """Recover the language-generation view from released task annotations."""
    result = deepcopy(task)
    if "canonical_intent" in result:
        return result
    symbolic = result["symbolic_ir"]
    goal = symbolic["goal"]
    phrase = "the " + goal["head"]
    for qualifier in goal.get("qualifiers", []):
        kind = qualifier["kind"]
        if kind == "in_region":
            phrase += " in the " + qualifier["label"]
        elif kind == "in_level":
            phrase += " on " + qualifier["label"]
        elif kind in {"near", "nearest_to", "farthest_from"}:
            relation = {"near": "near", "nearest_to": "nearest to",
                        "farthest_from": "farthest from"}[kind]
            phrase += " " + relation + " " + qualifier["reference"]["phrase"]
        else:
            raise ValueError(f"unsupported goal qualifier: {kind}")
    result["canonical_intent"] = {
        "goal": phrase,
        "vias": [{"name": via["name"], "phrase": via["phrase"]}
                 for via in symbolic.get("vias", [])],
        "orders": deepcopy(symbolic.get("orders", [])),
        "constraints": deepcopy(symbolic.get("constraints", [])),
    }
    return result


FIDELITY_SYSTEM_PROMPT = (
    'You audit dataset annotation fidelity, not model performance. Compare the instruction against '
    'the exact canonical intent. Report missing, reversed, added, ambiguous or incorrectly attached '
    'semantics. Judge every supplied task; do not invent issues just to find errors. All named VIA '
    'arrivals are mandatory unless explicitly soft. VIA list order is declaration order, NOT an '
    'unstated temporal order. earlier_event MUST happen before later_event; consecutive means '
    'immediate in the named event sequence, naturally expressed by direct transition. Constraints use'
    ' open AFTER/UNTIL arrival intervals. Check every target qualifier, quantifier, ordering '
    'direction, modality, count operator/value and scope. A scope is temporal, not a spatial '
    'qualifier on a counted class. Treat common faithful synonyms as equivalent. Entity descriptions '
    'are authoritative; no scene lookup needed. '
    + _REFERENCE_LANGUAGE_RULES
)
