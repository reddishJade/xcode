"""向用户提问的交互工具。"""

from __future__ import annotations

import sys
from typing import Any

import questionary

from xcode.agent.types import ToolSpec


def build_question_tool() -> ToolSpec:
    """构建向用户收集选择的工具。"""

    def question(data: ToolInput) -> str:
        if not sys.stdin.isatty():
            return (
                "Cannot ask questions in non-interactive mode. Please rephrase "
                "the request or use the interactive REPL."
            )
        questions = _questions(data.get("questions"))
        answers: list[list[str]] = []
        for item in questions:
            header = item.get("header")
            message = item["question"]
            options = item.get("options")
            if options:
                display_to_label = {
                    _choice_label(option["label"], option.get("description")): option[
                        "label"
                    ]
                    for option in options
                }
                choices = list(display_to_label)
                if item.get("multiple"):
                    selected = questionary.checkbox(message, choices=choices).ask()
                    answers.append(
                        [display_to_label[str(value)] for value in (selected or [])]
                    )
                else:
                    selected = questionary.select(message, choices=choices).ask()
                    answers.append(
                        [display_to_label[str(selected)]]
                        if selected is not None
                        else []
                    )
            else:
                selected = questionary.text(message, qmark=header or "?").ask()
                answers.append([str(selected)] if selected else [])
        return _format_answers(questions, answers)

    return ToolSpec(
        name="question",
        description=(
            "Ask the user one or more questions during execution. Options are "
            "optional; without options the user can enter free text."
        ),
        input_hint='JSON: {"questions":[{"question":"Proceed?","options":[{"label":"Yes"}]}]}',
        handler=question,
        schema={
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "header": {"type": "string", "maxLength": 30},
                            "question": {"type": "string"},
                            "multiple": {"type": "boolean"},
                            "options": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "label": {"type": "string"},
                                        "description": {"type": "string"},
                                    },
                                    "required": ["label"],
                                    "additionalProperties": False,
                                },
                                "minItems": 1,
                            },
                        },
                        "required": ["question"],
                        "additionalProperties": False,
                    },
                    "minItems": 1,
                }
            },
            "required": ["questions"],
            "additionalProperties": False,
        },
        prompt_snippet="Ask the user concise multiple-choice clarification questions.",
        prompt_guidelines=(
            "Use question only when clarification or a user decision is needed.",
            "When recommending an option, make it the first option and add '(Recommended)' to its label.",
        ),
    )


def _questions(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("questions must be a non-empty array")
    questions: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, dict):
            raise ValueError("each question must be an object")
        raw_header = raw.get("header")
        header = str(raw_header).strip() if raw_header is not None else None
        text = str(raw.get("question", "")).strip()
        options = raw.get("options")
        if header is not None and (not header or len(header) > 30):
            raise ValueError("question header must be 1-30 characters when provided")
        if not text:
            raise ValueError("question text is required")
        parsed: dict[str, Any] = {
            "question": text,
            "multiple": bool(raw.get("multiple", False)),
        }
        if header is not None:
            parsed["header"] = header
        if options is not None:
            if not isinstance(options, list) or not options:
                raise ValueError(
                    f"question options must be non-empty when provided: {text}"
                )
            parsed_options = []
            for option in options:
                if not isinstance(option, dict):
                    raise ValueError(f"question option must be an object: {text}")
                label = str(option.get("label", "")).strip()
                if not label:
                    raise ValueError(f"question option label is required: {text}")
                description = option.get("description")
                parsed_option = {"label": label}
                if isinstance(description, str) and description.strip():
                    parsed_option["description"] = description.strip()
                parsed_options.append(parsed_option)
            parsed["options"] = parsed_options
        questions.append(parsed)
    return questions


def _choice_label(label: str, description: object) -> str:
    if isinstance(description, str) and description.strip():
        return f"{label} - {description.strip()}"
    return label


def _format_answers(questions: list[dict[str, Any]], answers: list[list[str]]) -> str:
    formatted = []
    for index, question in enumerate(questions):
        answer = ", ".join(answers[index]) if answers[index] else "Unanswered"
        formatted.append(f'"{question["question"]}" = "{answer}"')
    return (
        "User has answered your questions: "
        + ", ".join(formatted)
        + ". You can now continue with the user's answers in mind."
    )
