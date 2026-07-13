"""向用户提问的交互工具。"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

import questionary

from xcode.agent.types import ToolInput, ToolSpec

QuestionPromptHandler = Callable[[list[dict[str, Any]]], list[list[str]]]


class _QuestionToolHandler:
    """在不同前端之间路由问题交互。"""

    def __init__(self) -> None:
        self.prompt_handler: QuestionPromptHandler | None = None

    def __call__(
        self, data: ToolInput, _on_update: Callable[[str], None] | None = None
    ) -> str:
        prompt_handler = self.prompt_handler
        if prompt_handler is None and not sys.stdin.isatty():
            return (
                "Cannot ask questions in non-interactive mode. Please rephrase "
                "the request or use the interactive REPL."
            )
        questions = _questions(data.get("questions"))
        if prompt_handler is not None:
            return _format_answers(questions, prompt_handler(questions))
        return _format_answers(questions, _ask_with_questionary(questions))


def build_question_tool() -> ToolSpec:
    """构建向用户收集选择的工具。"""
    return ToolSpec(
        name="question",
        description=(
            "Ask the user one or more questions during execution. Options are "
            "optional; without options the user can enter free text."
        ),
        input_hint='JSON: {"questions":[{"question":"Proceed?","options":[{"label":"Yes"}]}]}',
        handler=_QuestionToolHandler(),
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


def set_question_prompt_handler(
    tool: ToolSpec, handler: QuestionPromptHandler | None
) -> bool:
    """为 question 工具设置前端专用交互处理器。"""
    if not isinstance(tool.handler, _QuestionToolHandler):
        return False
    tool.handler.prompt_handler = handler
    return True


def _ask_with_questionary(
    questions: list[dict[str, Any]],
) -> list[list[str]]:
    """使用独立 CLI prompt 收集回答。"""
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
                    [display_to_label[str(selected)]] if selected is not None else []
                )
        else:
            selected = questionary.text(message, qmark=header or "?").ask()
            answers.append([str(selected)] if selected else [])
    return answers


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
