"""Reconstruct Claude Code hook events from gateway traffic.

Claude Code fans one user prompt into several model calls, and the hook events an
endpoint-installed hook would emit are spread across them: the prompt arrives on the
request of the first tool-bearing call, each tool call on the response that produced it,
and each tool result on the request of the next call. This module recovers those events
from what the guardrail already sees, so a proxy can score the same things the native
hooks do without a hook installed on the client.

Pure functions over already-decoded request data; no I/O, no proxy imports.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence, cast

from pydantic import BaseModel

from litellm.types.proxy.guardrails.guardrail_hooks.straiker import StraikerHookEvent

SCAFFOLD_PATTERN = re.compile(
    r"<(system-reminder|command-message|command-name|command-args"
    r"|local-command-stdout|local-command-caveat)>.*?</\1>",
    re.DOTALL,
)

SYSTEM_REMINDER_PREFIX = "<system-reminder>"

CLAUDE_CODE_CORE_TOOLS = frozenset({"Bash", "Read", "Edit", "TodoWrite"})

CLAUDE_CODE_USER_AGENT_PREFIX = "claude-cli/"

CLAUDE_CODE_SYSTEM_MARKERS = ("cc_version=", "cc_entrypoint=", "claude code")

CHATTER_MARKERS = (
    "[suggestion mode:",
    "suggest what the user might naturally type next",
    "the user stepped away and is coming back",
    "recap what you were doing in under",
    "write a 5-10 word title",
    "write the title in the predominant language",
    "you are an expert at summarizing conversations",
    "your task is to create a detailed summary of the conversation",
)

MCP_TOOL_PATTERN = re.compile(r"^mcp__(.+?)__(.+)$")

RequestKind = Literal["turn", "utility"]


@dataclass(frozen=True, slots=True)
class ToolResult:
    tool_use_id: str
    tool_name: str | None
    content: str
    is_error: bool


@dataclass(frozen=True, slots=True)
class ToolCall:
    tool_use_id: str
    tool_name: str
    tool_input: dict[str, object]


@dataclass(frozen=True, slots=True)
class ParsedRequest:
    kind: RequestKind
    user_prompt: str | None
    tool_results: tuple[ToolResult, ...]
    tool_count: int
    chatter_reason: str | None


def _as_mapping(value: object) -> Mapping[str, object]:
    """Guardrail inputs arrive either as plain dicts or as the pydantic models litellm
    assembles them into; which one depends on the API surface and whether the response
    streamed, so both have to read the same."""
    if isinstance(value, BaseModel):
        return cast("Mapping[str, object]", value.model_dump())
    return cast("Mapping[str, object]", value) if isinstance(value, Mapping) else {}


def _as_sequence(value: object) -> Sequence[object]:
    return cast("Sequence[object]", value) if isinstance(value, (list, tuple)) else ()


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def strip_scaffold(text: str) -> str:
    return SCAFFOLD_PATTERN.sub("", text)


def _prompt_from_string(text: str) -> str:
    residue = strip_scaffold(text)
    trailing = re.split(r"\nuser:", residue)
    return trailing[-1].strip()


def _text_blocks(content: object) -> tuple[str, ...]:
    return tuple(
        text
        for block in _as_sequence(content)
        if (mapping := _as_mapping(block)).get("type") == "text" and (text := _as_str(mapping.get("text"))) is not None
    )


def _system_text(system: object) -> str:
    if isinstance(system, str):
        return system
    return "\n".join(_text_blocks(system))


def _tool_names(request_data: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(
        name
        for tool in _as_sequence(request_data.get("tools"))
        for mapping in (_as_mapping(tool),)
        if (name := _as_str(mapping.get("name")) or _as_str(_as_mapping(mapping.get("function")).get("name")))
    )


def is_claude_code(request_data: Mapping[str, object], user_agent: str | None) -> bool:
    """Claude Code identifies itself three ways; any one is sufficient.

    The user agent is the only signal present on every call including the zero-tool
    utility calls. The billing-header markers in the first system block
    (``cc_version=`` / ``cc_entrypoint=``) are the current client's fingerprint;
    the literal string "claude code" no longer appears in its system prompt.
    """
    if user_agent and user_agent.startswith(CLAUDE_CODE_USER_AGENT_PREFIX):
        return True
    if not CLAUDE_CODE_CORE_TOOLS.isdisjoint(_tool_names(request_data)):
        return True
    system = _system_text(request_data.get("system")).lower()
    return any(marker in system for marker in CLAUDE_CODE_SYSTEM_MARKERS)


def _tool_names_by_id(messages: Sequence[object]) -> Mapping[str, str]:
    return {
        tool_use_id: name
        for message in messages
        for mapping in (_as_mapping(message),)
        if mapping.get("role") == "assistant"
        for tool_use_id, name in _assistant_tool_uses(mapping)
    }


def _tool_use_pair(block: object) -> tuple[str, str] | None:
    mapping = _as_mapping(block)
    if mapping.get("type") != "tool_use":
        return None
    tool_use_id = _as_str(mapping.get("id"))
    name = _as_str(mapping.get("name"))
    return (tool_use_id, name) if tool_use_id and name else None


def _tool_call_pair(call: object) -> tuple[str, str] | None:
    mapping = _as_mapping(call)
    tool_use_id = _as_str(mapping.get("id"))
    name = _as_str(_as_mapping(mapping.get("function")).get("name"))
    return (tool_use_id, name) if tool_use_id and name else None


def _assistant_tool_uses(message: Mapping[str, object]) -> tuple[tuple[str, str], ...]:
    pairs = (
        *(_tool_use_pair(block) for block in _as_sequence(message.get("content"))),
        *(_tool_call_pair(call) for call in _as_sequence(message.get("tool_calls"))),
    )
    return tuple(pair for pair in pairs if pair is not None)


def _tail_after_last_assistant(messages: Sequence[object]) -> Sequence[object]:
    last_assistant = max(
        (index for index, message in enumerate(messages) if _as_mapping(message).get("role") == "assistant"),
        default=-1,
    )
    return messages[last_assistant + 1 :]


def _block_text(content: object) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(_text_blocks(content))


def _tool_results(tail: Sequence[object], names: Mapping[str, str]) -> tuple[ToolResult, ...]:
    anthropic = tuple(
        ToolResult(
            tool_use_id=tool_use_id,
            tool_name=names.get(tool_use_id),
            content=_block_text(block_mapping.get("content")),
            is_error=bool(block_mapping.get("is_error")),
        )
        for message in tail
        for message_mapping in (_as_mapping(message),)
        if message_mapping.get("role") == "user"
        for block in _as_sequence(message_mapping.get("content"))
        for block_mapping in (_as_mapping(block),)
        if block_mapping.get("type") == "tool_result" and (tool_use_id := _as_str(block_mapping.get("tool_use_id")))
    )
    openai = tuple(
        ToolResult(
            tool_use_id=tool_use_id,
            tool_name=names.get(tool_use_id) or _as_str(message_mapping.get("name")),
            content=_block_text(message_mapping.get("content")),
            is_error=False,
        )
        for message in tail
        for message_mapping in (_as_mapping(message),)
        if message_mapping.get("role") == "tool" and (tool_use_id := _as_str(message_mapping.get("tool_call_id")))
    )
    return anthropic + openai


def _user_prompt(tail: Sequence[object]) -> str | None:
    user_contents = tuple(
        mapping.get("content")
        for message in tail
        for mapping in (_as_mapping(message),)
        if mapping.get("role") == "user"
    )
    if not user_contents:
        return None
    content = user_contents[-1]
    if isinstance(content, str):
        return _prompt_from_string(content) or None
    candidates = tuple(text for text in _text_blocks(content) if not text.lstrip().startswith(SYSTEM_REMINDER_PREFIX))
    return candidates[-1].strip() if candidates else None


def _chatter_reason(tool_count: int, prompt: str | None) -> str | None:
    if tool_count == 0:
        return "no_tools_utility"
    haystack = (prompt or "").lower()
    return next((f"marker:{marker}" for marker in CHATTER_MARKERS if marker in haystack), None)


def parse_request(request_data: Mapping[str, object], chatter_filter: bool = True) -> ParsedRequest:
    """Classify one model call and pull out the prompt and any tool results it carries.

    Claude Code resends the whole transcript every call, so the tool result the model is
    being handed back always sits after the last assistant message; the ``tool_use`` it
    answers is recoverable from the same transcript, which is what lets PostToolUse carry
    a tool name.
    """
    messages = _as_sequence(request_data.get("messages"))
    tail = _tail_after_last_assistant(messages)
    prompt = _user_prompt(tail)
    tool_count = len(_tool_names(request_data))
    reason = _chatter_reason(tool_count, prompt) if chatter_filter else None
    return ParsedRequest(
        kind="utility" if reason else "turn",
        user_prompt=prompt,
        tool_results=_tool_results(tail, _tool_names_by_id(messages)),
        tool_count=tool_count,
        chatter_reason=reason,
    )


def is_utility_call(request_data: Mapping[str, object], chatter_filter: bool = True) -> bool:
    """Whether this call is Claude Code scaffolding rather than user intent.

    The response side needs this too: a title-generation call produces a perfectly
    ordinary-looking answer, and scoring it is what makes gateways report detections on
    traffic the user never wrote.
    """
    if not chatter_filter:
        return False
    tail = _tail_after_last_assistant(_as_sequence(request_data.get("messages")))
    return _chatter_reason(len(_tool_names(request_data)), _user_prompt(tail)) is not None


def _tool_arguments(function: Mapping[str, object]) -> dict[str, object]:
    arguments = function.get("arguments")
    if isinstance(arguments, Mapping):
        return dict(_as_mapping(arguments))
    if not isinstance(arguments, str) or not arguments:
        return {}
    decoded = _decode_json(arguments)
    if decoded is None:
        return {"arguments": arguments}
    if isinstance(decoded, dict):
        return dict(_as_mapping(decoded))
    return {"arguments": decoded}


def _decode_json(text: str) -> object | None:
    try:
        return cast("object", json.loads(text))
    except json.JSONDecodeError:
        return None


def _tool_call(call: object) -> ToolCall | None:
    mapping = _as_mapping(call)
    function = _as_mapping(mapping.get("function"))
    tool_use_id = _as_str(mapping.get("id"))
    name = _as_str(function.get("name"))
    if not tool_use_id or not name:
        return None
    return ToolCall(tool_use_id=tool_use_id, tool_name=name, tool_input=_tool_arguments(function))


def parse_tool_calls(tool_calls: object) -> tuple[ToolCall, ...]:
    """Read the assembled tool calls off a response.

    LiteLLM assembles streamed tool calls before the guardrail runs, so both the
    Anthropic ``tool_use`` blocks and OpenAI ``tool_calls`` arrive here in the same
    OpenAI-shaped form regardless of surface or upstream provider.
    """
    parsed = (_tool_call(call) for call in _as_sequence(tool_calls))
    return tuple(call for call in parsed if call is not None)


def _mcp_fields(tool_name: str) -> tuple[str | None, str | None]:
    match = MCP_TOOL_PATTERN.match(tool_name)
    if match is None:
        return None, None
    return match.group(1), match.group(2)


def request_events(parsed: ParsedRequest, session_id: str, user_name: str | None) -> tuple[StraikerHookEvent, ...]:
    post_tool_use = tuple(
        StraikerHookEvent(
            hook_event_name="PostToolUse",
            session_id=session_id,
            user_name=user_name,
            tool_name=result.tool_name,
            tool_use_id=result.tool_use_id,
            tool_response=result.content,
            is_error=result.is_error,
        )
        for result in parsed.tool_results
    )
    if parsed.kind != "turn" or not parsed.user_prompt:
        return post_tool_use
    return post_tool_use + (
        StraikerHookEvent(
            hook_event_name="UserPromptSubmit",
            session_id=session_id,
            user_name=user_name,
            prompt=parsed.user_prompt,
        ),
    )


def response_events(
    tool_calls: tuple[ToolCall, ...],
    final_text: str | None,
    finish_reason: str | None,
    session_id: str,
    user_name: str | None,
) -> tuple[StraikerHookEvent, ...]:
    """PreToolUse for every tool the model wants to run, then Stop for a final answer.

    Stop has no native equivalent: the endpoint hook never sees the model's answer, but a
    gateway does, so it is emitted for telemetry and output-side scoring only.
    """
    pre_tool_use = tuple(
        StraikerHookEvent(
            hook_event_name="PreToolUse",
            session_id=session_id,
            user_name=user_name,
            tool_name=call.tool_name,
            tool_use_id=call.tool_use_id,
            tool_input=call.tool_input,
            mcp_server_name=mcp_server,
            mcp_tool_name=mcp_tool,
        )
        for call in tool_calls
        for mcp_server, mcp_tool in (_mcp_fields(call.tool_name),)
    )
    if tool_calls or not final_text:
        return pre_tool_use
    return pre_tool_use + (
        StraikerHookEvent(
            hook_event_name="Stop",
            session_id=session_id,
            user_name=user_name,
            app_response=final_text,
            stop_reason=finish_reason,
        ),
    )
