#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import sys
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer
from prompt_toolkit.completion import Completion
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SERVER_URL = os.environ.get(
    "LLAMA_SERVER_URL",
    "http://127.0.0.1:8080/v1/chat/completions",
)

MAX_TOKENS = int(os.environ.get("LLAMA_MAX_TOKENS", "2048"))
TEMPERATURE = float(os.environ.get("LLAMA_TEMPERATURE", "0.7"))
HTTP_TIMEOUT = float(os.environ.get("LLAMA_HTTP_TIMEOUT", "2"))

SESSION_DIR = Path.cwd()
DEFAULT_SESSION = SESSION_DIR / "default.json"
current_session = DEFAULT_SESSION

messages: list[dict[str, str]] = []
pending_files: list[dict[str, str]] = []
compact_start: int | None = None
context_window_tokens: int | None = None
last_usage: dict[str, Any] | None = None
last_timings: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Terminal styling
# ---------------------------------------------------------------------------

USE_COLOR = os.environ.get("NO_COLOR") is None and os.environ.get("TERM") != "dumb"

STYLE = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "gray": "\033[90m",
}


def color(text: Any, *styles: str) -> str:
    value = str(text)

    if not USE_COLOR:
        return value

    prefix = "".join(STYLE[name] for name in styles if name in STYLE)

    if not prefix:
        return value

    return f"{prefix}{value}{STYLE['reset']}"


def color_percent(percent: float) -> str:
    if percent >= 95:
        style = "red"
    elif percent >= 80:
        style = "yellow"
    else:
        style = "green"

    return color(f"{percent:.1f}%", style, "bold")


def role_color(role: str) -> str:
    if role == "user":
        return "cyan"

    if role == "assistant":
        return "green"

    if role == "system":
        return "magenta"

    return "white"


def label_value(label: str, value: Any, value_style: str = "white") -> str:
    return (
        f"{color(f'{label}:', 'gray')} "
        f"{color(value, value_style)}"
    )


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

COMMANDS: list[tuple[str, str, str]] = [
    ("/read", "FILE", "Attach one UTF-8 text file to the next message."),
    ("/glob", "PATTERN", "Attach all files matching a glob pattern."),
    ("/attachments", "", "List files waiting to be sent."),
    ("/clearfiles", "", "Remove pending attachments."),
    ("/system", "TEXT", "Add or replace the system prompt."),
    ("/save", "[NAME]", "Save the current conversation."),
    ("/load", "[NAME]", "Load a saved conversation."),
    ("/sessions", "", "List saved conversations."),
    ("/new", "", "Start a new conversation."),
    ("/history", "", "Show the conversation text."),
    ("/stats", "", "Show conversation roles and approximate sizes."),
    ("/context", "", "Show current context window usage."),
    ("/compact", "N", "Send only message N and later in future requests."),
    ("/compact", "clear", "Send the full conversation again."),
    ("/pop", "", "Remove the latest user/assistant exchange."),
    ("/show-system", "", "Display the current system prompt."),
    ("/settings", "", "Display current connection and generation settings."),
    ("/help", "", "Show this help."),
    ("/exit", "", "Save the current session and exit."),
    ("/quit", "", "Save the current session and exit."),
]

key_bindings = KeyBindings()


class SlashCommandCompleter(Completer):
    def get_completions(self, document: Any, complete_event: Any) -> Any:
        text = document.text_before_cursor

        if not text.startswith("/") or "\n" in text:
            return

        if " " in text:
            if text.endswith(" "):
                command = text.strip()
                partial = ""
            else:
                parts = text.rsplit(None, 1)

                if len(parts) != 2:
                    return

                command, partial = parts

            if command != "/compact":
                return

            if "clear".startswith(partial):
                yield Completion(
                    "clear",
                    start_position=-len(partial),
                    display_meta="Send the full conversation again.",
                )

            return

        partial = text
        seen: set[str] = set()

        for command, args, description in COMMANDS:
            if command in seen or not command.startswith(partial):
                continue

            seen.add(command)
            display = f"{command} {args}".rstrip()

            yield Completion(
                command,
                start_position=-len(partial),
                display=display,
                display_meta=description,
            )


def command_completion_matches(text: str) -> list[str]:
    if not text.startswith("/") or "\n" in text:
        return []

    if " " in text:
        if text.endswith(" "):
            command = text.strip()
            partial = ""
        else:
            parts = text.rsplit(None, 1)

            if len(parts) != 2:
                return []

            command, partial = parts

        if command == "/compact" and "clear".startswith(partial):
            return ["clear"]

        return []

    matches: list[str] = []
    seen: set[str] = set()

    for command, _, _ in COMMANDS:
        if command in seen or not command.startswith(text):
            continue

        seen.add(command)
        matches.append(command)

    return matches


def command_expects_argument(command: str) -> bool:
    return any(
        row_command == command and args
        for row_command, args, _ in COMMANDS
    )


def complete_single_command_match(buffer: Any, text: str) -> bool:
    matches = command_completion_matches(text)

    if len(matches) != 1:
        return False

    match = matches[0]

    if " " in text:
        partial = "" if text.endswith(" ") else text.rsplit(None, 1)[1]
        suffix = match[len(partial):]

        if suffix:
            buffer.insert_text(suffix)
            return True

        return False

    suffix = match[len(text):]

    if suffix:
        buffer.insert_text(suffix)
        return True

    if command_expects_argument(match):
        buffer.insert_text(" ")
        return True

    return False


def print_command_completion_help(text: str) -> None:
    prefix = text.strip()

    if not prefix.startswith("/"):
        return

    if " " in prefix:
        command = prefix.split()[0]
        rows = [
            row
            for row in COMMANDS
            if row[0] == command
        ]
    else:
        rows = [
            row
            for row in COMMANDS
            if row[0].startswith(prefix)
        ]

    if not rows:
        rows = COMMANDS

    print()
    print(color("Command help:", "bold", "cyan"))

    for command, args, description in rows:
        usage = f"{command} {args}".rstrip()
        print(f"  {color(f'{usage:18}', 'cyan')} {color(description, 'gray')}")


@key_bindings.add("enter")
def submit_prompt(event: Any) -> None:
    """Send the current prompt."""
    event.current_buffer.validate_and_handle()


@key_bindings.add("escape", "enter")
def insert_alt_enter_newline(event: Any) -> None:
    """Alt+Enter, or Esc followed by Enter, inserts a newline."""
    event.current_buffer.insert_text("\n")


@key_bindings.add("c-j")
def insert_ctrl_j_newline(event: Any) -> None:
    """Ctrl+J inserts a newline."""
    event.current_buffer.insert_text("\n")


@key_bindings.add("tab")
def complete_or_show_command_help(event: Any) -> None:
    """Complete slash commands, then show help on a second Tab."""
    buffer = event.current_buffer
    text = buffer.document.text_before_cursor

    if not text.startswith("/"):
        buffer.insert_text("\t")
        return

    if buffer.complete_state is not None:
        print_command_completion_help(text)
        buffer.cancel_completion()
        event.app.invalidate()
        return

    if complete_single_command_match(buffer, text):
        return

    buffer.start_completion(select_first=False)


# Shift+Enter is only distinguishable in terminals that emit a separate key
# sequence for it. prompt_toolkit versions and terminal protocols vary.
try:

    @key_bindings.add("s-enter")
    def insert_shift_enter_newline(event: Any) -> None:
        event.current_buffer.insert_text("\n")

except ValueError:
    pass


prompt_session: PromptSession[str] = PromptSession(
    multiline=True,
    completer=SlashCommandCompleter(),
    complete_while_typing=False,
    key_bindings=key_bindings,
    enable_history_search=True,
)


# ---------------------------------------------------------------------------
# Session handling
# ---------------------------------------------------------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Small terminal chat client for llama-server.",
    )
    parser.add_argument(
        "--server",
        help=(
            "llama-server URL. A base URL such as http://127.0.0.1:8080 "
            "is expanded to /v1/chat/completions."
        ),
    )
    parser.add_argument(
        "session",
        nargs="?",
        help=(
            "Session name or file. Names are stored in the current "
            "working directory; paths such as ./chat.json are used directly."
        ),
    )
    return parser.parse_args(argv)


def chat_completion_url(value: str) -> str:
    url = value.rstrip("/")
    parsed = urllib.parse.urlsplit(url)

    if not parsed.scheme or not parsed.netloc:
        raise ValueError("server URL must include scheme and host")

    if parsed.path.endswith("/v1/chat/completions"):
        return url

    if parsed.path in {"", "/"}:
        return urllib.parse.urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                "/v1/chat/completions",
                parsed.query,
                parsed.fragment,
            )
        )

    return url


def session_path(name: str | None) -> Path:
    if not name:
        return DEFAULT_SESSION

    safe_name = Path(name).name

    if not safe_name.endswith(".json"):
        safe_name += ".json"

    return SESSION_DIR / safe_name


def command_line_session_path(value: str | None) -> Path:
    if not value:
        return DEFAULT_SESSION

    path = Path(value).expanduser()
    is_path_like = (
        path.is_absolute()
        or os.sep in value
        or (os.altsep is not None and os.altsep in value)
        or value.startswith("~")
    )

    if is_path_like:
        if not path.name.endswith(".json"):
            path = path.with_name(path.name + ".json")

        return path

    return session_path(value)


def save_session(path: Path | None = None) -> None:
    if path is None:
        path = current_session

    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "messages": messages,
        "pending_files": pending_files,
        "compact_start": compact_start,
        "server_url": SERVER_URL,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }

    temporary_path = path.with_suffix(path.suffix + ".tmp")

    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    temporary_path.replace(path)


def load_session(path: Path | None = None) -> None:
    global messages, pending_files, compact_start

    if path is None:
        path = current_session

    payload = json.loads(path.read_text(encoding="utf-8"))

    loaded_messages = payload.get("messages", [])
    loaded_files = payload.get("pending_files", [])
    loaded_compact_start = payload.get("compact_start")

    if not isinstance(loaded_messages, list):
        raise ValueError("Invalid messages field in session file")

    if not isinstance(loaded_files, list):
        raise ValueError("Invalid pending_files field in session file")

    messages = loaded_messages
    pending_files = loaded_files
    compact_start = None

    if isinstance(loaded_compact_start, int):
        compact_start = loaded_compact_start
        normalize_compact_start()


def normalize_compact_start() -> None:
    global compact_start

    if compact_start is None:
        return

    if compact_start < 0 or compact_start >= len(messages):
        compact_start = None


def active_messages() -> list[dict[str, str]]:
    normalize_compact_start()

    if compact_start is None:
        return list(messages)

    system_messages = [
        message
        for message in messages[:compact_start]
        if message.get("role") == "system"
    ]

    return system_messages + messages[compact_start:]


def active_range_text() -> str:
    normalize_compact_start()

    if not messages:
        return "empty"

    if compact_start is None:
        return "all"

    start = compact_start + 1
    end = len(messages)
    system_before = any(
        message.get("role") == "system"
        for message in messages[:compact_start]
    )
    text = f"#{start}-#{end}"

    if system_before:
        text += " + system"

    return text


# ---------------------------------------------------------------------------
# File attachments
# ---------------------------------------------------------------------------

def add_file(filename: str) -> None:
    path = Path(filename).expanduser().resolve()

    if not path.exists():
        print(color(f"File does not exist: {path}", "yellow"))
        return

    if not path.is_file():
        print(color(f"Not a regular file: {path}", "yellow"))
        return

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        print(color(f"Not a UTF-8 text file: {path}", "yellow"))
        return
    except OSError as exc:
        print(color(f"Could not read {path}: {exc}", "yellow"))
        return

    pending_files.append(
        {
            "path": str(path),
            "content": content,
        }
    )

    print(
        color("Attached for next message: ", "green")
        + color(path, "white")
        + " "
        + color(f"({len(content):,} characters)", "gray")
    )


def build_user_message(text: str) -> str:
    if not pending_files:
        return text

    sections: list[str] = []

    for item in pending_files:
        path_json = json.dumps(item["path"], ensure_ascii=False)

        sections.append(
            f"<document path={path_json}>\n"
            f"{item['content']}\n"
            f"</document>"
        )

    sections.append(
        f"<request>\n"
        f"{text}\n"
        f"</request>"
    )

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# HTTP streaming
# ---------------------------------------------------------------------------

def endpoint_url(path: str) -> str:
    base_url = os.environ.get("LLAMA_SERVER_BASE_URL")

    if base_url:
        return base_url.rstrip("/") + path

    parsed = urllib.parse.urlsplit(SERVER_URL)
    base_path = parsed.path.rstrip("/")
    chat_path = "/v1/chat/completions"

    if base_path.endswith(chat_path):
        base_path = base_path[: -len(chat_path)]
    else:
        base_path = ""

    return urllib.parse.urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            base_path.rstrip("/") + path,
            "",
            "",
        )
    )


def request_json(
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: float = HTTP_TIMEOUT,
) -> dict[str, Any]:
    data = None
    method = "GET"

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        method = "POST"

    request = urllib.request.Request(
        endpoint_url(path),
        data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method=method,
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")

    loaded = json.loads(body)

    if not isinstance(loaded, dict):
        raise ValueError("Expected JSON object")

    return loaded


def get_context_window_tokens() -> int | None:
    global context_window_tokens

    if context_window_tokens is not None:
        return context_window_tokens

    try:
        props = request_json("/props")
        settings = props.get("default_generation_settings", {})
        n_ctx = settings.get("n_ctx")

        if isinstance(n_ctx, int) and n_ctx > 0:
            context_window_tokens = n_ctx

    except (
        OSError,
        urllib.error.URLError,
        urllib.error.HTTPError,
        ValueError,
        json.JSONDecodeError,
    ):
        return None

    return context_window_tokens


def estimate_tokens_from_chars(items: list[dict[str, str]]) -> int:
    chars = sum(len(item.get("content", "")) for item in items)
    return max(1, round(chars / 4)) if chars else 0


def has_user_message(items: list[dict[str, str]]) -> bool:
    return any(item.get("role") == "user" for item in items)


def count_input_tokens(items: list[dict[str, str]]) -> tuple[int | None, str]:
    if not items:
        return 0, "server"

    if not has_user_message(items):
        return estimate_tokens_from_chars(items), "estimate"

    payload = {
        "messages": items,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }

    try:
        data = request_json("/v1/chat/completions/input_tokens", payload)
        input_tokens = data.get("input_tokens")

        if isinstance(input_tokens, int):
            return input_tokens, "server"

    except (
        OSError,
        urllib.error.URLError,
        urllib.error.HTTPError,
        ValueError,
        json.JSONDecodeError,
    ):
        pass

    try:
        data = request_json("/apply-template", {"messages": items})
        prompt = data.get("prompt")

        if isinstance(prompt, str):
            tokens = request_json(
                "/tokenize",
                {
                    "content": prompt,
                    "add_special": False,
                    "parse_special": True,
                },
            ).get("tokens")

            if isinstance(tokens, list):
                return len(tokens), "server"

    except (
        OSError,
        urllib.error.URLError,
        urllib.error.HTTPError,
        ValueError,
        json.JSONDecodeError,
    ):
        pass

    return estimate_tokens_from_chars(items), "estimate"


def collect_context_stats() -> dict[str, Any]:
    active = active_messages()
    input_tokens, source = count_input_tokens(active)
    n_ctx = get_context_window_tokens()

    stats: dict[str, Any] = {
        "input_tokens": input_tokens,
        "n_ctx": n_ctx,
        "source": source,
        "active_messages": len(active),
    }

    if input_tokens is not None and n_ctx:
        stats["percent"] = input_tokens * 100 / n_ctx
        stats["remaining"] = n_ctx - input_tokens

    return stats


def format_number(value: Any) -> str:
    if isinstance(value, int):
        return f"{value:,}"

    return "?"


def format_context_status(stats: dict[str, Any], label: str = "Context") -> str:
    input_tokens = stats.get("input_tokens")
    n_ctx = stats.get("n_ctx")
    source = stats.get("source")
    label_text = color(label, "bold", "blue")

    if isinstance(input_tokens, int) and isinstance(n_ctx, int) and n_ctx > 0:
        percent = input_tokens * 100 / n_ctx
        remaining = n_ctx - input_tokens
        room_style = "green"

        if percent >= 95:
            room_style = "red"
        elif percent >= 80:
            room_style = "yellow"

        room = color(
            f"{remaining:,} left"
            if remaining >= 0
            else f"{abs(remaining):,} over",
            room_style,
        )
        text = (
            f"{label_text}: "
            f"{color(f'{input_tokens:,}', 'bold', 'white')}/"
            f"{color(f'{n_ctx:,}', 'white')} "
            f"{color('tokens', 'gray')} "
            f"({color_percent(percent)}, {room}, "
            f"{color('max out', 'gray')} {MAX_TOKENS:,})"
        )
    elif isinstance(input_tokens, int):
        text = (
            f"{label_text}: "
            f"{color(f'{input_tokens:,}', 'bold', 'white')} "
            f"{color('tokens', 'gray')}"
        )
    else:
        text = f"{label_text}: {color('unavailable', 'yellow')}"

    if source == "estimate":
        text += f" {color('approx', 'yellow')}"

    text += (
        f" {color('|', 'gray')} {color('next', 'gray')}: "
        f"{color(f'#{len(messages) + 1}', 'cyan', 'bold')}"
        f" {color('|', 'gray')} {color('active', 'gray')}: "
        f"{color(active_range_text(), 'cyan')}"
        f" {color('|', 'gray')} {color('messages', 'gray')}: "
        f"{color(len(messages), 'white')}"
    )

    if pending_files:
        pending_chars = sum(len(item["content"]) for item in pending_files)
        text += (
            f" {color('|', 'gray')} {color('pending files', 'gray')}: "
            f"{color(len(pending_files), 'yellow')} "
            f"{color(f'({pending_chars:,} chars)', 'yellow')}"
        )

    if last_usage:
        prompt_tokens = last_usage.get("prompt_tokens")
        completion_tokens = last_usage.get("completion_tokens")
        prompt_details = last_usage.get("prompt_tokens_details", {})
        cached_tokens = None

        if isinstance(prompt_details, dict):
            cached_tokens = prompt_details.get("cached_tokens")

        text += (
            f" {color('|', 'gray')} {color('last', 'gray')}: "
            f"{color('in', 'gray')} {format_number(prompt_tokens)}, "
            f"{color('out', 'gray')} {format_number(completion_tokens)}"
        )

        if isinstance(cached_tokens, int) and cached_tokens > 0:
            text += f", {color('cached', 'gray')} {cached_tokens:,}"

    if last_timings:
        predicted_per_second = last_timings.get("predicted_per_second")

        if isinstance(predicted_per_second, (int, float)):
            text += (
                f" {color('|', 'gray')} "
                f"{color(f'{predicted_per_second:.1f}', 'green')} "
                f"{color('tok/s', 'gray')}"
            )

    return text


def format_prompt(stats: dict[str, Any]) -> str:
    percent = stats.get("percent")
    message_number = len(messages) + 1

    if isinstance(percent, (int, float)):
        if percent >= 95:
            percent_style = "red"
        elif percent >= 80:
            percent_style = "yellow"
        else:
            percent_style = "green"

        return (
            f"{color('you', 'cyan', 'bold')} "
            f"{color(f'#{message_number}', 'gray')} "
            f"{color(f'{percent:.0f}%', percent_style, 'bold')}"
            f"{color('>', 'gray')} "
        )

    return (
        f"{color('you', 'cyan', 'bold')} "
        f"{color(f'#{message_number}', 'gray')}"
        f"{color('>', 'gray')} "
    )


def stream_response() -> tuple[str, bool, dict[str, Any] | None, dict[str, Any] | None]:
    request_body = {
        "messages": active_messages(),
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
    }

    request = urllib.request.Request(
        SERVER_URL,
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )

    output: list[str] = []
    interrupted = False
    usage: dict[str, Any] | None = None
    timings: dict[str, Any] | None = None

    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            try:
                for raw_line in response:
                    line = raw_line.decode(
                        "utf-8",
                        errors="replace",
                    ).strip()

                    if not line.startswith("data:"):
                        continue

                    data = line[5:].strip()

                    if not data:
                        continue

                    if data == "[DONE]":
                        break

                    try:
                        event = json.loads(data)

                        if isinstance(event.get("usage"), dict):
                            usage = event["usage"]

                        if isinstance(event.get("timings"), dict):
                            timings = event["timings"]

                        choices = event.get("choices", [])
                        if not choices:
                            continue

                        choice = choices[0]
                        delta = choice.get("delta", {})
                        text = delta.get("content") or ""

                    except (
                        json.JSONDecodeError,
                        KeyError,
                        IndexError,
                        TypeError,
                    ):
                        continue

                    if text:
                        print(text, end="", flush=True)
                        output.append(text)

            except KeyboardInterrupt:
                interrupted = True
                print(
                    "\n\n"
                    "[Generation interrupted - conversation remains active]",
                    flush=True,
                )

    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP {exc.code}: {body}"
        ) from exc

    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach llama-server: {exc}"
        ) from exc

    return "".join(output), interrupted, usage, timings


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def show_help() -> None:
    print(
        """
Commands:

  /read FILE
      Attach one UTF-8 text file to the next message.

  /glob PATTERN
      Attach all files matching a glob pattern.
      Quote the pattern to prevent your shell from expanding it.

  /attachments
      List files waiting to be sent with the next message.

  /clearfiles
      Remove pending attachments.

  /system TEXT
      Add or replace the system prompt.

  /save [NAME]
      Save the current conversation.

  /load [NAME]
      Load a saved conversation.

  /sessions
      List saved conversations.

  /new
      Start a new conversation.

  /history
      Show the conversation text.

  /stats
      Show conversation roles and approximate sizes.

  /context
      Show current context window usage.

  /compact N
      Send only message N and later in future requests.
      System prompts before N are still included.

  /compact clear
      Send the full conversation again.

  /pop
      Remove the latest user/assistant exchange.

  /show-system
      Display the current system prompt.

  /settings
      Display current connection and generation settings.

  /help
      Show this help.

  /exit
      Save the current session and exit.


Keyboard:

  Enter
      Send the prompt.

  Shift+Enter
      Insert a newline, when supported by the terminal.

  Alt+Enter
      Insert a newline.

  Ctrl+J
      Insert a newline.

  Tab
      Complete slash commands.

  Tab, Tab
      Show help for matching slash commands.

  Ctrl+C while typing
      Clear/cancel the current input.

  Ctrl+C while generating
      Stop only the current generation and keep the program running.

  Ctrl+D
      Save and exit.


Environment variables:

  LLAMA_SERVER_URL
      Default:
      http://127.0.0.1:8080/v1/chat/completions
      Can be overridden with --server.

  LLAMA_SERVER_BASE_URL
      Optional base URL for /props, /tokenize and token counting.
      Default: derived from LLAMA_SERVER_URL

  LLAMA_MAX_TOKENS
      Default: 2048

  LLAMA_TEMPERATURE
      Default: 0.7

  LLAMA_HTTP_TIMEOUT
      Timeout in seconds for lightweight stats requests.
      Default: 2
"""
    )


def show_sessions() -> None:
    SESSION_DIR.mkdir(parents=True, exist_ok=True)

    session_files = sorted(
        SESSION_DIR.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not session_files:
        print("No saved sessions.")
        return

    for path in session_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            count = len(payload.get("messages", []))
        except (OSError, ValueError, json.JSONDecodeError):
            count = -1

        if count >= 0:
            print(f"{path.stem:30} {count:5} messages")
        else:
            print(f"{path.stem:30} unreadable")


def parse_attached_user_message(
    content: str,
) -> tuple[list[tuple[str, int]], str] | None:
    attachments: list[tuple[str, int]] = []
    position = 0
    document_prefix = "<document path="

    while content.startswith(document_prefix, position):
        header_end = content.find(">\n", position)

        if header_end < 0:
            return None

        path_value = content[
            position + len(document_prefix):header_end
        ]

        try:
            path = json.loads(path_value)
        except json.JSONDecodeError:
            return None

        if not isinstance(path, str):
            return None

        document_start = header_end + 2
        document_end = content.find("\n</document>", document_start)

        if document_end < 0:
            return None

        document_content = content[document_start:document_end]
        attachments.append((path, len(document_content)))
        position = document_end + len("\n</document>")

        if content.startswith("\n\n", position):
            position += 2

    request_prefix = "<request>\n"

    if not attachments or not content.startswith(request_prefix, position):
        return None

    request_start = position + len(request_prefix)
    request_suffix = "\n</request>"

    if not content.endswith(request_suffix):
        return None

    return attachments, content[request_start:-len(request_suffix)]


def print_history_message(number: int, message: dict[str, str]) -> None:
    role = message.get("role", "?")
    content = message.get("content", "")
    labels = {
        "assistant": "assistant",
        "system": "system",
        "user": "you",
    }
    label = labels.get(role, role)
    header_style = role_color(role)

    print()
    print(
        color("--- ", "gray")
        + color(f"{number}: {label}", "bold", header_style)
        + color(f" ({len(content):,} chars) ", "gray")
        + color("---", "gray")
    )

    parsed = (
        parse_attached_user_message(content)
        if role == "user"
        else None
    )

    if parsed:
        attachments, request = parsed
        print(color("attachments:", "yellow"))

        for path, chars in attachments:
            print(
                f"  {color(path, 'white')} "
                f"{color(f'({chars:,} chars)', 'gray')}"
            )

        print()
        print(color("prompt:", "cyan"))
        content = request

    if content:
        print(textwrap.indent(content, color("  ", "gray")))
    else:
        print(color("  [empty]", "gray"))


def show_history() -> None:
    if not messages:
        print("Conversation is empty.")
        return

    print(
        f"{color('Active context', 'bold', 'blue')}: "
        f"{color(active_range_text(), 'cyan')}"
    )

    for number, message in enumerate(messages, start=1):
        if compact_start == number - 1:
            print()
            print(color("=== active context starts here ===", "yellow", "bold"))

        print_history_message(number, message)


def show_stats() -> None:
    if not messages:
        print("Conversation is empty.")
        return

    print(
        f"{color('Active context', 'bold', 'blue')}: "
        f"{color(active_range_text(), 'cyan')}"
    )

    for number, message in enumerate(messages, start=1):
        role = message.get("role", "?")
        content = message.get("content", "")
        lines = content.count("\n") + 1
        active = (
            "*"
            if (
                compact_start is None
                or number - 1 >= compact_start
                or role == "system"
            )
            else " "
        )

        print(
            f"{color(active, 'green')}"
            f"{color(f'{number:3}', role_color(role))}: "
            f"{color(f'{role:10}', role_color(role))} "
            f"{len(content):9,} {color('chars', 'gray')} "
            f"{lines:6,} {color('lines', 'gray')}"
        )

    if compact_start is not None:
        print(color("* included in active context", "green"))


def remove_last_exchange() -> None:
    global messages

    if not messages:
        print("Conversation is empty.")
        return

    removed = 0

    while messages and removed < 2:
        if messages[-1].get("role") == "system":
            break

        messages.pop()
        removed += 1

    normalize_compact_start()
    print(color(f"Removed {removed} message(s).", "green"))


def handle_compact(args: list[str]) -> None:
    global compact_start, last_usage, last_timings

    if not args:
        print(
            f"{color('Active context', 'bold', 'blue')}: "
            f"{color(active_range_text(), 'cyan')}"
        )
        print(
            color("Usage:", "gray")
            + " "
            + color("/compact N", "cyan")
            + " or "
            + color("/compact clear", "cyan")
        )
        return

    value = args[0].lower()

    if value in {"clear", "off", "none", "reset"}:
        compact_start = None
        last_usage = None
        last_timings = None
        save_session()
        print(
            color(
                "Compaction cleared. Future requests send the full conversation.",
                "green",
            )
        )
        return

    try:
        number = int(value)
    except ValueError:
        print(
            color("Usage:", "gray")
            + " "
            + color("/compact N", "cyan")
            + " or "
            + color("/compact clear", "cyan")
        )
        return

    if number < 1 or number > len(messages):
        print(
            color(
                f"Message number must be between 1 and {len(messages)}.",
                "yellow",
            )
        )
        return

    compact_start = number - 1
    last_usage = None
    last_timings = None
    save_session()
    print(
        color(f"Compacted at message #{number}.", "green")
        + " "
        + color("Future requests use active context:", "gray")
        + " "
        + color(active_range_text(), "cyan")
        + "."
    )


def handle_command(line: str) -> None:
    global current_session, messages, pending_files
    global compact_start, last_usage, last_timings

    try:
        parts = shlex.split(line)
    except ValueError as exc:
        print(color(f"Invalid command: {exc}", "yellow"))
        return

    if not parts:
        return

    command = parts[0].lower()
    args = parts[1:]

    if command in {"/exit", "/quit"}:
        save_session()
        raise SystemExit(0)

    if command == "/help":
        show_help()
        return

    if command == "/read":
        if not args:
            print(color("Usage:", "gray") + " " + color("/read FILE", "cyan"))
            return

        add_file(args[0])
        return

    if command == "/glob":
        if not args:
            print(
                color("Usage:", "gray")
                + " "
                + color("/glob PATTERN", "cyan")
            )
            return

        pattern = os.path.expanduser(args[0])
        matches = sorted(
            glob.glob(pattern, recursive=True)
        )

        regular_files = [
            filename
            for filename in matches
            if Path(filename).is_file()
        ]

        if not regular_files:
            print(color("No files matched.", "yellow"))
            return

        for filename in regular_files:
            add_file(filename)

        return

    if command == "/attachments":
        if not pending_files:
            print(color("No pending attachments.", "gray"))
            return

        for item in pending_files:
            print(
                color(item["path"], "white")
                + " "
                + color(f"({len(item['content']):,} characters)", "gray")
            )

        return

    if command == "/clearfiles":
        pending_files = []
        print(color("Pending attachments cleared.", "green"))
        return

    if command == "/system":
        if not args:
            print(
                color("Usage:", "gray")
                + " "
                + color("/system TEXT", "cyan")
            )
            return

        text = " ".join(args)
        removed_before_compact = 0

        compact_was_at_start = compact_start == 0

        if compact_start is not None:
            removed_before_compact = sum(
                1
                for message in messages[:compact_start]
                if message.get("role") == "system"
            )

        messages = [
            message
            for message in messages
            if message.get("role") != "system"
        ]

        if compact_start is not None:
            compact_start -= removed_before_compact
            compact_start = max(compact_start, 0)

        messages.insert(
            0,
            {
                "role": "system",
                "content": text,
            },
        )

        if compact_start is not None and not compact_was_at_start:
            compact_start += 1

        normalize_compact_start()
        save_session()
        print(color("System prompt updated.", "green"))
        return

    if command == "/show-system":
        system_messages = [
            message["content"]
            for message in messages
            if message.get("role") == "system"
        ]

        if not system_messages:
            print(color("No system prompt is set.", "gray"))
        else:
            print(system_messages[0])

        return

    if command == "/save":
        path = session_path(args[0]) if args else current_session
        save_session(path)
        print(color("Saved: ", "green") + color(path, "cyan"))
        return

    if command == "/load":
        path = session_path(args[0]) if args else current_session

        if not path.exists():
            print(color(f"No such session: {path}", "yellow"))
            return

        try:
            load_session(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(color(f"Could not load session: {exc}", "yellow"))
            return

        current_session = path
        print(
            color("Loaded: ", "green")
            + color(path, "cyan")
            + " "
            + color(f"({len(messages)} messages)", "gray")
        )
        last_usage = None
        last_timings = None
        return

    if command == "/sessions":
        show_sessions()
        return

    if command == "/new":
        messages = []
        pending_files = []
        compact_start = None
        last_usage = None
        last_timings = None
        save_session()
        print(color("Started a new conversation.", "green"))
        return

    if command == "/context":
        stats = collect_context_stats()
        print(format_context_status(stats))
        return

    if command == "/history":
        show_history()
        return

    if command == "/stats":
        show_stats()
        return

    if command == "/compact":
        handle_compact(args)
        return

    if command == "/pop":
        remove_last_exchange()
        save_session()
        return

    if command == "/settings":
        print(label_value("Server URL", SERVER_URL, "white"))
        print(label_value("Base URL", endpoint_url(""), "white"))
        print(label_value("Max tokens", f"{MAX_TOKENS:,}", "white"))
        print(label_value("Temperature", TEMPERATURE, "white"))
        print(label_value("Session file", current_session, "cyan"))
        print(label_value("Messages", len(messages), "white"))
        print(label_value("Next message", f"#{len(messages) + 1}", "cyan"))
        print(label_value("Active", active_range_text(), "cyan"))
        print(label_value("Attachments", len(pending_files), "white"))
        return

    print(color(f"Unknown command: {command}", "yellow"))
    print(
        color("Use ", "gray")
        + color("/help", "cyan")
        + color(" to show available commands.", "gray")
    )


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    global SERVER_URL, context_window_tokens, current_session
    global pending_files, last_usage, last_timings

    args = parse_args(sys.argv[1:] if argv is None else argv)

    if args.server:
        try:
            SERVER_URL = chat_completion_url(args.server)
            context_window_tokens = None
        except ValueError as exc:
            raise SystemExit(f"Invalid --server: {exc}") from exc

    current_session = command_line_session_path(args.session)

    SESSION_DIR.mkdir(parents=True, exist_ok=True)

    print(label_value("Session file", current_session, "cyan"))

    if current_session.exists():
        try:
            load_session()
            print(
                color("Loaded previous session with ", "green")
                + color(len(messages), "white")
                + color(" messages.", "green")
            )

        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(color(f"Could not load previous session: {exc}", "yellow"))
    else:
        print(color("Starting a new session.", "green"))

    print(label_value("Server", SERVER_URL, "white"))
    print(label_value("Maximum output", f"{MAX_TOKENS:,} tokens", "white"))
    print(label_value("Temperature", TEMPERATURE, "white"))
    print(
        color("Enter", "cyan")
        + color(" sends. ", "gray")
        + color("Shift+Enter", "cyan")
        + color(", ", "gray")
        + color("Alt+Enter", "cyan")
        + color(" or ", "gray")
        + color("Ctrl+J", "cyan")
        + color(" inserts a newline.", "gray")
    )
    print(
        color("Ctrl+C", "cyan")
        + color(" interrupts generation without exiting. Use ", "gray")
        + color("/help", "cyan")
        + color(" for commands.", "gray")
    )

    while True:
        try:
            stats = collect_context_stats()
            print(f"\n{format_context_status(stats)}")

            with patch_stdout():
                line = prompt_session.prompt(
                    ANSI(format_prompt(stats)),
                    prompt_continuation="...  ",
                ).strip()

        except EOFError:
            print()
            save_session()
            return

        except KeyboardInterrupt:
            print()
            continue

        if not line:
            continue

        if line.startswith("/"):
            handle_command(line)
            continue

        user_content = build_user_message(line)
        sent_files = pending_files
        pending_files = []

        messages.append(
            {
                "role": "user",
                "content": user_content,
            }
        )

        stats = collect_context_stats()
        print(f"\n{format_context_status(stats, 'Sending')}")
        print(
            "\n"
            + color("assistant", "green", "bold")
            + color(">", "gray")
            + " ",
            end="",
            flush=True,
        )

        try:
            answer, interrupted, usage, timings = stream_response()

        except RuntimeError as exc:
            print(f"\n{color(f'Error: {exc}', 'red')}")

            # Remove the unanswered user message.
            if (
                messages
                and messages[-1].get("role") == "user"
            ):
                messages.pop()

            pending_files = sent_files + pending_files
            save_session()
            continue

        if usage:
            last_usage = usage

        if timings:
            last_timings = timings

        if answer:
            messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )
        else:
            # No output was generated, so remove the user message to avoid
            # leaving an invalid unanswered turn in the history.
            if (
                messages
                and messages[-1].get("role") == "user"
            ):
                messages.pop()

            pending_files = sent_files + pending_files

        save_session()

        if interrupted:
            if answer:
                print(
                    color("Partial response saved.", "yellow")
                    + color(
                        " You can ask the model to continue or change direction.",
                        "gray",
                    )
                )
            else:
                print(
                    color("No response content was generated.", "yellow")
                    + color(" The unanswered user message was removed.", "gray")
                )


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print("\n" + color("Saving session and exiting.", "green"))
        try:
            save_session()
        except OSError as exc:
            print(color(f"Could not save session: {exc}", "red"), file=sys.stderr)
        raise SystemExit(130)
