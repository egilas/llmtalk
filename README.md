# llmchat

`llmchat` is a small terminal chat client for OpenAI-compatible chat completion servers.

It keeps local conversation sessions, streams responses, supports text file attachments, and includes a few workflow helpers for long-running local model conversations.

## Features

- Streaming chat through an OpenAI-compatible `/v1/chat/completions` endpoint
- Local JSON session files
- Optional session selection from the command line
- Manual context compaction with `/compact`
- Context/token usage status when supported by the server
- File attachments with `/read` and `/glob`
- Slash commands with Tab completion
- Readable `/history` and compact `/stats`
- Colored terminal status output, with `NO_COLOR=1` support

## Requirements

- Python 3.10 or newer
- A running OpenAI-compatible chat completion server
- `prompt_toolkit`

Install Python dependencies:

```sh
python3 -m pip install -r requirements.txt
```

## Usage

Start with the default server and default session:

```sh
python3 llmchat.py
```

Use a named session:

```sh
python3 llmchat.py work
```

Named sessions are stored in:

```text
the current working directory
```

For example, `python3 llmchat.py work` uses `./work.json`, and `python3 llmchat.py` uses `./default.json`.

Use a specific session file path:

```sh
python3 llmchat.py ./work.json
```

Use a different server:

```sh
python3 llmchat.py --server http://127.0.0.1:8081
```

If `--server` is a base URL, `llmchat` expands it to:

```text
/v1/chat/completions
```

You can also pass the full endpoint:

```sh
python3 llmchat.py --server http://127.0.0.1:8081/v1/chat/completions work
```

## Configuration

Environment variables:

```text
LLAMA_SERVER_URL       Chat completions endpoint
LLAMA_SERVER_BASE_URL  Optional base URL for token stats endpoints
LLAMA_MAX_TOKENS       Maximum response tokens, default 2048
LLAMA_TEMPERATURE      Sampling temperature, default 0.7
LLAMA_HTTP_TIMEOUT     Timeout for lightweight stats requests, default 2
NO_COLOR               Disable terminal colors when set
```

`--server` overrides `LLAMA_SERVER_URL`.

## Commands

```text
/read FILE        Attach one UTF-8 text file to the next message
/glob PATTERN     Attach all files matching a glob pattern
/attachments      List files waiting to be sent
/clearfiles       Remove pending attachments
/system TEXT      Add or replace the system prompt
/save [NAME]      Save the current conversation
/load [NAME]      Load a saved conversation
/sessions         List saved conversations
/new              Start a new conversation
/history          Show the conversation text
/stats            Show message roles, character counts, and line counts
/context          Show current context window usage
/compact N        Send only message N and later in future requests
/compact clear    Send the full conversation again
/pop              Remove the latest user/assistant exchange
/show-system      Display the current system prompt
/settings         Display connection and generation settings
/help             Show command help
/exit             Save and exit
```

Press `Tab` after a slash command prefix to complete it. Press `Tab` again while the completion menu is open to print command help.

## Manual Compaction

Long sessions can grow beyond the model context window. `llmchat` never compacts automatically, but you can choose the active context yourself.

Example:

```text
/compact 39
```

The full conversation remains available in `/history`, but future requests only send message `#39` and later. Any system prompt before message `#39` is still included.

Disable compaction:

```text
/compact clear
```

## Context Stats

When supported by the server, `llmchat` displays token usage above the prompt:

```text
Context: 3,120/8,192 tokens (38.1%, 5,072 left, max out 2,048) | next: #43 | active: #39-#42 + system | messages: 42
```

For llama.cpp-style servers, `llmchat` can use:

- `GET /props`
- `POST /v1/chat/completions/input_tokens`
- `POST /apply-template`
- `POST /tokenize`

If token counting is unavailable, it falls back to an approximate character-based estimate.
