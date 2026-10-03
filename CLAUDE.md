# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

On-demand CLI that fetches inbox mail from Fastmail (JMAP) and ProtonMail (Proton Mail Bridge IMAP), classifies each message locally with Cloudflare/clef, a 27B decision model, and moves it into a folder. Categories come from a YAML config. The target machine is NixOS with an RTX 3090 (24 GB), so Clef runs 4-bit NF4 via bitsandbytes and uses about 17 GB of VRAM.

## Commands

Run everything inside `nix develop`. It sets `LD_LIBRARY_PATH` (libcuda from `/run/opengl-driver/lib`), `TRITON_LIBCUDA_PATH` (NixOS has no `/sbin/ldconfig`), and the Python uv uses.

```sh
uv sync
uv run pytest                                   # all tests (no GPU/network needed)
uv run pytest tests/test_fastmail.py::test_move_errors   # single test
uv run email-organizer test-model               # GPU smoke test: load Clef, run model-card examples
uv run email-organizer quantize                 # one-off: save NF4 copy to ~/.local/share/email-organizer/clef-nf4
uv run email-organizer run --account fastmail --limit 20 --dry-run --show-all --config config.yaml
uv run email-organizer undo <run_id>
uv run email-organizer runs
```

Config defaults to `~/.config/email-organizer/config.yaml` (or `$EMAIL_ORGANIZER_CONFIG`). A local `config.yaml` in the repo root is gitignored, and is used with `--config config.yaml`. Secrets come from the env vars named in the config: `FASTMAIL_API_TOKEN` and `PROTON_BRIDGE_PASSWORD`.

The Claude Code Bash sandbox has no GPU, no network, and a read-only uv cache. Inside it, run tests with `.venv/bin/python -m pytest -p no:cacheprovider`, since `uv run` fails there. The user has to run `test-model`, `quantize`, `run` and `undo`. Commits need `--no-gpg-sign` because `~/.gnupg` is hidden, and only with the user's permission.

## Architecture

The flow is `cli.py` → `organizer.run_account()` → provider + classifier + state. `organizer.py` holds all run/undo logic, independent of the CLI, so tests drive it with `FakeProvider`/`StubClassifier` from `tests/conftest.py`.

- **Run pipeline** (`organizer.run_account`): fetch up to `limit` unprocessed inbox messages, classify all of them, move them grouped by target folder, then record every decision. Moves and records happen only after classification has finished.
- **Decisions → actions** (`plan_action`): top probability below `threshold` is `uncertain` (left in place). A category whose `folder` is `INBOX` is `kept`. Otherwise the action is `moved`. Dry runs log `would-moved` and never mark messages processed. Failed moves are logged as `error` and are not marked processed, so they get retried.
- **State** (`state.py`, SQLite at `~/.local/state/email-organizer/state.db`): `processed(account, msg_key)` drives skipping. `decisions` is the per-run log used by `undo`. Undo moves `moved` rows back to `from_folder` and deletes their `processed` rows, so the next run reconsiders those messages.
- **Provider contract** (`providers/base.py`): `fetch_inbox(limit, skip)`, `ensure_folders(names)`, `move(refs, folder)`, `close()`. `MessageRef(id, key, folder)` carries what a move needs. `key` is the stable dedupe key. Folder names are config-level; `"INBOX"` means the inbox (see `is_inbox`).
  - **Fastmail** (`fastmail.py`): `id` = `key` = the JMAP email id. A move sets `mailboxIds` to `{target: true}`, which replaces all mailbox membership. Folders resolve by `Parent/Child` path. `List-Unsubscribe` must be requested as `header:…:asURLs`. RFC 8621 forbids `asText` for RFC 2369 headers, and Fastmail returns `invalidArguments` for it.
  - **Proton** (`proton.py`): `key` = Message-ID, because Bridge UIDs change across resyncs. `id` = `"uidvalidity:uid"`, and it's trusted only for INBOX with a matching UIDVALIDITY. Otherwise the message is located with `SEARCH HEADER Message-ID`, which is how undo finds messages in their target folder. Config folder `X` maps to `Folders/X`, and a `Labels/…` name passes through unchanged. Fetches use `BODY.PEEK[]`, so messages stay unread. Pass a fake client through `client_factory` in tests.
- **Classifier** (`classifier.py`): torch and transformers are imported lazily, so the CLI and tests work without the ML stack. It downloads or locates the snapshot, puts it on `sys.path`, and imports `load_release_model`, `encode_record` and `collate_records` from Clef's own `joint_schema_model.py`. Each email becomes one `choice` question whose options are the category ids (`build_record`). `classify()` batches by `batch_size` and retries one message at a time on CUDA OOM.

## Clef facts (verified from `joint_schema_model.py` in the HF snapshot)

- `load_release_model` calls `Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype, device_map={"": device}, **kwargs)`. Passing `device_map` yourself conflicts with it. The joint head is loaded separately from `joint_head.safetensors` and stays BF16.
- The forward pass runs only the text model (`base_model.model.language_model` when there is no media), never `lm_head` logits. The head does read `lm_head.weight` rows for option embeddings, so `lm_head` must stay unquantized (`skip_quant_modules`).
- `encode_record` trims the state to fit `max_length`, never the schema. VRAM headroom beyond the weights is about 6 GB, so the defaults are `max_length=2048` and `batch_size=1`. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set in `email_organizer/__init__.py` before torch loads.
- A `score` question returns option ids `"0"…"n-1"`, not labels.
- `quantize` writes a bnb-4bit checkpoint plus Clef's non-weight files. `is_prequantized()` detects `quantization_config` in its `config.json`, and the loader then skips passing a `BitsAndBytesConfig`.
- The message about the "fast path" for flash-linear-attention or causal-conv1d is harmless. transformers falls back to the PyTorch implementation, which is slower.

## Gotchas

- Don't wrap model loading in `rich` `console.status()`. Its live display captures stderr, which breaks the transformers progress bar.
- `config.py` models use `extra="forbid"`, so unknown YAML keys are errors. `verify: false` for Proton is rejected unless the host is localhost.
