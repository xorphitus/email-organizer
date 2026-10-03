# email-organizer

On-demand CLI that reads inbox mail from **Fastmail** (JMAP) and **ProtonMail** (via Proton Mail
Bridge IMAP), classifies each message locally with [Cloudflare/clef](https://huggingface.co/Cloudflare/clef),
and moves it into a folder. Categories live in a YAML file you edit.

Clef is a 27B decision model: you give it a `state` and `questions`, and one forward pass returns a
probability per option, so there is no text output to parse. Each email becomes a single `choice`
question whose options are your categories.

## Requirements

- NVIDIA GPU with 24 GB (tested target: RTX 3090). The model is ~54 GB in BF16, so it is loaded in
  4-bit NF4 (~16 GB) with bitsandbytes; input is capped at 4096 tokens.
- ~54 GB of disk for the model snapshot.
- Python 3.12 + [uv](https://docs.astral.sh/uv/). On NixOS use the flake's dev shell so the
  pip-installed torch can find `libcuda`:

```sh
nix develop          # or: direnv with `use flake`
uv sync
```

## Setup

### 1. Download the model

```sh
uv run hf download Cloudflare/clef      # ~54 GB into ~/.cache/huggingface
```

To keep it elsewhere, use `--local-dir /path/to/clef` and set `model.path` in the config.

### 2. Smoke-test the GPU

```sh
uv run email-organizer test-model
```

This loads the model, prints parameter counts per dtype (`uint8` = 4-bit; the vision tower,
`lm_head` and the joint head should be `bfloat16`), runs the two model-card examples, and reports
VRAM. It exits non-zero unless the support-ticket example routes to `technical` / `Today`.
If the modules kept in BF16 are wrong, adjust `model.skip_quant_modules`.

### 3. Config

```sh
mkdir -p ~/.config/email-organizer
cp config.example.yaml ~/.config/email-organizer/config.yaml
```

Each category has `criteria` (what Clef reads; describe the mail that belongs there) and a
`folder`. `folder: INBOX` means leave the message where it is. Messages whose top probability is
below `model.threshold` stay in the inbox and are reported as *uncertain*.

### 4. Fastmail

Settings → Privacy & Security → Integrations → **API tokens** → new token with **Mail** access
(read/write). Then:

```sh
export FASTMAIL_API_TOKEN=fmu1-…
```

Folders are matched by full path (`Parent/Child`) and created when missing.

### 5. ProtonMail (Bridge)

Requires a paid Proton plan. Install and sign in to [Proton Mail Bridge](https://proton.me/mail/bridge),
then copy the IMAP settings it shows (host `127.0.0.1`, port `1143`, STARTTLS, and a
Bridge-generated password, which is *not* your Proton password):

```sh
export PROTON_BRIDGE_PASSWORD=…
```

Bridge uses a self-signed certificate. Either pin it:

```sh
openssl s_client -connect 127.0.0.1:1143 -starttls imap </dev/null 2>/dev/null \
  | openssl x509 -noout -fingerprint -sha256
```

and put the value in `cert_fingerprint`, or set `verify: false` (only allowed for localhost).
A config folder `Receipts` maps to Bridge's `Folders/Receipts`. Use `Labels/<name>` explicitly to
target a label instead. Messages are fetched with `BODY.PEEK[]` (they stay unread) and tracked by
`Message-ID`, because Bridge UIDs can change across resyncs.

## Usage

```sh
uv run email-organizer run --account fastmail --limit 20 --dry-run   # classify only, print summary
uv run email-organizer run --account fastmail --limit 20             # move for real
uv run email-organizer runs                                          # list recent runs
uv run email-organizer undo <run_id>                                 # move that run's messages back
```

- `--account all` (default) processes every configured account.
- Messages already handled are skipped on later runs. `--reprocess` ignores that, which is useful
  after tuning criteria. Undone messages are forgotten, so the next run reconsiders them.
- `--show-all` lists every message with its top category and probability. Probabilities for all
  options are stored in the state DB either way.

State (processed messages + per-run decision log) lives in
`~/.local/state/email-organizer/state.db`.

## Development

```sh
uv run pytest
```

Tests use a fake provider, a stub classifier, an `httpx.MockTransport` JMAP server and a fake IMAP
client, so no GPU or network is needed.

## Tuning notes

- Start with `--dry-run`, check the uncertain list, and reword `criteria` or adjust `threshold`.
- If NF4 hurts accuracy noticeably, compare the logged probabilities against criteria changes
  before reaching for bigger quantization (`bf16` needs ~54 GB of VRAM).
- `batch_size` defaults to 2. A CUDA OOM retries that batch one message at a time.
