"""YAML configuration, validated with pydantic."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_CONFIG_PATH = Path(
    os.environ.get("EMAIL_ORGANIZER_CONFIG", "~/.config/email-organizer/config.yaml")
).expanduser()
DEFAULT_STATE_DB = Path(
    os.environ.get("XDG_STATE_HOME", "~/.local/state")
).expanduser() / "email-organizer" / "state.db"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(_Strict):
    repo: str = "Cloudflare/clef"
    # Local snapshot directory; when unset the repo is resolved via snapshot_download.
    path: Path | None = None
    quantization: Literal["nf4", "bf16"] = "nf4"
    max_length: int = Field(4096, ge=256)
    # Passed to encode_record so long bodies are cut before the schema is.
    max_state_tokens: int | None = 3584
    threshold: float = Field(0.6, ge=0.0, le=1.0)
    batch_size: int = Field(2, ge=1)
    # Body characters kept before tokenization (cheap pre-truncation).
    body_chars: int = Field(12000, ge=0)
    # Modules kept in BF16 when quantizing. Names follow Qwen3_5ForConditionalGeneration;
    # `email-organizer test-model` prints which modules ended up unquantized.
    skip_quant_modules: list[str] = ["visual", "lm_head"]
    instructions: str = "Which folder should this email be filed in?"


class Category(_Strict):
    criteria: str
    folder: str

    @field_validator("criteria", "folder")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v.strip()


class FastmailAccount(_Strict):
    type: Literal["fastmail"]
    name: str
    token_env: str = "FASTMAIL_API_TOKEN"
    session_url: str = "https://api.fastmail.com/jmap/session"

    def token(self) -> str:
        token = os.environ.get(self.token_env)
        if not token:
            raise RuntimeError(f"account {self.name}: environment variable {self.token_env} is not set")
        return token


class ProtonAccount(_Strict):
    type: Literal["proton"]
    name: str
    host: str = "127.0.0.1"
    port: int = 1143
    username: str
    password_env: str = "PROTON_BRIDGE_PASSWORD"
    # TLS: by default the Bridge cert must be pinned (cert_fingerprint, SHA-256 hex) or
    # trusted via ca_file. verify: false disables checks and is only allowed for localhost.
    verify: bool = True
    cert_fingerprint: str | None = None
    ca_file: Path | None = None
    folder_prefix: str = "Folders/"

    @field_validator("verify")
    @classmethod
    def _verify_localhost_only(cls, v: bool, info) -> bool:
        host = info.data.get("host", "")
        if not v and host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("verify: false is only allowed for a localhost Bridge")
        return v

    def password(self) -> str:
        pw = os.environ.get(self.password_env)
        if not pw:
            raise RuntimeError(f"account {self.name}: environment variable {self.password_env} is not set")
        return pw


Account = Annotated[FastmailAccount | ProtonAccount, Field(discriminator="type")]


class Config(_Strict):
    model: ModelConfig = ModelConfig()
    categories: dict[str, Category]
    accounts: list[Account]
    state_db: Path = DEFAULT_STATE_DB

    @field_validator("categories")
    @classmethod
    def _at_least_two(cls, v: dict[str, Category]) -> dict[str, Category]:
        if len(v) < 2:
            raise ValueError("define at least two categories")
        return v

    @field_validator("accounts")
    @classmethod
    def _unique_names(cls, v: list) -> list:
        names = [a.name for a in v]
        if len(names) != len(set(names)):
            raise ValueError("account names must be unique")
        return v

    def account(self, name: str) -> FastmailAccount | ProtonAccount:
        for a in self.accounts:
            if a.name == name:
                return a
        raise KeyError(name)


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> Config:
    path = Path(path).expanduser()
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    cfg = Config.model_validate(data)
    cfg.state_db = cfg.state_db.expanduser()
    return cfg
