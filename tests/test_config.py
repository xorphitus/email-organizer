from pathlib import Path

import pytest
from pydantic import ValidationError

from email_organizer.config import Config, FastmailAccount, ProtonAccount, load_config

ROOT = Path(__file__).resolve().parent.parent


def test_example_config_loads():
    cfg = load_config(ROOT / "config.example.yaml")
    assert cfg.model.quantization == "nf4"
    assert cfg.model.threshold == 0.6
    assert cfg.model.batch_size == 1
    assert set(cfg.categories) == {"important", "paper_trail", "feed"}
    assert cfg.categories["important"].folder == "INBOX"
    assert cfg.categories["paper_trail"].folder == "Paper Trail"
    assert isinstance(cfg.account("fastmail"), FastmailAccount)
    proton = cfg.account("proton")
    assert isinstance(proton, ProtonAccount) and proton.port == 1143


def _base(**over):
    data = {
        "categories": {"a": {"criteria": "A", "folder": "A"}, "b": {"criteria": "B", "folder": "B"}},
        "accounts": [{"name": "fm", "type": "fastmail"}],
    }
    data.update(over)
    return data


def test_defaults():
    cfg = Config.model_validate(_base())
    assert cfg.model.max_length == 2048 and cfg.model.batch_size == 1
    assert cfg.model.repo == "Cloudflare/clef"


@pytest.mark.parametrize(
    "over",
    [
        {"categories": {"a": {"criteria": "A", "folder": "A"}}},  # fewer than two
        {"categories": {"a": {"criteria": " ", "folder": "A"}, "b": {"criteria": "B", "folder": "B"}}},
        {"model": {"threshold": 1.5}},
        {"model": {"quantization": "int3"}},
        {"model": {"unknown_key": 1}},
        {"accounts": [{"name": "x", "type": "gmail"}]},
        {"accounts": [{"name": "x", "type": "fastmail"}, {"name": "x", "type": "fastmail"}]},
        {"accounts": [{"name": "p", "type": "proton", "username": "u", "host": "10.0.0.5", "verify": False}]},
    ],
)
def test_invalid(over):
    with pytest.raises(ValidationError):
        Config.model_validate(_base(**over))


def test_token_from_env(monkeypatch):
    acct = FastmailAccount(type="fastmail", name="fm", token_env="TEST_FM_TOKEN")
    monkeypatch.delenv("TEST_FM_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="TEST_FM_TOKEN"):
        acct.token()
    monkeypatch.setenv("TEST_FM_TOKEN", "secret")
    assert acct.token() == "secret"
