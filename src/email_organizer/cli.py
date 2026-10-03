from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .config import DEFAULT_CONFIG_PATH, Config, load_config

app = typer.Typer(no_args_is_help=True, help="Classify inbox mail with Cloudflare Clef and file it into folders.")
console = Console()

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="Path to config.yaml")]


def _load(config: Path) -> Config:
    try:
        return load_config(config)
    except FileNotFoundError:
        console.print(f"[red]config not found: {config}[/] (copy config.example.yaml there)")
        raise typer.Exit(2)


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


@app.command()
def run(
    account: Annotated[str, typer.Option(help="Account name from the config, or 'all'")] = "all",
    limit: Annotated[int, typer.Option(help="Max unprocessed inbox messages per account")] = 200,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Classify only; do not move anything")] = False,
    reprocess: Annotated[bool, typer.Option("--reprocess", help="Ignore the processed-messages list")] = False,
    show_all: Annotated[bool, typer.Option("--show-all", help="List every message, not just uncertain ones")] = False,
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
) -> None:
    """Fetch inbox mail, classify it, and move it into category folders."""
    from .classifier import ClefClassifier
    from .organizer import run_account
    from .providers import make_provider
    from .state import StateDB

    cfg = _load(config)
    accounts = cfg.accounts if account == "all" else [cfg.account(account)]
    state = StateDB(cfg.state_db)
    run_id = state.start_run(dry_run)

    classifier: ClefClassifier | None = None

    def get_classifier() -> ClefClassifier:
        nonlocal classifier
        if classifier is None:
            # No console.status() here: rich would capture the transformers progress bar.
            console.print("loading Clef…")
            classifier = ClefClassifier(cfg.model, cfg.categories)
            classifier.load()
        return classifier

    all_outcomes = []
    for acct in accounts:
        try:
            provider = make_provider(acct)
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]{acct.name}: cannot connect: {e}[/]")
            continue
        try:
            outcomes = run_account(
                provider, get_classifier, state, cfg.categories, run_id,
                limit=limit, dry_run=dry_run, reprocess=reprocess,
            )
        finally:
            provider.close()
        console.print(f"{acct.name}: {len(outcomes)} message(s) classified")
        all_outcomes.extend((acct.name, o) for o in outcomes)

    _print_summary(run_id, all_outcomes, dry_run, show_all)


def _print_summary(run_id: str, outcomes: list, dry_run: bool, show_all: bool) -> None:
    counts = Counter((acct, o.decision.category or o.action, o.action) for acct, o in outcomes)
    t = Table(title=f"run {run_id}" + (" (dry run)" if dry_run else ""))
    for col in ("account", "category", "action", "count"):
        t.add_column(col)
    for (acct, cat, action), n in sorted(counts.items()):
        t.add_row(acct, cat, f"would-{action}" if dry_run and action == "moved" else action, str(n))
    console.print(t)

    listed = [(a, o) for a, o in outcomes if show_all or o.action in ("uncertain", "error")]
    if listed:
        d = Table(title="messages" if show_all else "uncertain / errors")
        for col in ("account", "from", "subject", "top", "p", "action"):
            d.add_column(col, overflow="fold")
        for acct, o in listed:
            d.add_row(
                acct, o.message.sender[:40], o.message.subject[:60],
                o.decision.top or "-", f"{o.decision.prob:.2f}", o.decision.error or o.action,
            )
        console.print(d)
    if not dry_run and any(o.action == "moved" for _, o in outcomes):
        console.print(f"undo with: [bold]email-organizer undo {run_id}[/]")


@app.command()
def undo(run_id: str, config: ConfigOpt = DEFAULT_CONFIG_PATH) -> None:
    """Move the messages a run moved back to the inbox."""
    from .organizer import undo_run
    from .providers import make_provider
    from .state import StateDB

    cfg = _load(config)
    state = StateDB(cfg.state_db)
    providers = {}

    def provider_for(name: str):
        if name not in providers:
            providers[name] = make_provider(cfg.account(name))
        return providers[name]

    try:
        counts = undo_run(state, run_id, provider_for)
    except (KeyError, ValueError) as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    finally:
        for p in providers.values():
            p.close()
    if not counts:
        console.print("nothing to undo")
    for acct, n in counts.items():
        console.print(f"{acct}: moved {n} message(s) back")


@app.command()
def runs(limit: int = 10, config: ConfigOpt = DEFAULT_CONFIG_PATH) -> None:
    """List recent runs."""
    from .state import StateDB

    cfg = _load(config)
    t = Table()
    for col in ("run_id", "started (UTC)", "dry", "messages", "moved", "undone"):
        t.add_column(col)
    for r in StateDB(cfg.state_db).recent_runs(limit):
        t.add_row(r["run_id"], r["started_at"], "yes" if r["dry_run"] else "", str(r["n"]),
                  str(r["moved"] or 0), str(r["undone"] or 0))
    console.print(t)


@app.command("test-model")
def test_model(config: ConfigOpt = DEFAULT_CONFIG_PATH) -> None:
    """Load Clef, run the model-card examples, and report probabilities and VRAM."""
    import torch

    from .classifier import MODEL_CARD_EXAMPLES, ClefClassifier, dtype_report
    from .config import ModelConfig

    cfg = load_config(config).model if config.exists() else ModelConfig()
    if not torch.cuda.is_available():
        console.print("[red]CUDA is not available to torch (check LD_LIBRARY_PATH / driver)[/]")
        raise typer.Exit(1)
    console.print(f"GPU: {torch.cuda.get_device_name(0)}, quantization: {cfg.quantization}")

    clf = ClefClassifier(cfg, {})
    console.print("loading Clef…")
    clf.load()
    console.print(f"VRAM after load: {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

    t = Table(title="parameters by dtype (uint8 = 4-bit)")
    for col in ("module", "dtype", "params"):
        t.add_column(col)
    for prefix, dtypes in dtype_report(clf.model).items():
        for dt, n in dtypes.items():
            t.add_row(prefix, dt, f"{n:,}")
    console.print(t)

    torch.cuda.reset_peak_memory_stats()
    results = clf.predict(MODEL_CARD_EXAMPLES)
    for i, res in enumerate(results, 1):
        console.print(f"[bold]example {i}[/]")
        for qid, probs in res.items():
            console.print(f"  {qid}: " + ", ".join(f"{k}={v:.3f}" for k, v in probs.items()))
    console.print(f"peak VRAM during inference: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

    ticket = results[1]
    urgency = ticket["urgency"]
    # Score options may be ids or labels; "Today" is the last (highest) option either way.
    ok = (max(ticket["department"], key=ticket["department"].get) == "technical"
          and max(urgency, key=urgency.get) == list(urgency)[-1])
    console.print("[green]support ticket → technical / Today: OK[/]" if ok
                  else "[red]support ticket did not route to technical / Today[/]")
    raise typer.Exit(0 if ok else 1)
