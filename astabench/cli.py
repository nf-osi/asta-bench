import hashlib
import importlib.metadata
import importlib.resources
import json
import logging
import os
import shutil
import tarfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import click
from agenteval.cli import (
    EVAL_CONFIG_FILENAME,
    SCORES_FILENAME,
    SUMMARY_FILENAME,
    cli as ae_cli,
    eval_command,
)
from agenteval.config import SuiteConfig, Task, load_suite_config
from agenteval.io import atomic_write_file
from agenteval.models import EvalConfig, TaskResults
from agenteval.score import TaskResult, process_eval_logs
from agenteval.summary import compute_summary_statistics
from litellm import model_cost as litellm_model_cost
from litellm import register_model

DEFAULT_CONFIG = "v1.0.0"
DEFAULT_SPLIT = "validation"
SPLIT_NAMES = ["validation", "test"]
LOCAL_COST_OVERRIDES = "litellm_cost_overrides.json"
ARCHIVE_SUFFIXES: tuple[tuple[str, str], ...] = (
    (".tar.gz", "tar"),
    (".tgz", "tar"),
    (".tar", "tar"),
    (".zip", "zip"),
    (".eval", "zip"),
)


def get_config_path(config_name: str) -> str:
    """Get the path to a config file in the package's config directory."""
    with importlib.resources.path("astabench.config", f"{config_name}.yml") as path:
        return os.path.abspath(path)


for param in eval_command.params:
    if isinstance(param, click.Option) and param.name == "config_path":
        param.default = get_config_path(DEFAULT_CONFIG)
        param.show_default = True
        param.hidden = True
    elif isinstance(param, click.Option) and param.name == "split":
        param.type = click.Choice(SPLIT_NAMES, case_sensitive=False)


def _load_override_cost_map() -> dict:
    overrides_path = importlib.resources.files("astabench.config") / LOCAL_COST_OVERRIDES
    if not overrides_path.is_file():
        raise click.ClickException(
            f"Missing cost override file at {overrides_path}. Add your model rates there."
        )
    with overrides_path.open("r", encoding="utf-8") as overrides_file:
        return json.load(overrides_file)


def _register_cost_map() -> None:
    model_costs = _load_override_cost_map()
    register_model(model_cost=model_costs)

    hash_obj = hashlib.sha256()
    hash_obj.update(json.dumps(litellm_model_cost, sort_keys=True).encode())
    click.echo(f"Model costs hash {hash_obj.hexdigest()}.")
    click.echo(f"litellm version: {importlib.metadata.version('litellm')}")


@contextmanager
def _suppress_agenteval_warnings():
    target_loggers = [
        logging.getLogger("agenteval.score"),
        logging.getLogger("agenteval.summary"),
    ]
    prev_levels = [logger.level for logger in target_loggers]
    try:
        for logger in target_loggers:
            logger.setLevel(logging.ERROR)
        yield
    finally:
        for logger, level in zip(target_loggers, prev_levels):
            logger.setLevel(level)


def _match_archive_suffix(name: str) -> tuple[str, str] | None:
    for suffix, kind in ARCHIVE_SUFFIXES:
        if name.endswith(suffix):
            return suffix, kind
    return None


def _extract_archives(root: Path) -> None:
    for item in sorted(root.iterdir()):
        if not item.is_file():
            continue
        match = _match_archive_suffix(item.name)
        if not match:
            continue
        suffix, archive_type = match
        dest_name = item.name[: -len(suffix)]
        dest = item.with_name(dest_name)
        if dest.exists():
            continue
        click.echo(f"Extracting {item.name} -> {dest.name}")
        dest.mkdir(parents=True, exist_ok=True)
        try:
            if archive_type == "tar":
                with tarfile.open(item, "r:*") as tar:
                    tar.extractall(dest)
            else:
                with zipfile.ZipFile(item) as zf:
                    zf.extractall(dest)
        except (tarfile.TarError, zipfile.BadZipFile) as exc:
            raise click.ClickException(f"Failed to extract {item}: {exc}") from exc


def _looks_like_log_dir(path: Path) -> bool:
    return path.is_dir() and (
        (path / EVAL_CONFIG_FILENAME).exists() or (path / "_journal").is_dir()
    )


def _discover_log_dirs(root: Path) -> list[Path]:
    if _looks_like_log_dir(root):
        return [root]

    log_dirs = {cfg.parent for cfg in root.rglob(EVAL_CONFIG_FILENAME)}
    if not log_dirs:
        log_dirs = {journal.parent for journal in root.rglob("_journal") if journal.is_dir()}

    if not log_dirs:
        log_dirs = {child for child in root.iterdir() if _looks_like_log_dir(child)}

    return sorted(log_dirs)


def _ensure_eval_archive(log_dir: Path, search_roots: Iterable[Path]) -> None:
    if any(log_dir.glob("*.eval")):
        return

    candidates = []
    seen: set[Path] = set()
    for base in search_roots:
        if not base:
            continue
        candidate = base / f"{log_dir.name}.eval"
        if candidate.exists():
            candidates.append(candidate)
        seen.add(base)

    if not candidates:
        # Fallback: look for matching archives directly under each search root
        for base in seen:
            for path in base.glob("*.eval"):
                if path.stem == log_dir.name:
                    candidates.append(path)
                    break
            if candidates:
                break

    if not candidates:
        click.echo(f"Warning: No .eval archive found for {log_dir}. Model usage costs may be missing.")
        return

    destination = log_dir / candidates[0].name
    shutil.copy2(candidates[0], destination)


def _ensure_eval_config(log_dir: Path, suite_config, split: str) -> None:
    config_path = log_dir / EVAL_CONFIG_FILENAME
    if config_path.exists():
        return
    eval_config = EvalConfig(suite_config=suite_config, split=split)
    atomic_write_file(config_path, eval_config.model_dump_json(indent=2))
    click.echo(f"Created {EVAL_CONFIG_FILENAME} for {log_dir} using split={split}.")


def _suite_config_with_results(
    suite_config: SuiteConfig, split: str, results: list[TaskResult] | None
) -> SuiteConfig:
    split_obj = suite_config.get_split(split)
    existing_task_names = {task.name for task in split_obj.tasks} if split_obj else set()
    result_names = {res.task_name for res in results or []}
    if not result_names:
        return suite_config

    if result_names.issubset(existing_task_names):
        return suite_config

    from agenteval.config import Split  # local import to avoid circular import

    tasks = []
    for result in results or []:
        primary_metric = result.metrics[0].name if result.metrics else "score"
        tasks.append(
            Task(
                name=result.task_name,
                path=result.task_name,
                primary_metric=primary_metric,
                tags=["custom"],
            )
        )

    return SuiteConfig(
        name=f"{suite_config.name}-local",
        version=suite_config.version,
        splits=[Split(name=split, tasks=tasks, macro_average_weight_adjustments=None)],
    )


def _score_single_log_dir(log_dir: Path) -> None:
    config_path = log_dir / EVAL_CONFIG_FILENAME
    if not config_path.exists():
        raise click.ClickException(f"Missing {EVAL_CONFIG_FILENAME} in {log_dir}")

    eval_config = EvalConfig.model_validate_json(config_path.read_text())
    with _suppress_agenteval_warnings():
        outcome = process_eval_logs(
            str(log_dir),
            reference_tasks=eval_config.suite_config.get_tasks(eval_config.split),
        )
    if outcome.errors:
        for error in outcome.errors:
            click.echo(f"  - {error}")
        raise click.ClickException(f"Errors processing logs in {log_dir}")

    task_results = TaskResults(results=outcome.results)
    suite_config = _suite_config_with_results(
        eval_config.suite_config, eval_config.split, task_results.results
    )
    with _suppress_agenteval_warnings():
        stats = compute_summary_statistics(
            suite_config,
            eval_config.split,
            task_results.results or [],
        )

    scores_path = log_dir / SCORES_FILENAME
    atomic_write_file(scores_path, task_results.model_dump_json(indent=2))
    click.echo(f"Wrote scores to {scores_path}")

    summary_path = log_dir / SUMMARY_FILENAME
    atomic_write_file(summary_path, json.dumps(stats.model_dump(mode="json"), indent=2))
    click.echo(f"Wrote summary scores to {summary_path}")


cli = ae_cli
cli.commands.pop("score", None)


@cli.command(
    name="score",
    help="Score one or more log directories (or archives) and compute usage costs.",
)
@click.argument("log_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--split",
    type=click.Choice(SPLIT_NAMES, case_sensitive=False),
    default=DEFAULT_SPLIT,
    show_default=True,
    help="Split to assume when generating eval_config.json for extracted logs missing one.",
)
@click.option(
    "--config-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=get_config_path(DEFAULT_CONFIG),
    show_default=True,
    help="Suite config used when generating eval_config.json entries.",
)
def astabench_score(log_dir: Path, split: str, config_path: Path) -> None:
    log_dir = log_dir.resolve()
    click.echo(f"Preparing logs in {log_dir}")
    if not _looks_like_log_dir(log_dir):
        _extract_archives(log_dir)

    log_dirs = _discover_log_dirs(log_dir)
    if not log_dirs:
        raise click.ClickException(
            f"No log directories found under {log_dir}. Ensure archives were extracted."
        )

    suite_config = load_suite_config(str(config_path))
    _register_cost_map()

    failures: list[tuple[Path, Exception]] = []
    for idx, run_dir in enumerate(log_dirs, start=1):
        click.echo(f"[{idx}/{len(log_dirs)}] Scoring {run_dir}")
        try:
            _ensure_eval_archive(run_dir, {run_dir, run_dir.parent, log_dir})
            _ensure_eval_config(run_dir, suite_config, split)
            _score_single_log_dir(run_dir)
        except Exception as exc:  # pragma: no cover - surfaced to user
            failures.append((run_dir, exc))

    if failures:
        details = "\n".join(f"- {path}: {exc}" for path, exc in failures)
        raise click.ClickException(f"Failed to score some logs:\n{details}")


if __name__ == "__main__":
    cli()
