"""High-level helper that runs the optimiser with guided prompts."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from optimize.run import execute, load_yaml, parse_args


def _prompt_choice(label: str, choices: Iterable[str], default: Optional[str] = None) -> str:
    items = list(dict.fromkeys(str(choice) for choice in choices if choice))
    if not items:
        return input(f"{label} (enter value): ").strip() or (default or "")

    while True:
        print(f"\n{label}:")
        for idx, value in enumerate(items, start=1):
            marker = " (default)" if default and value == default else ""
            print(f"  {idx}. {value}{marker}")
        raw = input("Select option (press Enter for default or number): ").strip()
        if not raw:
            return default or items[0]
        if raw.isdigit():
            sel = int(raw)
            if 1 <= sel <= len(items):
                return items[sel - 1]
        if raw:
            return raw
        print("Invalid selection. Try again.")


def _prompt_bool(label: str) -> Optional[bool]:
    while True:
        raw = input(f"Enable {label}? [y/n, Enter=skip]: ").strip().lower()
        if not raw:
            return None
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please answer with 'y' or 'n'.")


def _prompt_float(label: str, default: Optional[float]) -> Optional[float]:
    suffix = f" [{default}]" if default is not None else ""
    raw = input(f"{label}{suffix}: ").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print("Invalid number, ignoring input.")
        return default


def _prompt_int(label: str, default: Optional[int]) -> Optional[int]:
    suffix = f" [{default}]" if default is not None else ""
    raw = input(f"{label}{suffix}: ").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print("Invalid integer, ignoring input.")
        return default


def _extract_symbol_choices(backtest_cfg: Dict[str, object]) -> List[str]:
    choices: List[str] = []
    for entry in backtest_cfg.get("symbols", []) or []:
        if isinstance(entry, dict):
            alias = entry.get("alias") or entry.get("name") or entry.get("symbol") or entry.get("id")
            if alias:
                choices.append(str(alias))
        else:
            choices.append(str(entry))
    return choices


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive quick-start for Pine optimisation")
    parser.add_argument("--params", type=Path, default=Path("config/params.yaml"))
    parser.add_argument("--backtest", type=Path, default=Path("config/backtest.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--n-trials", type=int, help="Override Optuna trial count")
    args = parser.parse_args()

    params_cfg = load_yaml(args.params)
    backtest_cfg = load_yaml(args.backtest)

    symbol_default = str(params_cfg.get("symbol", "")) or None
    backtest_cfg.setdefault("symbols", backtest_cfg.get("symbols", []))

    symbol = _prompt_choice("Select symbol", _extract_symbol_choices(backtest_cfg) or [symbol_default or ""], symbol_default)

    backtest_window = params_cfg.get("backtest", {}) or {}
    start_default = backtest_window.get("from") or ""
    end_default = backtest_window.get("to") or ""
    start = input(f"Backtest start date [{start_default}]: ").strip() or start_default
    end = input(f"Backtest end date [{end_default}]: ").strip() or end_default

    risk_cfg = params_cfg.get("risk", {}) or {}
    leverage = _prompt_float("Leverage", risk_cfg.get("leverage"))
    qty_pct = _prompt_float("Position size %", risk_cfg.get("qty_pct"))

    if args.n_trials is not None:
        trial_override = args.n_trials
    else:
        trial_override = _prompt_int("Optuna trials", params_cfg.get("search", {}).get("n_trials"))

    bool_params = [name for name, spec in (params_cfg.get("space") or {}).items() if isinstance(spec, dict) and spec.get("type") == "bool"]
    enable_tokens: List[str] = []
    disable_tokens: List[str] = []
    if bool_params:
        print("\nToggle optional filters:")
        auto_yes = input("모든 옵션을 예(Enable)로 설정할까요? [y/N]: ").strip().lower()
        if auto_yes in {"y", "yes"}:
            enable_tokens.extend(bool_params)
        else:
            for name in bool_params:
                decision = _prompt_bool(name)
                if decision is True:
                    enable_tokens.append(name)
                elif decision is False:
                    disable_tokens.append(name)

    cli_args: List[str] = [
        "--params",
        str(args.params),
        "--backtest",
        str(args.backtest),
        "--output",
        str(args.output),
        "--data",
        str(args.data),
    ]

    if symbol:
        cli_args.extend(["--symbol", symbol])
    if start:
        cli_args.extend(["--start", start])
    if end:
        cli_args.extend(["--end", end])
    if leverage is not None:
        cli_args.extend(["--leverage", str(leverage)])
    if qty_pct is not None:
        cli_args.extend(["--qty-pct", str(qty_pct)])
    if trial_override is not None:
        cli_args.extend(["--n-trials", str(trial_override)])

    for token in enable_tokens:
        cli_args.extend(["--enable", token])
    for token in disable_tokens:
        cli_args.extend(["--disable", token])

    print("\nStarting optimisation... this may take a few minutes depending on n_trials.")
    execute(parse_args(cli_args))
    print("\nOptimisation complete. Review the generated files in", args.output)


if __name__ == "__main__":
    main()

