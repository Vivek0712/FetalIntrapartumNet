"""The ``fin`` command: one subcommand per step of the pipeline.

    fin <command> [arguments]       (also: python -m fin <command> ...)
    fin <command> --help            arguments of one command

Each command runs the ``main`` function of the module listed below.
"""

from __future__ import annotations

import importlib
import sys

from fin import PaperDirectoryError, RepositoryNotFoundError

COMMANDS: dict[str, tuple[str, str]] = {
    "jnu": ("fin.data.jnu", "JNU-IFM: build (frame table, masks, locked split) or parse-descriptor"),
    "match-aop": ("fin.data.fhpsaop", "match JNU-IFM SPHead frames to the FH-PS-AoP release; reference AoPs"),
    "build-iugc": ("fin.data.iugc", "IUGC 2024 external test set"),
    "splits": ("fin.data.splits", "patient-grouped cross-validation folds"),
    "cache": ("fin.data.cache", "classification / segmentation image caches"),
    "fetch-third-party": ("fin.third_party", "fetch FAT-Net, MALUNet and the challenge AoP code at pinned commits"),
    "train-cls": ("fin.training.classification", "train one classifier"),
    "eval-cls": ("fin.evaluation.classification", "evaluate one classifier run on the test split"),
    "train-seg": ("fin.training.segmentation", "train one segmenter"),
    "eval-seg": ("fin.evaluation.segmentation", "evaluate one segmenter run; AoP ceiling"),
    "schedule": ("fin.pipeline", "resumable schedules: cls, cv, authorsplit, seg"),
    "aggregate": ("fin.evaluation.aggregate", "summaries: cls, seg, cv, author-split"),
    "ext469": ("fin.evaluation.external469", "second external AoP set (IUGC 2025 release)"),
    "e2e": ("fin.evaluation.e2e", "end-to-end gate -> segmenter -> AoP on the JNU-IFM test split"),
    "check-geometry": ("fin.evaluation.geometry_check", "verify fin.geometry against the unchanged challenge function"),
    "robustness": ("fin.robustness.run", "corruption benchmark of the classifiers"),
    "efficiency": ("fin.efficiency.bench", "parameters, GMACs, latency, throughput"),
    "collect-results": ("fin.reporting.collect", "copy small run outputs into results/runs (portable paths)"),
    "fill-results": ("fin.reporting.fill_results", "manuscript values and Tables 1-5 (--paper-dir / FIN_PAPER)"),
    "supp-tables": ("fin.reporting.supp_tables", "supplementary tables (--paper-dir / FIN_PAPER)"),
    "figures": ("fin.reporting.figures", "result figures 6-9 and Bland-Altman (--paper-dir / FIN_PAPER)"),
    "method-figures": ("fin.reporting.method_figures", "method figures 1-5 (--paper-dir / FIN_PAPER)"),
    "readme-table": ("fin.reporting.readme_table", "README results table from results/runs"),
    "repo-figures": ("fin.reporting.repo_figures", "README pipeline diagram, result charts and example (docs/assets)"),
    "describe-keys": ("fin.reporting.describe_keys", "registry lines for new \\res{} keys (--paper-dir / FIN_PAPER)"),
}


def usage() -> str:
    """Help text listing every command."""
    width = max(len(c) for c in COMMANDS)
    lines = [__doc__.strip(), "", "commands:"]
    lines += [f"  {c:<{width}}  {h}" for c, (_, h) in COMMANDS.items()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """Dispatch ``fin <command> ...`` to the command's module."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        sys.stdout.write(usage() + "\n")
        return
    cmd, rest = argv[0], argv[1:]
    if cmd not in COMMANDS:
        sys.exit(f"unknown command {cmd!r}\n\n{usage()}")
    try:
        module = importlib.import_module(COMMANDS[cmd][0])
    except RepositoryNotFoundError as e:
        sys.exit(f"fin {cmd}: {e}")
    sys.argv = [f"fin {cmd}", *rest]
    try:
        module.main(rest)
    except PaperDirectoryError as e:
        sys.exit(f"fin {cmd}: {e}")


if __name__ == "__main__":
    main()
