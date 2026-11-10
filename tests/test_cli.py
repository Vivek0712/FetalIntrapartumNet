"""The ``fin`` command: every subcommand is listed and has argparse help."""

import importlib

import pytest

from fin import cli


def test_usage_lists_every_command(capsys):
    cli.main([])
    out = capsys.readouterr().out
    for cmd in cli.COMMANDS:
        assert cmd in out


@pytest.mark.parametrize("cmd", list(cli.COMMANDS))
def test_every_command_has_help(cmd):
    module = importlib.import_module(cli.COMMANDS[cmd][0])
    assert module.__doc__
    with pytest.raises(SystemExit) as e:
        module.main(["--help"])
    assert e.value.code == 0


@pytest.mark.parametrize("cmd", ["fill-results", "supp-tables", "figures", "method-figures"])
def test_paper_commands_need_a_manuscript_folder(cmd, monkeypatch):
    monkeypatch.delenv("FIN_PAPER", raising=False)
    with pytest.raises(SystemExit) as e:
        cli.main([cmd])
    assert "--paper-dir" in str(e.value.code)
