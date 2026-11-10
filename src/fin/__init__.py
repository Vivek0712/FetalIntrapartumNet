"""FetalIntrapartumNet: official code of the paper.

"FetalIntrapartumNet: a controlled evaluation of standard-plane selection, segmentation and
angle-of-progression measurement in intrapartum transperineal ultrasound" (B Lakshmanan, Vivek Raja P S,
J Aron Yuvaraj, M P Nitharsan).

Subpackages
    fin.data         dataset construction (JNU-IFM, FH-PS-AoP matching, IUGC 2024), splits, loaders
    fin.models       the eleven networks (seven classifiers, four segmenters)
    fin.training     classification and segmentation training
    fin.evaluation   metrics, test-set evaluation, aggregation, external AoP set, split checks
    fin.robustness   corruption benchmark of the classifiers
    fin.efficiency   parameters, operation counts and latency
    fin.reporting    result values, LaTeX tables and figures, the README results table and charts
"""

__version__ = "1.0.0"


class RepositoryNotFoundError(RuntimeError):
    """The repository checkout (configs, splits, results) cannot be located."""


class PaperDirectoryError(RuntimeError):
    """A paper-reporting command was run without a manuscript folder (``--paper-dir`` or ``FIN_PAPER``)."""
