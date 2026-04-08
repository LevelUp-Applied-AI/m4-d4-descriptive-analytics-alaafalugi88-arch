from pathlib import Path
import pandas as pd


def test_drill_file_exists():
    assert Path("drill_eda.py").exists(), "drill_eda.py not found"


def test_script_runs():
    """Import and run the module — should not raise."""
    import importlib
    import sys
    sys.path.insert(0, ".")
    mod = importlib.import_module("drill_eda")


def test_summary_output_exists():
    assert Path("output/summary.csv").exists(), "output/summary.csv not found"


def test_summary_has_expected_stats():
    df = pd.read_csv("output/summary.csv", index_col=0)
    for stat in ["mean", "std", "min", "max"]:
        assert stat in df.index or any(stat in str(idx).lower() for idx in df.index), \
            f"Summary missing statistic: {stat}"


def test_distribution_plot_exists():
    assert Path("output/distributions.png").exists(), "output/distributions.png not found"


def test_correlation_plot_exists():
    assert Path("output/correlation.png").exists(), "output/correlation.png not found"