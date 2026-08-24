# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for src/rocprof_compute_analyze/analysis_base.py."""

import argparse
import gzip
from pathlib import Path
from types import SimpleNamespace

import common
import pandas as pd
import pytest

from rocprof_compute_analyze.analysis_base import OmniAnalyze_Base
from utils import schema
from utils.specs import MachineSpecsCDNA

MODULE = "rocprof_compute_analyze.analysis_base"

# Trimmed MI350 sysinfo, mirroring tests/workloads/vcopy/MI350/sysinfo.csv.
MI350_SYSINFO = {
    "ip_blocks": "SQ|LDS|SQC|TA|TD|TCP|TCC|SPI|CPC|CPF|roofline",
    "version": "3",
    "compute_partition": "SPX",
    "memory_partition": "NPS1",
    "gpu_series": "MI350",
    "gpu_model": "MI350",
    "gpu_arch": "gfx950",
    "cu_per_gpu": "256",
    "total_l2_chan": "128",
    "num_xcd": "8",
}


def _write_results_gz(path: Path, content: str) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(content)


def test_concat_result_csvs_concatenates_rocpd_results(tmp_path, monkeypatch) -> None:
    """Concatenates rocpd long-form results_*.csv.gz into one pmc_perf.csv."""
    common.patch_console(monkeypatch, MODULE, "debug", "warning")

    header = "GPU_ID,Kernel_Name,Counter_Name,Counter_Value\n"
    _write_results_gz(
        tmp_path / "results_pmc_perf_0.csv.gz",
        header + "0,kernel_a,SQ_WAVES,10\n0,kernel_a,SQ_WAVES,20\n",
    )
    _write_results_gz(
        tmp_path / "results_pmc_perf_1.csv.gz",
        header + "0,kernel_a,SQ_BUSY_CYCLES,30\n",
    )

    inst = OmniAnalyze_Base.__new__(OmniAnalyze_Base)
    inst.concat_result_csvs(
        sorted(tmp_path.glob("results_*.csv.gz")), tmp_path / "pmc_perf.csv"
    )
    merged = pd.read_csv(tmp_path / "pmc_perf.csv")

    assert list(merged.columns) == [
        "GPU_ID",
        "Kernel_Name",
        "Counter_Name",
        "Counter_Value",
    ]
    assert len(merged) == 3
    assert set(merged["Counter_Name"]) == {"SQ_WAVES", "SQ_BUSY_CYCLES"}
    assert sorted(merged["Counter_Value"].tolist()) == [10, 20, 30]


def test_concat_result_csvs_skips_empty_and_errors_when_all_empty(
    tmp_path, monkeypatch
) -> None:
    mocks = common.patch_console(monkeypatch, MODULE, "debug", "warning")
    (tmp_path / "results_pmc_perf_0.csv.gz").write_bytes(b"")
    (tmp_path / "results_pmc_perf_1.csv.gz").write_bytes(b"")

    inst = OmniAnalyze_Base.__new__(OmniAnalyze_Base)
    with pytest.raises(SystemExit):
        inst.concat_result_csvs(
            sorted(tmp_path.glob("results_*.csv.gz")),
            tmp_path / "pmc_perf.csv",
        )

    assert not (tmp_path / "pmc_perf.csv").exists()
    skipped = [
        call.args[0]
        for call in mocks["warning"].call_args_list
        if "Skipping empty" in str(call.args[0])
    ]
    assert len(skipped) == 2


def test_concat_result_csvs_skips_zero_byte_compressed_pass(
    tmp_path, monkeypatch
) -> None:
    common.patch_console(monkeypatch, MODULE, "debug", "warning")
    header = "GPU_ID,Kernel_Name,Counter_Name,Counter_Value\n"
    _write_results_gz(
        tmp_path / "results_pmc_perf_0.csv.gz",
        header + "0,kernel_a,SQ_WAVES,10\n",
    )
    (tmp_path / "results_pmc_perf_1.csv.gz").write_bytes(b"")

    inst = OmniAnalyze_Base.__new__(OmniAnalyze_Base)
    inst.concat_result_csvs(
        sorted(tmp_path.glob("results_*.csv.gz")), tmp_path / "pmc_perf.csv"
    )

    assert pd.read_csv(tmp_path / "pmc_perf.csv")["Counter_Value"].tolist() == [10]


def test_join_workload_csvs_finds_compressed_results(tmp_path, monkeypatch) -> None:
    """join_workload_csvs picks up compressed results_*.csv.gz artifacts."""
    common.patch_console(monkeypatch, MODULE, "debug", "warning", "log")

    header = "GPU_ID,Kernel_Name,Counter_Name,Counter_Value\n"
    _write_results_gz(
        tmp_path / "results_pmc_perf_0.csv.gz",
        header + "0,kernel_a,SQ_WAVES,10\n",
    )

    inst = OmniAnalyze_Base.__new__(OmniAnalyze_Base)
    inst.join_workload_csvs(tmp_path)

    assert pd.read_csv(tmp_path / "pmc_perf.csv")["Counter_Value"].tolist() == [10]


def test_concat_result_csvs_errors_on_truncated_compressed_results(
    tmp_path, monkeypatch
) -> None:
    """Partial .csv.gz from a killed profile run must not leave pmc_perf.csv behind."""
    common.patch_console(monkeypatch, MODULE, "debug", "warning")
    header = "GPU_ID,Kernel_Name,Counter_Name,Counter_Value\n"
    rows = "".join(f"0,kernel_a,SQ_WAVES,{i}\n" for i in range(2000))
    whole = gzip.compress((header + rows).encode("utf-8"))
    (tmp_path / "results_pmc_perf_0.csv.gz").write_bytes(whole[: len(whole) // 2])

    inst = OmniAnalyze_Base.__new__(OmniAnalyze_Base)
    with pytest.raises(SystemExit):
        inst.concat_result_csvs(
            sorted(tmp_path.glob("results_*.csv.gz")),
            tmp_path / "pmc_perf.csv",
        )

    assert not (tmp_path / "pmc_perf.csv").exists()


def test_concat_result_csvs_errors_when_only_headers(tmp_path, monkeypatch) -> None:
    """Header-only results files must not leave a reusable pmc_perf.csv behind."""
    common.patch_console(monkeypatch, MODULE, "debug", "warning")
    header = "GPU_ID,Kernel_Name,Counter_Name,Counter_Value\n"
    _write_results_gz(tmp_path / "results_pmc_perf_0.csv.gz", header)
    _write_results_gz(tmp_path / "results_pmc_perf_1.csv.gz", header)

    inst = OmniAnalyze_Base.__new__(OmniAnalyze_Base)
    with pytest.raises(SystemExit):
        inst.concat_result_csvs(
            sorted(tmp_path.glob("results_*.csv.gz")),
            tmp_path / "pmc_perf.csv",
        )

    assert not (tmp_path / "pmc_perf.csv").exists()


def test_concat_result_csvs_rejects_wide_legacy_results(tmp_path, monkeypatch) -> None:
    """Wide legacy results_*.csv without Counter_Name are rejected."""
    common.patch_console(monkeypatch, MODULE, "debug", "warning")
    _write_results_gz(
        tmp_path / "results_pmc_perf_0.csv.gz",
        "GPU_ID,Kernel_Name,Dispatch_ID,SQ_WAVES\n0,kernel_a,0,10\n",
    )

    inst = OmniAnalyze_Base.__new__(OmniAnalyze_Base)
    with pytest.raises(SystemExit):
        inst.concat_result_csvs(
            sorted(tmp_path.glob("results_*.csv.gz")),
            tmp_path / "pmc_perf.csv",
        )

    assert not (tmp_path / "pmc_perf.csv").exists()


def test_sanitize_rejects_paths_sharing_a_workload_name(tmp_path, monkeypatch) -> None:
    """Reject two paths whose last two components match."""
    mock_error = common.patch_console(monkeypatch, MODULE, "error")["error"]
    paths = [[str(tmp_path / parent / "vcopy" / "MI300")] for parent in ("a", "b")]
    for path in paths:
        Path(path[0]).mkdir(parents=True)

    # The mock records instead of exiting, so sanitize runs on to a later error.
    with pytest.raises(SystemExit):
        OmniAnalyze_Base(argparse.Namespace(tui=False, path=paths), {}).sanitize()

    assert "last two components" in mock_error.call_args.args[1]


def _specs_correction_analyzer(tmp_path, monkeypatch, specs_correction):
    """Analyzer wired to a minimal MI350 workload, ready for initalize_runs().

    Panel config generation is stubbed out so the run loop can be exercised
    without loading the real arch YAML.
    """
    # MI350_SYSINFO is trimmed, so silence the "missing specs" warnings the
    # spec-to-frame conversion emits for the fields it leaves unset.
    common.patch_console(monkeypatch, "utils.specs", "warning")
    pd.DataFrame([MI350_SYSINFO]).to_csv(tmp_path / "sysinfo.csv", index=False)

    args = argparse.Namespace(
        path=[[str(tmp_path)]],
        specs_correction=specs_correction,
        no_roof=True,
        normal_unit="per_kernel",
        gpu_kernel=None,
        list_stats=False,
        filter_metrics=None,
        tui=False,
        config_dir=str(tmp_path),
    )
    inst = OmniAnalyze_Base(args, {})
    monkeypatch.setattr(inst, "generate_configs", lambda *a, **kw: {})
    inst._arch_configs = {MI350_SYSINFO["gpu_arch"]: schema.ArchConfig()}
    inst.set_soc({
        MI350_SYSINFO["gpu_arch"]: SimpleNamespace(
            _mspec=MachineSpecsCDNA(**MI350_SYSINFO)
        )
    })
    return inst


def test_initalize_runs_applies_specs_correction(tmp_path, monkeypatch) -> None:
    """--specs-correction overrides the specs analysis runs on."""
    inst = _specs_correction_analyzer(tmp_path, monkeypatch, "num_xcd:4,cu_per_gpu:64")

    workload = inst.initalize_runs()[str(tmp_path)]

    assert workload.sys_info["num_xcd"].item() == "4"
    assert workload.sys_info["cu_per_gpu"].item() == "64"
    # Uncorrected specs come through untouched, including the ip_blocks entry
    # initalize_runs() reads right after applying the correction.
    assert workload.sys_info["gpu_arch"].item() == "gfx950"
    assert workload.sys_info["total_l2_chan"].item() == "128"
    assert "SQ" in workload.avail_ips
    assert "roofline" in workload.avail_ips


def test_initalize_runs_without_specs_correction_keeps_sysinfo(
    tmp_path, monkeypatch
) -> None:
    """Without the option the workload keeps the recorded sysinfo values."""
    inst = _specs_correction_analyzer(tmp_path, monkeypatch, None)

    workload = inst.initalize_runs()[str(tmp_path)]

    assert workload.sys_info["num_xcd"].item() == 8
    assert workload.sys_info["cu_per_gpu"].item() == 256
