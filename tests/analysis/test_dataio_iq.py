"""Interleaved IQ binary export/import and CSV export round-trips."""
import numpy as np
import pytest

from signal_analysis.dataio import (IQ_DTYPES, MAX_FILE_BYTES, read_iq_binary, read_samples,
                                    write_samples)


@pytest.fixture
def samples():
    rng = np.random.default_rng(1234)
    return (rng.normal(0, 0.25, 4000) + 1j * rng.normal(0, 0.25, 4000)).astype(np.complex64)


@pytest.mark.parametrize("dtype", IQ_DTYPES)
@pytest.mark.parametrize("endian", ["little", "big"])
def test_iq_binary_round_trip(tmp_path, samples, dtype, endian):
    path = tmp_path / f"iq_{dtype}_{endian}.bin"
    write_samples(path, samples, "iq16" if dtype == "int16" else "iq32", endian=endian)
    assert path.stat().st_size == samples.size * (2 if dtype == "int16" else 4) * 2
    back = read_iq_binary(str(path), dtype, endian)
    if dtype == "int16":
        np.testing.assert_allclose(back, samples, atol=1.5 / 32768, rtol=0.05)
    else:
        np.testing.assert_array_equal(back, samples)


def test_read_samples_dispatches_binary(tmp_path, samples):
    path = tmp_path / "iq.bin"
    write_samples(path, samples, "iq16")
    back = read_samples(str(path), binary_dtype="int16")
    np.testing.assert_allclose(back, samples, atol=1.5 / 32768, rtol=0.05)


def test_read_samples_requires_dtype_for_binary(tmp_path, samples):
    path = tmp_path / "iq.raw"
    write_samples(path, samples, "iq16")
    with pytest.raises(ValueError, match="binary_dtype"):
        read_samples(str(path))


def test_truncated_binary_rejected(tmp_path, samples):
    path = tmp_path / "bad.bin"
    write_samples(path, samples, "iq16")
    data = path.read_bytes()
    path.write_bytes(data[:-1])
    with pytest.raises(ValueError, match="截断"):
        read_iq_binary(str(path), "int16", "little")


def test_int16_scaling_and_clipping(tmp_path):
    huge = np.array([1.0 + 0j, -1.0 + 0j, 2.0 + 0j, -2.0 + 0j, 0.5 - 0.5j], dtype=np.complex64)
    path = tmp_path / "clip.bin"
    write_samples(path, huge, "iq16")
    raw = np.frombuffer(path.read_bytes(), dtype="<i2")
    assert list(raw[0::2]) == [32767, -32767, 32767, -32767, 16384]
    assert list(raw[1::2]) == [0, 0, 0, 0, -16384]
    back = read_iq_binary(str(path), "int16", "little")
    np.testing.assert_allclose(back[:4], [1, -1, 1, -1], atol=1 / 32768)
    np.testing.assert_allclose(back[4], 0.5 - 0.5j, atol=1 / 32768)


def test_csv_round_trip(tmp_path, samples):
    path = tmp_path / "iq.csv"
    write_samples(path, samples, "csv")
    lines = path.read_text().splitlines()
    assert len(lines) == samples.size
    assert len(lines[0].split(",")) == 2
    back = read_samples(str(path))
    np.testing.assert_allclose(back, samples, atol=1e-6)


def test_npy_round_trip(tmp_path, samples):
    path = tmp_path / "iq.npy"
    write_samples(path, samples, "npy")
    back = read_samples(str(path))
    np.testing.assert_array_equal(back, samples.astype(np.complex64))


def test_unknown_format_rejected(tmp_path, samples):
    with pytest.raises(ValueError):
        write_samples(tmp_path / "iq.bin", samples, "sigmf")


def test_endian_mismatch_produces_garbage_not_equality(tmp_path, samples):
    path = tmp_path / "iq.bin"
    write_samples(path, samples, "iq32", endian="big")
    little = read_iq_binary(str(path), "float32", "little")
    assert not np.allclose(little, samples, rtol=1e-6)


def test_file_size_cap_constant():
    assert MAX_FILE_BYTES == 512 * 1024 * 1024
