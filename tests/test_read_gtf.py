import polars as pl
import simple_gtf


def test_read_gtf(resource_path_root):
    gtf_path = resource_path_root / "Mus_musculus.small.gtf.gz"
    expected = resource_path_root / "Mus_musculus.expected.parquet"
    gtf = simple_gtf.read_gtf(gtf_path)
    expected = pl.read_parquet(expected)

    # NOTE: column ordering is unpredictable, so we normalize
    assert set(gtf.columns) == set(expected.columns)

    expected = expected.select(gtf.columns)
    assert gtf.sort("feature_id").equals(expected)
