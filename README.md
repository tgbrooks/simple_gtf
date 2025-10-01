# simple-gtf

Simple GTF reader outputting as a Polars dataframe.

GTF files are notoriously inconsistent so this may not work with all.
It has been primarily tested with Ensembl and FlyBase GTF files.
This repository is primarily intended for personal use and no guarantees are made about its helpfulness for others.

# Installation

```
pip install git+https://github.com/tgbrooks/simple_gtf
# or, if using uv:
uv add "simple_gtf @ git+https://github.com/tgbrooks/simple_gtf"
```

# Use

```
import simple_gtf

# Read in a gtf file
gtf = simple_gtf.read_gtf("example.gtf.gz")

# For example, get transcript-gene pairs as a polars dataframe
gtf.select("transcript_id", "gene_id").explode("transcript_id").explode("gene_id").drop_nulls()
```

# Alternatives

[gffutils](https://daler.github.io/gffutils/) is much more full-featured and persists its parsing into a sqlite file, which makes subsequent access very fast.
However, it parses attributes as json entries in the table instead of as separate columns, which makes attributes less convenient.
