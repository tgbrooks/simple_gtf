import pathlib
import polars as pl


def read_gtf(gtf_path: str | pathlib.Path) -> pl.DataFrame:
    """
    Load a polars dataframe from a gtf or gtf.gz file pathlib

    The returned dataframe has the following columns:
        seqname - name of the chromosome or scaffold; chromosome names can be given with or without the 'chr' prefix.
        source - name of the program that generated this feature, or the data source (database or project name)
        feature - feature type name, e.g. gene, exon, trancript, CDS, five_prime_utr, ...
        start - Start position* of the feature, with sequence numbering starting at 1.
        end - End position* of the feature, with sequence numbering starting at 1.
        score - A floating point value.
        strand - defined as + (forward) or - (reverse).
        frame - One of '0', '1' or '2'. '0' indicates that the first base of the feature is the first base of a codon, '1' that the second base is the first base of a codon, and so on..
    as well as one column per attribute.

    Attributes vary from GTF file source but from Ensembl expect to have:
        gene_id
        gene_name
        gene_biotype
        transcript_id
        ccds_id
        exon_id
        exon_number
        gene_source
        havana_gene
        havana_gene_version
        transcript_source
        transcript_biotype
        havana_transcript
        havana_transcript_version
        transcript_support_level
        protein_id
        tag
        gene_version
        transcript_version
        exon_version
        protein_version

    Attribute columns have dtype of lists of strings. This is true even for columns like gene_id that typically have only one entry.
    To make use of these, the explode() command is useful. For example, to get the mapping of transcript ids to gene ids:

        gtf = read_gtf("example.gtf.gz")
        gtf.select("transcript_id", "gene_id").explode("transcript_id").explode("gene_id").drop_nulls()
    """

    gtf_columns = [
        "seqname",
        "source",
        "feature",
        "start",
        "end",
        "score",
        "strand",
        "frame",
        "attribute",
    ]
    gtf_contents = pl.scan_csv(
        gtf_path,
        separator="\t",
        has_header=False,
        new_columns=gtf_columns,
        comment_prefix="#",
        schema_overrides={
            "seqname": pl.Categorical,
            "source": pl.Categorical,
            "feature": pl.Categorical,
            "start": pl.Int64,
            "end": pl.Int64,
            "score": pl.Float64,
            "strand": pl.Categorical,
            "frame": pl.Categorical,
            "attribute": pl.Utf8,
        },
        null_values=".",
    )

    attributes = (
        gtf_contents
        # .lazy()
        .select("attribute")
        .with_columns(
            pl.col("attribute")
            .str.strip_suffix(";")
            .str.split("; ")
            .list.eval(
                pl.element().str.extract_groups(
                    r"(?<attr_name>\w+) \"(?<attr_val>\w+)\""
                )
            )
        )
        .collect()
    )

    features = gtf_contents.select(
        "seqname",
        "source",
        "feature",
        "start",
        "end",
        "score",
        "strand",
        "frame",
    )

    all_attributes = (
        attributes.lazy()
        .select(
            pl.col("attribute")
            .list.eval(pl.element().struct.field("attr_name"))
            .alias("attr_name"),
        )
        .explode("attr_name")
        .unique()
        .collect()
    ).drop_nulls()["attr_name"]

    wide_attributes = attributes.select(
        *[
            pl.col("attribute")
            .list.eval(
                pl.when(pl.element().struct.field("attr_name") == attr_name)
                .then(pl.element().struct.field("attr_val"))
                .otherwise(None)
            )
            .list.drop_nulls()
            .alias(attr_name)
            for attr_name in all_attributes
        ]
    )

    gtf = pl.concat(
        [
            features.collect(),
            wide_attributes,
        ],
        how="horizontal",
    )
    return gtf
