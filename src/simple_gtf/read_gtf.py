import pathlib
import polars as pl

attribute_types = {
    "exon_number": pl.Int32,
    "gene_version": pl.Int32,
    "transcript_support_level": pl.Categorical,
    "transcript_biotype": pl.Categorical,
    "gene_source": pl.Categorical,
    "tag": pl.Categorical,
    "exon_version": pl.Int32,
    "transcript_version": pl.Int32,
    "gene_biotype": pl.Categorical,
    "transcript_source": pl.Categorical,
    "protein_version": pl.Int32,
}


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

    Attribute columns have dtype of lists. This is true even for columns like gene_id that typically have only one entry.
    To make use of these, the explode() command is useful. For example, to get the mapping of transcript ids to gene ids:

        gtf = read_gtf("example.gtf.gz")
        gtf.select("transcript_id", "gene_id").explode("transcript_id").explode("gene_id").drop_nulls().unique()
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
    gtf_contents = pl.read_csv(
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

    # Process in batches for reduced memory
    N = 100_000
    temp = []
    for i in range(0, gtf_contents.height, N):
        # Parse the attribute columns into separate name:value pairs
        # this column is a list of values like:
        # name "value"; name2 "value2";
        temp.append(
            gtf_contents[i : i + N]
            .lazy()
            .select(
                pl.col("attribute")
                .str.strip_suffix(";")
                .str.split("; ")
                .list.eval(
                    pl.element().str.extract_groups(
                        r"(?<attr_name>\w+) \"(?<attr_val>\w+)\""
                    )
                )
            )
            .select(
                pl.col("attribute").cast(
                    pl.List(
                        pl.Struct({"attr_name": pl.Categorical, "attr_val": pl.String})
                    )
                )
            )
            .collect()
        )
    attributes = pl.concat(temp)

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

    # List all attribute names present
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

    # Convert attributes from list-of-(name:val) to one column per name (each of dtype list of values)
    attr_columns = []
    for attr_name in all_attributes:
        this_col = attributes.select(
            pl.col("attribute")
            # .list.filter(pl.element().struct.field("attr_name") == attr_name) # weirdly slow, so we use the map-dropnulls approach
            .list.eval(
                pl.when(pl.element().struct.field("attr_name") == attr_name)
                .then(pl.element().struct.field("attr_val"))
                .otherwise(None)
            )
            .list.drop_nulls()
            .alias(attr_name)
        )

        if attr_name in attribute_types:
            # Some columns have an expected type and so we try to cast those
            try:
                this_col = this_col.cast(
                    pl.List(attribute_types.get(attr_name, pl.String))
                )  # Cast to the expected type
            except pl.exceptions.InvalidOperationError:
                # Couldn't convert a column as expected, just leave it as default (string)
                pass
        attr_columns.append(this_col)

    # Aggregate all feature data and attributes together
    gtf = pl.concat(
        [
            features,
            *attr_columns,
        ],
        how="horizontal",
    )
    return gtf
