import pandas as pd


def regroup_metadata(metadata: pd.DataFrame, criterion=None, reset_index: bool = True) -> pd.DataFrame:
    """
    Groups the metadata.

    Each group of data (e.g. containing data and label) should have.
    Each group must have its own unique index, where indices are range.
    Each group will be recovered by metadata.loc[index]

    Parameters
    ----------
    metadata : pd.DataFrame
        metadata to use
    criterion : str or List[str]
        The name of the columns based on which the metadata should be categorized
    reset_index : bool, optional
        Whether to reset the level-0 indexing to range. If not given, will reset

    """

    if criterion is None:
        return metadata

    # Group based on the criterion
    metadata = metadata.groupby(criterion).apply(lambda x: x.reset_index(drop=True))

    # Rename the indices to be range
    # Also rename the index level 0 name to be 'index' (instead of criterion)
    if reset_index:
        metadata = metadata.rename(
            index={key: value for value, key in enumerate(metadata.index.get_level_values(0).unique(), start=0)},
            level=0,
        )
        metadata.index.names = [None, *metadata.index.names[1:]]

    return metadata
