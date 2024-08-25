"""Utils for parsing filenames."""

import math
from typing import Dict, List

import pandas as pd
import parse


def parse_strings(filenames: List[str], patterns: Dict[str, str], no_match_error: bool = True) -> pd.DataFrame:
    """Parses a list of strings in a panda table using multiple patterns.

    For exampple:
    >>> filenames = ["a_1_2.png", "a_2_3.png", "a_3_4.png", "b_1.png", "b_2.png", "b_3.png"]
    >>> pattern = {"a": "a_{x:d}_{y:d}.png", "b": "b_{x:d}.png"}
    >>> dataframes = parse_strings(filenames, pattern)
    >>> print(dataframes["a"])
    >>> print(dataframes["b"])

    Gives
             a
    x y
    1 2  a_1_2.png
    2 3  a_2_3.png
    3 4  a_3_4.png
            b
    x
    1  b_1.png
    2  b_2.png
    3  b_3.png
    """
    compiled_parsers: Dict[str, parse.Parser] = {}
    dimensions: Dict[str, tuple[str]] = {}
    keys_column_types: Dict[str, Dict[str, type]] = {}

    # getting the coord type from format string. parse does not support "s" for strings.
    # https://github.com/r1chardj0n3s/parse/issues/116
    str_to_type: Dict[str, type] = {"": str, "d": int, "f": float}
    columns: Dict[str, List[str]] = {}
    prefix = {}
    postfix = {}
    for name, pattern in patterns.items():
        compiled_parser = parse.compile(pattern)
        compiled_parsers[name] = compiled_parser
        dimensions[name] = tuple(compiled_parser._named_fields)

        for dim_name in dimensions[name]:
            name_type = compiled_parser._name_types[dim_name]
            if name_type == "":
                type_str = ""
            else:
                type_str = parse.extract_format(name_type, {})["type"]
            key_type = str_to_type[type_str]
            if dim_name in keys_column_types:
                if keys_column_types[dim_name] != key_type:
                    raise ValueError(
                        f"Dimension {dim_name} has different types {keys_column_types[dim_name]} and {key_type}"
                    )
            else:
                keys_column_types[dim_name] = str_to_type[type_str]

        columns[name] = list(dimensions[name])
        if name in columns[name]:
            raise ValueError(f"colums {name} already in the keys {dimensions}")
        columns[name].append(name)

        # get the prefix and postfix of the pattern to speed up the parsing
        prefix[name] = pattern.split("{")[0]
        postfix[name] = pattern.split("}")[-1]

    # parse the filenames and add them to the dataframes
    rows = {}
    for name in patterns.keys():
        rows[name] = []

    for filename in filenames:
        parsed = {}
        for name, compiled_parser in compiled_parsers.items():
            if filename.startswith(prefix[name]) and filename.endswith(postfix[name]):
                parsed_ = compiled_parser.parse(filename)
                if parsed_ is not None:
                    parsed[name] = parsed_

        if len(parsed) > 1:
            raise ValueError(f"Filename {filename} matches multiple patterns {parsed.keys()}")
        elif len(parsed) == 0 and no_match_error:
            raise ValueError(f"Filename {filename} does not match any pattern")
        name = list(parsed.keys())[0]
        # add row with the parsed values and the filename
        row = {dim_name: parsed[name].named[dim_name] for dim_name in dimensions[name]}
        row[name] = filename
        rows[name].append(row)

    # create the dataframe with the right columns for each pattern
    dataframes = {}
    for name in patterns.keys():
        df = pd.DataFrame(rows[name], columns=columns[name])
        df.set_index(list(dimensions[name]), inplace=True)
        dataframes[name] = df

    return dataframes


def dataframe_is_sparse(df: pd.DataFrame) -> bool:
    """Check that the dataframe is sparse.

    When a DataFrame with a multi-index does not contain all possible combinations
    of the index levels, it's referred to as a "sparse" multi-index.
    This means that the index is not fully populated with all potential
    combinations of the levels in the multi-index.
    """
    index = df.index
    if len(index.names) == 0:
        return True
    if len(index.names) == 1:
        return True

    num_unique_values_per_index = [len(index.get_level_values(name).unique()) for name in index.names]
    return len(index) < math.prod(num_unique_values_per_index)


def merge_multi_index_dfs(*dfs, how="left"):
    """Merge multiple DataFrames with potentially different multi-index structures.

    Parameters:
    - *dfs: DataFrames to be merged.
    - how: Type of merge to perform ('left', 'right', 'outer', 'inner').

    Returns:
    - Merged DataFrame with multi-index if possible.

    Some values are duplicated over multiple rows when the set of index levels is different.
    """
    if len(dfs) < 2:
        raise ValueError("At least two DataFrames are required for merging.")

    # Reset index for all DataFrames to prepare for merge
    dfs_reset = [df.reset_index() for df in dfs]

    # Identify common index levels across all DataFrames
    common_index_levels = set(dfs_reset[0].columns)
    for df in dfs_reset[1:]:
        common_index_levels &= set(df.columns)

    if not common_index_levels:
        raise ValueError("No common index levels to merge on.")

    # Convert common_index_levels to a list
    common_index_levels = list(common_index_levels)

    # Perform the iterative merge
    merged_df = dfs_reset[0]
    for df in dfs_reset[1:]:
        merged_df = pd.merge(merged_df, df, on=common_index_levels, how=how)

    # Restore multi-index if possible
    # Determine index levels to use based on the original DataFrames
    index_levels = []
    for df in dfs:
        index_levels.extend([col for col in df.index.names if col not in index_levels])

    if index_levels:
        merged_df.set_index(list(index_levels), inplace=True)

    return merged_df


def pivot_dataframe(df, keep_index_cols, fill_value=None):
    """Pivot a DataFrame with a multi-index to keep specified index columns.

    It reates new columns from the remaining index columns.

    Parameters:
    - df: The DataFrame with multi-index.
    - keep_index_cols: List of index columns to keep in the resulting DataFrame.
    - fill_value: Value to fill missing entries with (default is None).

    Returns:
    - A DataFrame with the specified index columns and new columns created from remaining index columns.
    """    
    # Ensure the DataFrame has a multi-index
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("DataFrame does not have a multi-index.")

    # Reset index to turn multi-index into columns
    df_reset = df.reset_index()

    # Determine the columns to pivot
    pivot_cols = [col for col in df_reset.columns if col not in keep_index_cols + [df_reset.columns[-1]]]

    if not pivot_cols:
        raise ValueError("No remaining index columns to pivot on.")

    # Pivot the table
    pivot_table = df_reset.pivot(index=keep_index_cols, columns=pivot_cols, values=df_reset.columns[-1])

    # Flatten the columns
    pivot_table.columns = [f"{df_reset.columns[-1]}_{pivot}_{col}" for pivot in pivot_cols for col in pivot_table.columns]

    # Fill NaN values if a fill_value is provided
    if fill_value is not None:
        pivot_table.fillna(fill_value, inplace=True)

    return pivot_table


def batcher(dataframe, batch_size):
    """Create a generator that yields batches of data from a DataFrame.

    Parameters:
    - dataframe: DataFrame to load data from.
    - batch_size: Number of rows to include in each batch.

    Yields:
    - Batches of data from the DataFrame.
    """
    num_batches = math.ceil(len(dataframe) / batch_size)
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        yield dataframe.iloc[start_idx:end_idx]

def map(dataframe, funcs:Dict[str, callable]):
    """Load data from a DataFrame and apply a set of functions to it.

    Parameters:
    - dataframe: DataFrame to load data from.
    - funcs: Dictionary of functions to apply on each column.

    Returns:
    - A list of results from applying the functions to the data.
    """     
    for name, func in funcs.items():
        results.append(func(batch))
    return results


def visualize(dataframe, funcs:Dict[str, callable]):
    """Load data from a DataFrame and visualize it using a set of functions.

    Parameters:
    - dataframe: DataFrame to load data from.
    - funcs: Dictionary of functions to visualize the data.

    Returns:
    - A list of visualizations of the data.
    """
    for name, func in funcs.items():
        visualizations.append(func(batch))
    return visualizations

def test_parse_strings():
    filenames = ["a_1_2.png", "a_1_3.png", "a_2_3.png", "a_3_4.png", "b_1.png", "b_2.png", "b_3.png"]
    pattern = {"a": "a_{x:d}_{y:d}.png", "b": "b_{x:d}.png"}
    dataframes = parse_strings(filenames, pattern)
    print(dataframes["a"])

    df = dataframes["a"]
    for level_name, level_values in zip(df.index.names, df.index.levels):
        print(f"Level '{level_name}' has data type: {level_values.dtype}")

    merge_multi_index_dfs(dataframes["a"], dataframes["b"])

    pivot_dataframe(dataframes["a"], ["x"], fill_value="")

    merge_multi_index_dfs(pivot_dataframe(dataframes["a"], ["x"], fill_value=""), dataframes["b"])
    assert dataframe_is_sparse(dataframes["a"])
    assert len(dataframes["a"]) == 4
    assert dataframes["a"].loc[(1, 2)]["a"] == "a_1_2.png"
    assert dataframes["a"].loc[(1, 3)]["a"] == "a_1_3.png"
    assert dataframes["a"].loc[(2, 3)]["a"] == "a_2_3.png"
    assert dataframes["a"].loc[(3, 4)]["a"] == "a_3_4.png"
    print(dataframes["b"])

    assert len(dataframes["b"]) == 3
    assert dataframes["b"].loc[1]["b"] == "b_1.png"
    assert dataframes["b"].loc[2]["b"] == "b_2.png"
    assert dataframes["b"].loc[3]["b"] == "b_3.png"
    print("All tests passed")


if __name__ == "__main__":
    test_parse_strings()
