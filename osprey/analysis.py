"""Functions for analizing osprey results."""

from __future__ import division, print_function, absolute_import

import pandas as pd
import numpy as np

from .trials import Trial


def results_as_dataframe(config, *, project_name=None, project_name_filter=None,
                         expand_parameters=True, trace_summary=True):
    """Turn results into a dataframe

    Parameters
    ----------
    config : Config
        The osprey configuration object
    project_name : str
        Only include the specified project name
    project_name_filter : function
        Apply this function to pick project names
    expand_parameters : bool
        Make each parameter its own column
    trace_summary : bool
        Create a new field 'trace_summary' which summarizes
        any tracebacks

    Returns
    -------
    df : DataFrame
        The DataFrame

    """
    session = config.trials()
    items = [cursor.to_dict() for cursor in session.query(Trial).all()]
    df = pd.DataFrame(items).set_index('id')

    if project_name is not None:
        if project_name_filter is not None:
            raise ValueError("Give either project_name or project_name_filter")

        def project_name_filter(pn):
            return pn == project_name

    if project_name_filter is not None:
        df = df[project_name_filter(df['project_name'].str)]

    if expand_parameters:
        # Expand parameters and then join back in
        expando = pd.DataFrame(list(df['parameters']), index=df.index)
        df = pd.concat((df, expando), axis=1).drop('parameters', 1)

    if trace_summary:
        df['trace_summary'] = [t.splitlines()[-1][:100]
                               if t is not None else "(none)"
                               for t in df['traceback']]

    return df


def summarize(df):
    """Give some stats about the results.

    Parameters
    ----------
    df : DataFrame
        The results DataFrame
    """

    print(df['status'].value_counts())

    if 'trace_summary' in df.columns:
        print(df['trace_summary'].value_counts())


def clean_results(df, *, dropna=True, remove_novar=True, compute_median=True,
                  recompute_mean=False, verbose=True):
    """Sanitize a results DataFrame to make it easier to work with

    Parameters
    ----------
    df : DataFrame
        The results DataFrame
    dropna : bool
        Drop rows with "NA" mean_test_score
    remove_novar : bool
        Remove parameters (columns) that are the same for all results
    compute_median : bool
        Compute median values
    recompute_mean : bool
        Re-compute values from test_scores and train_scores. This will
        save the results in columns "mymean_test_score"
        and "mymean_train_score". I don't know why these are different
        from the precomputed means.
    verbose : bool
        Print informative messages

    Returns
    -------
    cdf : DataFrame
        The cleaned DataFrame

    """

    if dropna:
        cdf = df.dropna(subset=['mean_test_score'])
    else:
        cdf = df.copy()

    if remove_novar:
        for col in cdf.columns:
            if '__' not in col:
                # only remove zero variance parameter columns
                continue
            try:
                if len(cdf[col].unique()) == 1:
                    cdf = cdf.drop(col, 1)
                    if verbose: print("Dropping", col)
            except TypeError as e:
                if verbose: print("Can't process", col, ":", e)

    if compute_median:
        cdf['median_test_score'] = cdf.apply(
            lambda r: np.median(r['test_scores']), axis=1)
        cdf['median_train_score'] = cdf.apply(
            lambda r: np.median(r['train_scores']), axis=1)

    if recompute_mean:
        cdf['mymean_test_score'] = cdf.apply(
            lambda r: np.mean(r['test_scores']), axis=1)
        cdf['mymean_train_score'] = cdf.apply(
            lambda r: np.mean(r['train_scores']), axis=1)


def aggregate(cdf, group_by, *, max_by='mean_test_score', include=None):
    """Aggregate by a column, find max score, and show assoc parameters

    Parameters
    ----------
    cdf : DataFrame
        The cleaned dataframe
    group_by : str
        The field to group by
    max_by : str
        The field to take the max of
    include : list of str
        Columns to also include in the resulting DataFrame

    Returns
    -------
    agged : DataFrame
        Aggregated DataFrame
    """
    dundercols = [cn for cn in cdf.columns if "__" in cn]

    if include is None:
        include = ['mean_test_score', 'mean_train_score',
                   'median_test_score', 'median_train_score']

    if max_by not in (include + dundercols):
        raise ValueError("Must include `max_by` in resulting dataframe")

    agged = cdf.groupby([group_by]).apply(
        lambda j: j.loc[j[max_by].idxmax()]
    )[include + dundercols].sort(max_by)

    return agged


def percentile(cdf, group_by, percent=95, *, sort_by='mean_test_score'):
    """Get percentile of given field.

    Parameters
    ----------
    cdf : DataFrame
        The cleaned DataFrame
    group_by : str
        The field to group by
    percent : float
        The percentile to compute
    sort_by : str
        The field to sort by. This should match whatever you use
        for max_by in ``aggregate()``. Only this column will be returned
    """

    per = cdf.groupby([group_by]).agg(
        lambda x: np.percentile(x, percent)
    )[[sort_by]].sort(sort_by)
    return per