from __future__ import division
import pandas


def total_power_demand(project):
    """Sum the power demand of each location and return a data frame with the result

    Args:
        project (Project): a project

    Return:
        a pandas DataFrame with the power demand at each location
    """
    result = pandas.DataFrame()
    for location in project.locations:
        temp = location.result.copy()
        temp = temp.rename(columns={'power_demand': str(location.category) + '_demand'})
        result = pandas.concat([result, temp[str(location.category) + '_demand']], axis=1)

    result['total'] = result.sum(axis=1)
    del temp
    return result
