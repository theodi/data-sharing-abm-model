import numpy as np


def delete_data(data_deleters, data_held, data_value, tick, alpha):
    """
    Function to delete the data of consumers who have required firms to do so
    Also calculates how much data value is lost to that firm
    """
    data_to_be_del = data_held * data_deleters[None, :, None, :, None]
    data_held -= data_to_be_del
    data_value_lost = (
        data_to_be_del[:tick]
        * np.power(np.exp(-alpha), np.arange(tick)[::-1])[:, None, None, None, None]
    )
    data_value -= data_value_lost.sum(axis=0)
    return data_held, data_value
