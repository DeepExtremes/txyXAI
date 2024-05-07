import dask.array as da
import numpy as np
import xarray as xr


def _mode(array: xr.Variable, axis: int=0):
    if array.sum() == 0:
        return 0
    mode_dict = {}
    if isinstance(array, da.Array):
        carray = array.compute()
    else:
        carray = array
    for elem in carray:
        if elem == 0:
            continue
        if elem in mode_dict:
            mode_dict[elem] += 1
        else:
            mode_dict[elem] = 1
    max_k = 0
    max_v = 0
    for k, v in mode_dict.items():
        if v > max_v:
            max_k = k
            max_v = v
    return max_k


def aggregate_events(mc: xr.Dataset,
                     aggregate_codes_independently: bool=True,
                     new_cube: bool=True):
    """
    Aggregates events and event labels of a minicube given in a daily temporal
    resolution to the 5-day resolution of the minicube. For each period, the
    most frequent event is selected to determine the label. If there is no
    event, it is set to 0. For the determination of event codes, see
    description of parameter aggregate_codes_independently.

    :param mc: A minicube (or any other xarray dataset) with variables events
    and event_labels in daily temporal resolution, and a time dimension in a
    5-day resolution
    :param aggregate_codes_independently: If True, the values of the events
    array are aggregated as above. Otherwise, It is ensured that the event code
    will correspond to the event, even if the code might not be the most
    frequent in the period.
    :param new_cube: Whether the events shall be written to a new cube
    (default). If False, the existing cube will be updated with the new values.
    :return:
    """
    resampled_event_labels = \
        mc.event_labels.resample(event_time='5D').reduce(_mode)
    missing_steps = len(mc.time) - len(resampled_event_labels)
    resampled_event_label_values = np.concatenate((
        resampled_event_labels.values, np.zeros(missing_steps, dtype=np.uint16)
    ))
    resampled_event_labels = xr.DataArray(
        data=resampled_event_label_values,
        dims='time'
    )
    if aggregate_codes_independently:
        resampled_events = mc.events.resample(event_time='5D').reduce(_mode)
        resampled_event_values = np.concatenate((
            resampled_events.values, np.zeros(missing_steps, dtype=np.uint8)
        ))
        resampled_events = xr.DataArray(
            data=resampled_event_values,
            dims='time'
        )
    else:
        new_event_values = []
        for i in range(0, len(mc.event_time), 5):
            sub_event_values = mc.events.values[i:i + 5]
            event_label_mode = resampled_event_labels[int(i/5)].values
            if event_label_mode == 0:
                event_mode = _mode(sub_event_values)
                new_event_values.append(event_mode)
            else:
                sub_event_label_values = mc.event_labels.values[i:i + 5]
                index = \
                    np.where(sub_event_label_values == event_label_mode)[0][0]
                new_event_values.append(int(sub_event_values[index]))
        new_event_values = np.concatenate((
            new_event_values, np.zeros(missing_steps, dtype=np.uint8)
        ))
        resampled_events = xr.DataArray(
            name='events', data=new_event_values, dims='time'
        )
    if new_cube:
        return xr.Dataset(data_vars={'events': resampled_events,
                                     'event_labels': resampled_event_labels},
                          coords={'time': mc.time})
    mc = mc.assign(variables={'event_labels': resampled_event_labels,
                              'events': resampled_events})
    mc = mc.drop_vars('event_time')
    return mc
