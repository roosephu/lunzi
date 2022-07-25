from __future__ import annotations

from collections import defaultdict
from typing import Union
from datetime import datetime
import pickle

import pandas as pd
import numpy as np
from box import Box


class Tracker:
    HEARTBEAT_RATE = 300   # in seconds
    MAX_CACHE_SIZE = 8192  # in bytes

    def __init__(self, filename: str):
        self.io = open(filename, "wb", buffering=self.MAX_CACHE_SIZE)

        self._step = 0  # next available step
        self._cache_time = datetime.now()

    def _should_flush(self) -> bool:
        return (datetime.now() - self._cache_time).total_seconds() >= self.HEARTBEAT_RATE - 1

    async def heartbeat(self):
        import asyncio

        while True:
            await asyncio.sleep(self.HEARTBEAT_RATE)
            self.flush()

    def append(self, commit=True, **kwargs):
        kwargs['_step'] = self._step
        if commit:
            self._step += 1

        data = pickle.dumps(kwargs)
        self.io.write(data)

        if self._should_flush():
            self.flush()

    def flush(self):
        self.io.flush()
        self._cache_time = datetime.now()

    def clean(self, value):
        import torch

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray) and value.size == 1:
            value = value.flatten()[0]
        return value

    def log(self, data, commit=True):
        data = {key: self.clean(value) for key, value in data.items()}
        self.append(log=data, commit=commit)

    def config(self, config, commit=True):
        self.append(config=config, commit=commit)

    def close(self):
        self.flush()
        self.io.close()


def flatten_dict(x, y=None, prefix='', sep='.') -> dict:
    if y is None:
        y = {}

    for key, value in x.items():
        key = f"{prefix}{sep}{key}" if prefix else key
        if isinstance(value, dict):
            flatten_dict(value, y=y, prefix=key, sep=sep)
        else:
            y[key] = value

    return y


class TrackerRun:
    def __init__(self, io: str) -> None:
        history = defaultdict(list)
        self.config = Box()

        with open(io, 'rb') as f:
            while True:
                try:
                    data = pickle.load(f)
                except EOFError:
                    break

                step = data['_step']
                if 'config' in data:
                    self.config.merge_update(data['config'])

                if 'log' in data:
                    for key, value in flatten_dict(data['log']).items():
                        history[key].append({'_step': step, key: value})

        self.history = {key: pd.DataFrame.from_records(value, index='_step') for key, value in history.items()}

    def metrics(self, *keys: str) -> pd.DataFrame:
        return pd.concat([self.history[k] for k in keys], join='inner', axis='columns')

    def keys(self):
        return self.history.keys()

    def __getitem__(self, keys: Union[str, list[str]]) -> pd.DataFrame:
        if isinstance(keys, str):
            return self.history[keys]
        return self.metrics(*keys)

    def __call__(self, *keys: str) -> Union[np.ndarray, list[np.ndarray]]:
        if len(keys) == 1:
            key = keys[0]
            return self.history[key][key].to_numpy()
        metrics = self.metrics(*keys)
        return [metrics[key].to_numpy() for key in keys]

    def plot(self, x: str, y: Union[str, list[str]], label: bool = None, **kwargs):
        if isinstance(y, str):
            y = [y]
        if label is not None:
            label = [label] * len(y)

        df = self[[x] + y]
        df = df.set_index(x)

        df.plot(y=y, label=label, **kwargs)


__all__ = ['Tracker', 'TrackerRun']
