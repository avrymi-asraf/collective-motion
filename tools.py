import timeit
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

import pandas as pd
import numpy as np

from typing import Tuple, Callable, Union, List, Optional, Any

Opt = Optional
NumType = Union[int, float,torch.Tensor]
TensorType = torch.Tensor
BoxDimType = Tuple[NumType, NumType]
RangeType = Tuple[NumType, NumType]
LocType = Tuple[NumType, NumType]
IndScalarType = Union[int, TensorType]
UpdateFuncType = Callable[[TensorType, dict], TensorType]
IndexsType = Union[TensorType, List[IndScalarType], Tuple[IndScalarType, ...], range]
OptimazerType = Union[optim.Optimizer, optim.Adam, optim.SGD, optim.RMSprop]
LossType = Union[nn.Module, nn.CrossEntropyLoss, nn.MSELoss, nn.BCELoss]


def locatoin_and_angel_to_line(
    data: TensorType, speed: bool = True, size: float = 1.0
) -> TensorType:
    """
    Get the location and angle of items, and return x,y so every couple of numbers is a line. between the lines, there is a nan to space them out.

    Args:
        data (TensorType): tensor of shape (m,4) representing x,y, speed, acceleration, angel. even though speed are not used in this function.
        speed (bool, optional): if calculate the speed. Defaults to True.
        size (float, optional): the size of the lines. Defaults to 1.0.

    Returns:
        TensorType: tensor of shape (m,2) representing x,y of the lines. where every couple of numbers is a line and between the lines, there is a nan.
    """
    x, y, s, angel = data.T
    if not speed:
        s = 1
    new_x = torch.column_stack(
        [x, size * s * torch.cos(angel) + x, torch.nan * torch.empty_like(x)]
    ).flatten()
    new_y = torch.column_stack(
        [y, size * s * torch.sin(angel) + y, torch.nan * torch.empty_like(y)]
    ).flatten()
    return torch.column_stack((new_x, new_y)).T


def add_parameters(data: TensorType) -> TensorType:
    """
    for each row in the data, add parameters, speed, acceleration, and angle.

    Args:
        data (TensorType): tensor (N,t,2) where N is several particles, t is time steps and 2 is x,y location in the plane.

    Returns:
        TensorType: tensor (N,t-2,4) where N is the number of particles, speed, angle.
    """
    v = (data[2:] - data[:-2]) / 2  # speed by x and y
    # dx, dy = data[:-1] - data[1:]
    theta = (
        ## Calculate the angle of the vector, and convert it to a degree, and remove 90 degrees so front is 0
        (torch.atan2(v[:, :, 1].reshape(-1), v[:, :, 0].reshape(-1)))
    ).reshape(v.shape[0], v.shape[1], 1)
    return torch.cat(
        [
            data[1:-1],
            torch.linalg.norm(v, dim=-1).unsqueeze(-1),
            theta,
        ],
        dim=-1,
    )


def plot_timeline_with_direction(
    data: TensorType,
    title: str,
    figsize: BoxDimType = (1200, 1200),
    steps: Opt[IndexsType] = None,
    items: Opt[IndexsType] = None,
    duration: int = 1000,
    **kwargs,
) -> go.Figure:
    """
    plot timeline of data with direction

    Args:
        data (TensorType): tensor of shape (t,m,4) where t is the number of time steps and m is the number of objects,
        and the last dim is (x,y, speed, acceleration, angle)
        title (str): title of the plot
        figsize (BoxDimType, optional): figure size. Defaults to (10,10).
        items (Opt[IndexType], optional): index for items to plot, if not assigned plot all. Defaults to None.
        **kwargs: kwargs for go.Layout
    Returns:
        go.Figure: animation figure of timeline
    """

    def xy_as_dict(data, **kwargs):
        x, y = locatoin_and_angel_to_line(data, **kwargs)
        return dict(x=x, y=y)

    # get axis size
    max_range = max([data[:, :, 0].max().item(), data[:, :, 1].max().item()])
    items = range(data.shape[1]) if items is None else items
    steps = range(data.shape[0]) if steps is None else steps
    fig = go.Figure(
        data=[
            go.Scatter(
                xy_as_dict(data[steps[0], items], size=3),
                mode="lines",
            ),
            go.Scatter(
                x=data[steps[0], items, 0], y=data[steps[0], items, 1], mode="markers"
            ),
        ],
        layout=go.Layout(
            xaxis=dict(range=[0, max_range], constrain="domain"),
            yaxis=dict(scaleanchor="x", scaleratio=1),
            title=title,
            width=figsize[0],
            height=figsize[1],
            **kwargs,
        ),
    ).update_layout(
        updatemenus=[
            dict(
                type="buttons",
                buttons=[dict(label="Play", method="animate", args=[None])],
            )
        ],
    )

    frames = [
        go.Frame(
            data=[
                go.Scatter(xy_as_dict(data[t, items], size=3), mode="lines"),
                go.Scatter(x=data[t, items, 0], y=data[t, items, 1], mode="markers"),
            ],
        )
        for t in steps
    ]
    fig.frames = frames
    return fig


def get_index_neighbors(data: TensorType, radius: float) -> TensorType:
    """
    get the indexes of the neighbors of each data point

    Args:
        radius (float): radios of the neighbors
        data (TensorType): tensor of shape (N,2) of x,y

    Returns:
        TensorType: tensor of shape (N, N) of the indexes of the neighbors,
        so data[res[i]] is the neighbors of data[i]
    """
    # For each data point, get the distance from any other data point
    dis_mat = torch.stack(
        [torch.linalg.norm(data[i, :] - data[:, :], dim=1) for i in range(len(data))]
    )
    # For each data point, get the indexes of the data points that are in the ratio
    indexes = torch.stack([dis_mat[i] < radius for i in range(len(data))]) ^ torch.eye(
        len(data), dtype=torch.bool
    )
    return indexes


def create_timeline_series(
    data: TensorType, f: UpdateFuncType, steps: int, *arg, **kargs
) -> TensorType:
    """
    create a timeline for data,

    Args:
        data (TensorType): input data, tensor (t, N,d)
        f (UpdateFuncType): function to update data, function :(N,d)->(N,d)
        steps (int): num of steps to advance
        *arg: args for f
        **kargs: kargs for f

    Returns:
        TensorType: tensor (t+steps,N,d), so ret[t] is represent the data int time t
    """
    if steps < 0:
        raise ValueError("steps must be grow then 0")
    t, N, d = data.shape
    res = torch.empty(t + steps, N, d)
    res[:t] = data.clone()
    for step in range(t, t + steps):
        res[step] = f(res[step - 1], *arg, **kargs)
    return res


def evaluat_in_indexes(
    data: TensorType, indexes: IndexsType, func: UpdateFuncType, *args, **karags
) -> TensorType:
    """
    evaluate func in indexes points and return the resuls

    Args:
        data (TensorType): tensor (N,d) of the data
        indexes (IndexsType): indexes of the points to evaluate, can be range, tuple,list or tensor 1D
        func (UpdateFuncType): function to evaluate, function :(N,d)->(N,d)
        *args: args for func
        **karags: kargs for func
    Returns:
        TensorType: tensor (len(indexes),d) of the results
    """
    return torch.stack([func(data[i], *args, **karags) for i in indexes])


def normelize_data(
    data: TensorType,
    loc_range: RangeType = (-5, 5),
    speed_range: RangeType = (0, 10),
    max_loc: Opt[NumType] = None,
    max_speed: Opt[NumType] = None,
) -> TensorType:
    """normelize the data, by default the data will be normelized to (-5, 5) and (0, 10)

    Args:
        data (TensorType): tensor (t,N,4) where t is the time, N is the number of agents and 4 is the x,y,speed,angel.
        loc_range (RangeType, optional): range of loctoin, It must be square. Defaults to (-5, 5).
        speed_range (RangeType, optional): range of speed. Defaults to (0, 10).
        max_loc (Opt[LocType], optional): give max locatoin to normlize by it, if not given take max from data. Defaults to None.
        max_speed (Opt[int], optional): give max speed to normlize by it,if not given take max from data. Defaults to None.
    """
    if max_loc is None:
        max_loc = data[:, :, :2].abs().max()
    if max_speed is None:
        max_speed = data[:, :, :2].max()
    ret = torch.empty_like(data)
    ret[:, :, :2] = (data[:, :, :2] / max_loc) * (
        loc_range[1] - loc_range[0]
    ) - loc_range[0]
    ret[:, :, 2] = (data[:, :, 2] / max_speed) * (speed_range[1] - speed_range[0])
    ret[:, :, 3] = data[:, :, 3]
    return ret