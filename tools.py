import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import timeit
import numpy as np
from typing import Tuple


BoxDimType = Tuple[float, float]
TensorType = torch.Tensor


def spherical_to_cartesian(
    r: TensorType, theta: TensorType, phi: TensorType, normalize=True
) -> TensorType:
    """return the cartesian coordinates of the input spherical coordinates

    Args:
        r (TensorType): tensor (N, ) of r coordinates
        theta (TensorType): tensor (N, ) of theta coordinates, angle should be in degree (0,180)
        phi (TensorType): tensor (N, ) of phi coordinates, angle should be in degree (0,360)
        normalize (bool, optional): if need to normalize the value. Defaults to True.

    Returns:
        TensorType: tensor (N, 3) of cartesian coordinates
    """ """"""
    out = torch.stack(
        [
            r * torch.sin(torch.deg2rad(theta)) * torch.cos(np.deg2rad(phi)),
            r * torch.sin(torch.deg2rad(theta)) * torch.sin(torch.deg2rad(phi)),
            r * torch.cos(torch.deg2rad(theta)),
        ]
    )
    if normalize:
        return out / torch.linalg.norm(out)
    return out


def cartesian_to_spherical(
    x: TensorType, y: TensorType, z: TensorType
) -> Tuple[TensorType, TensorType, TensorType]:
    """return the spherical coordinates of the input cartesian coordinates

    Args:
        x (TensorType): tensor (N, ) of x coordinates
        y (TensorType): tensor (N, ) of y coordinates
        z (TensorType): tensor (N, ) of z coordinates

    Returns:
        Tuple[TensorType, TensorType, TensorType]: r, theta, phi of the input cartesian coordinates
    """
    r = torch.linalg.norm([x, y, z])
    theta = torch.rad2deg(torch.arccos(z / r))
    phi = torch.rad2deg(torch.arctan2(y, x))
    return r, theta, phi


def cone_by_spherical(
    x: TensorType,
    y: TensorType,
    z: TensorType,
    theta: TensorType,
    phi: TensorType,
    r: float = 1,
) -> go.Cone:
    """return a plotly cone object by the spherical coordinates, which is a tensor of shape (N, 5) with
    the first three columns as the x, y, z coordinates, and the last two columns as the theta, phi angles.
    r is the radius of the cone.

    Args:
        x (TensorType): tensor (N, ) of x coordinates
        y (TensorType): tensor (N, ) of y coordinates
        z (TensorType): tensor (N, ) of z coordinates
        theta (TensorType): tensor (N, ) of theta angles, angle should be in degree (0,180)
        phi (TensorType): tensor (N, ) of phi angles, angle should be in degree (0,360)
        r (float, optional): radius of cone. Defaults to 1.

    Returns:
        go.Cone: _description_
    """
    u, v, w = spherical_to_cartesian(r, theta, phi, normalize=False)
    return go.Cone(x=x, y=y, z=z, u=u, v=v, w=w, anchor="tail", sizemode="absolute")


def show_cordinte(
    data_for_fig: TensorType, box_dim: BoxDimType = (-5, 5), nticks: int = 4
) -> go.Figure:
    """return a plotly figure of the data, which is a tensor of shape (N, 5) with
    the first three columns as the x, y, z coordinates, and the last two columns as the theta, phi angles.

    Args:
        data_for_fig (Tensor): tensor of shape (N, 5)
        box_dim (BoxDimType, optional): dim box to show. Defaults to (-5, 5).
        nticks (int, optional): num of ticks . Defaults to 4.

    Returns:
        go.Figur: 3d plotly figure of the data
    """
    return go.Figure(data=cone_by_spherical(*data_for_fig.T)).update_layout(
        scene=dict(
            aspectratio=dict(x=1, y=1, z=0.8),
            camera_eye=dict(x=1.2, y=1.2, z=0.6),
            xaxis=dict(
                nticks=nticks,
                range=box_dim,
            ),
            yaxis=dict(
                nticks=nticks,
                range=box_dim,
            ),
            zaxis=dict(
                nticks=nticks,
                range=box_dim,
            ),
        ),
        coloraxis_showscale=False,
    )
