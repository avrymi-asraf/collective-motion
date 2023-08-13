from tools import *


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
    out = torch.column_stack(
        [
            r * torch.sin(torch.deg2rad(theta)) * torch.cos(torch.deg2rad(phi)),
            r * torch.sin(torch.deg2rad(theta)) * torch.sin(torch.deg2rad(phi)),
            r * torch.cos(torch.deg2rad(theta)),
        ]
    )
    if normalize:
        return out / torch.linalg.norm(out, dim=1)[:, None]
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
    r: TensorType = torch.ones(1),
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
    u, v, w = spherical_to_cartesian(r, theta, phi, normalize=False).T
    return go.Cone(
        x=x,
        y=y,
        z=z,
        u=u,
        v=v,
        w=w,
        anchor="tail",
        sizemode="absolute",
        colorscale=None,
    )


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


def slider_time_line(
    data: TensorType, box_dim: BoxDimType = (-5, 5), nticks: int = 4
) -> go.Figure:
    """return a plotly figure of the data, with a slider to change the time

    Args:
        data (TensorType): data of shape (N, 5)
        with the first three columns as the x, y, z coordinates,
        and the last two columns as the theta, phi angles.

    Returns:
        go.Figure: figure with slider.
    """
    fig = go.Figure(data=cone_by_spherical(*data[0].T))
    steps = []
    for i in range(1, len(data)):
        fig.add_trace(cone_by_spherical(*data[i].T))
        step = dict(
            method="update",
            args=[
                {"visible": [False] * len(data)},
                {"title": f"Time: {i}"},
            ],
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [
        dict(
            active=0,
            currentvalue={"prefix": "Time: "},
            pad={"t": 50},
            steps=steps,
        )
    ]

    fig.update_layout(
        sliders=sliders,
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
    return fig


def create_timeline_series(
    data: TensorType, f: UpdateFuncType, steps: int, *arg, **kargs
) -> TensorType:
    """create timeline for data,

    Args:
        data (TensorType): input data, tensor (N,d)
        f (UpdateFuncType): function to update data, function :(N,d)->(N,d)
        steps (int): num of steps to advance
        *arg: args for f
        **kargs: kargs for f

    Returns:
        TensorType: tensor (steps,N,d), so ret[t] is represent the data int time t
    """
    if steps < 0:
        raise ValueError("steps must be grow then 0")
    res = torch.empty(steps + 1, *data.shape)
    res[0] = data
    for t in range(1, steps + 1):
        res[t] = f(res[t - 1], *arg, **kargs)
    return res

def plot_timeline(
    data: TensorType,
    title: str,
    figsize: BoxDimType = (1200, 1200),
    steps: Opt[IndexsType] = None,
    items: Opt[IndexsType] = None,
    duration: int = 1000,
    **kwargs,
) -> go.Figure:
    """plot timeline of data

    Args:
        data (TensorType): tensor of shape (t,m,2) where t is the number of time steps and m is the number of objects
        title (str): title of the plot
        figsize (BoxDimType, optional): figure size. Defaults to (10,10).
        items (Opt[IndexType], optional): indexs for items to plot, if not assimnte plot all. Defaults to None.
        **kwargs: kwargs for go.Layout
    Returns:
        go.Figure: animation figure of timeline
    """
    # get axis size
    max_range = max([data[:, :, 0].max().item(), data[:, :, 1].max().item()])
    items = range(data.shape[1]) if items is None else items
    steps = range(data.shape[0]) if steps is None else steps
    fig = go.Figure(
        data=[
            go.Scatter(
                x=data[steps[0], items, 0],
                y=data[steps[0], items, 1],
                mode="markers",
            )
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
            data=go.Scatter(x=data[t, items, 0], y=data[t, items, 1], mode="markers"),
        )
        for t in steps
    ]
    fig.frames = frames
    return fig




def get_avrage_velocity_by_radios(data: TensorType, radios: float = 5.0) -> TensorType:
    """get the avrage velocity of the input data

    Args:
        data (TensorType): tensor (N,5) of x,y,z,theta,phi
        radios (float, optional): radios of . Defaults to 5.0.

    Returns:
        TensorType: tensor (N,2) new velocity of the input data
    """
    # for each data point, get the distance from any other data point
    dis_mat = torch.stack(
        [torch.linalg.norm(data[i, :2] - data[:, :2], dim=1) for i in range(len(data))]
    )
    # for each data point, get the avrage velocity of the data points that are in the radios
    new_velocity = torch.stack(
        [data[dis_mat[i] < radios, 3:].mean(dim=0) for i in range(len(data))]
    )
    return new_velocity


def vicsek_update(
    data: TensorType,
    base_speed: float = 0.1,
    radius: float = 1.0,
    in_place: bool = False,
) -> TensorType:
    new_velocity = get_avrage_velocity_by_radios(data, radius)
    if in_place:
        data[:, 3:] = new_velocity
        data[:, :3] += base_speed * spherical_to_cartesian(
            torch.ones(data.shape[0]),
            new_velocity[:, 0],
            new_velocity[:, 1],
        )
        return data
    else:
        return torch.column_stack(
            [
                data[:,:3]
                + base_speed
                * spherical_to_cartesian(
                    torch.ones(data.shape[0]),
                    new_velocity[:, 0],
                    new_velocity[:, 1],
                ),
                new_velocity,
            ]
        )