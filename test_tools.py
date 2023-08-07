from tools import *
import time


def test_show_cordinte():
    data = torch.rand(40, 5)
    data[:, 3:] = data[:, 3:] * 360
    data[:, :3] = data[:, :3] * 4 - 2
    show_cordinte(data).show()


def test_slider_time_line():
    def f(x):
        return x + 0.1

    start = time.time()
    data = torch.rand(1000, 5)
    data[:, 3:] = data[:, 3:] * 360
    data[:, :3] = data[:, :3]
    timeline = create_timeline_series(data, f, 20)
    print(
        f"create timeline series with simple fucnction f cost {time.time() - start} seconds"
    )
    slider_time_line(timeline).show()


if __name__ == "__main__":
    # test_show_cordinte()
    test_slider_time_line()
