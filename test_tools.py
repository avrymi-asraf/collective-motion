from tools import *


def test_show_cordinte():
    data = torch.rand(40, 5)
    data[:, 3:] = data[:, 3:] * 360
    data[:, :3] = data[:, :3] * 4 - 2
    show_cordinte(data).show()


if __name__ == "__main__":
    test_show_cordinte()
