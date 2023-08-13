from tools import *
import time

my_data = torch.tensor(
    (
        [
            [
                [0, 0],  # f1
                [10, 10],  # f2
                [25, 0],  # f3
            ],  # t1
            [
                [1, 1],  # f1
                [10, 11],  # f2
                [23, -1],  # f3
            ],  # t2
            [
                [2, 2],  # f1
                [10, 12],  # f2
                [21, -2],  # f3
            ],  # t3
            [
                [3, 3],  # f1
                [10, 13],  # f2
                [19, -3],  # f3
            ],  # t4
            [
                [4, 4],  # f1
                [10, 14],  # f2
                [17, -4],  # f3
            ],  # t5
            [
                [5, 5],  # f1
                [10, 15],  # f2
                [15, -4],  # f3
            ],  # t6
            [
                [6, 6],  # f1
                [10, 16],  # f2
                [17, -4],  # f3
            ],  # t7
            [
                [7, 7],  # f1
                [10, 17],  # f2
                [19, -3],  # f3
            ],  # t8
            [
                [8, 8],  # f1
                [10, 18],  # f2
                [25, 0],  # f3
            ],  # t9
        ]
    ),
    dtype=torch.float64,
)


def test_plot_timeline_with_direction():
    re = add_parameters(my_data)
    plot_timeline_with_direction(re, "test").show()


if __name__ == "__main__":
    # test_show_cordinte()
    test_plot_timeline_with_direction()
