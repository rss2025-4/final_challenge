import numpy as np

from final_challenge.homography import (
    get_foot,
    homography_line,
    line_direction,
    line_x_equals,
    line_y_equals,
    matrix_rot,
    point_coord,
    shift_line,
)


def _check_eq(a, b):
    assert np.allclose(a, b), f"expects equal, got\n{a}\nand\n{b}\n"


def _check_eq_after_norm(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    assert np.allclose(a, b), f"expects equal, got\n{a}\nand\n{b}\n"


def _line():
    return (1.234, 3.456, 4.567)


def test_xy_equals():
    _check_eq(line_direction(line_x_equals(1)), (0, 1))
    _check_eq(line_direction(line_x_equals(-1)), (0, 1))

    _check_eq(line_direction(line_y_equals(1)), (1, 0))
    _check_eq(line_direction(line_y_equals(-1)), (1, 0))


def test_rot():
    line = _line()
    a = 0.1

    x, y = line_direction(line)
    ans = (x + 1j * y) * (np.cos(a) + 1j * np.sin(a))

    x2, y2 = line_direction(homography_line(matrix_rot(a), line))
    _check_eq(x2 + 1j * y2, ans)


def test_shift():
    _check_eq_after_norm(shift_line(line_x_equals(10) * 1.2, 2), line_x_equals(8) * 2.3)

    _check_eq_after_norm(shift_line(line_y_equals(10) * 3.4, 2), line_y_equals(12) * 4.5)


def test_foot():
    _check_eq(point_coord(get_foot((1.23, 10), line_x_equals(1))), (1, 10))

    _check_eq(point_coord(get_foot((10, 1.23), line_y_equals(1))), (10, 1))
