from typing import Any, Callable, Generic, Iterable, ParamSpec, TypeVar, final

T = TypeVar("T")
R = TypeVar("R")
P = ParamSpec("P")


def unique(x: Iterable[T]) -> T:
    it = iter(x)
    first = next(it)
    try:
        sec = next(it)
        raise ValueError(f"expected unique, got at least two elements: {first} and {sec}")
    except StopIteration:
        return first


@final
class _empty_t:
    pass


_empty = _empty_t()


class cast(Generic[T]):
    def __init__(self, _: T | _empty_t = _empty) -> None:
        pass

    def __call__(self, a: T) -> T:
        return a


class cast_unchecked(Generic[T]):
    def __init__(self, _: T | _empty_t = _empty) -> None:
        pass

    def __call__(self, a) -> T:
        return a


cast_unchecked_ = cast_unchecked()


def cast_fsig(f1: Callable[P, Any]) -> Callable[[Callable[..., R]], Callable[P, R]]:
    def inner(f2: Callable[..., R]) -> Callable[P, R]:
        return f2

    return inner


old_zip = zip


@cast_unchecked(zip)
def zip(*args):
    args = list(map(list, args))
    n = len(args[0])
    for arg in args[1:]:
        assert len(arg) == n, f"length mismatch: {list(map(len, args))}"
    return list(old_zip(*args))
