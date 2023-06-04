import functools
import sys
from typing import TypeVar, Callable, Any, Iterator

if sys.version_info < (3, 8):
    # Ignoring type for mypy to avoid "Incompatible import" error (https://github.com/python/mypy/issues/4427).
    from typing_extensions import Protocol  # type: ignore
else:
    from typing import Protocol

_DIn = TypeVar("_DIn")


class Decorator(Protocol):
    """Protocol to mark a function as returning its child with identical signature."""

    def __call__(self, name: str) -> Callable[[_DIn], _DIn]:
        ...


# This is how functools.partials seems to do it, too, to retain the return type
PartialT = TypeVar("PartialT")


def partial(
    func: Callable[..., PartialT], *args: Any, **kwargs: Any
) -> Callable[..., PartialT]:
    """Wrapper around functools.partial that retains docstrings and can include
    other workarounds if needed.
    """
    partial_func = functools.partial(func, *args, **kwargs)
    partial_func.__doc__ = func.__doc__
    return partial_func


class Generator(Iterator):
    """Custom generator type. Used to annotate function arguments that accept
    generators so they can be validated by pydantic (which doesn't support
    iterators/iterables otherwise).
    """

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not hasattr(v, "__iter__") and not hasattr(v, "__next__"):
            raise TypeError("not a valid iterator")
        return v
