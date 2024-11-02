from typing import Callable, ParamSpec, TypeVar, cast

import numpy as np
import numpy.typing as npt

_P = ParamSpec("_P")
_P2 = ParamSpec("_P2")
_T = TypeVar("_T")
_Self = TypeVar("_Self")

NPFloats = npt.NDArray[np.float64]
NPFloatMatrix = np.ndarray[tuple[int, int], np.dtype[np.float64]]
NPImages = np.ndarray[tuple[int, int, int, int], np.dtype[np.float64]]
NPInt = npt.NDArray[np.int64]
NPIntMatrix = np.ndarray[tuple[int, int], np.dtype[np.int64]]
RecursiveDict = dict[str, dict[str, "RecursiveDict | str | int"]]


# https://stackoverflow.com/a/71968448
def copy_args(kwargs_call: Callable[_P, object]) -> Callable[[Callable[..., _T]], Callable[_P, _T]]:
    """Decorator does nothing but returning the casted original function"""

    def return_func(func: Callable[..., _T]) -> Callable[_P, _T]:
        return cast(Callable[_P, _T], func)

    return return_func
