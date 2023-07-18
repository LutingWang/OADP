__all__ = [
    'BBox',
    'BBoxes',
    'BBoxesXYXY',
    'BBoxesXYWH',
    'BBoxesCXCYWH',
]

import numbers
from abc import ABC, abstractmethod
from typing import Generator, TypeVar, Tuple
from typing_extensions import Self

import einops
import torch

BBox = Tuple[numbers.Real, numbers.Real, numbers.Real, numbers.Real]
T = TypeVar('T', bound='BBoxes')


class BBoxes(ABC):

    def __init__(self, bboxes: torch.Tensor) -> None:
        """Initialize.

        Args:
            bboxes: :math:`n \\times 4`.

        Bounding boxes must be 2 dimensional and the second dimension must be
        of size 4:

            >>> bboxes = torch.tensor([[10.0, 20.0, 40.0, 100.0]])
            >>> BBoxesXYXY(bboxes[0])
            Traceback (most recent call last):
            ...
            ValueError: bboxes must be at least 2-dim
            >>> BBoxesXYXY(bboxes[:, :3])
            Traceback (most recent call last):
            ...
            ValueError: bboxes must have 4 columns
        """
        if bboxes.ndim < 2:
            raise ValueError('bboxes must be at least 2-dim')
        if bboxes.shape[-1] != 4:
            raise ValueError('bboxes must have 4 columns')
        self._bboxes = bboxes

    def __len__(self) -> int:
        """Number of bboxes.

        Examples:

            >>> bboxes = torch.tensor([
            ...     [5.0, 15.0, 8.0, 18.0],
            ...     [5.0, 15.0, 8.0, 60.0],
            ... ])
            >>> len(BBoxesXYXY(bboxes))
            2
        """
        return self._bboxes.shape[0]

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self._bboxes})'

    def __add__(self, other: Self) -> Self:
        """Concatenate bboxes.

        Args:
            other: :math:`n' \\times 4`.

        Returns:
            :math:`(n + n') \\times 4`, where `n` is the length of `self`.

        Examples:

            >>> a = torch.tensor([[5.0, 15.0, 8.0, 18.0]])
            >>> b = torch.tensor([[5.0, 15.0, 8.0, 60.0]])
            >>> BBoxesXYXY(a) + BBoxesXYXY(b)
            BBoxesXYXY(tensor([[ 5., 15.,  8., 18.],
                    [ 5., 15.,  8., 60.]]))
        """
        bboxes = torch.cat([self._bboxes, other._bboxes])
        return self.__class__(bboxes)

    def __getitem__(self, indices) -> Self:
        """Get specific bboxes.

        Args:
            indices: a index or multiple indices.

        Returns:
            If `indices` refers to a single box, return a ``tuple``.
            Otherwise, return ``BBoxes``.

        Examples:

            >>> bboxes = torch.tensor([
            ...     [5.0, 15.0, 8.0, 18.0],
            ...     [5.0, 15.0, 8.0, 60.0],
            ...     [5.0, 15.0, 8.0, 105.0],
            ... ])
            >>> BBoxesXYXY(bboxes)[0]
            BBoxesXYXY(tensor([[ 5., 15.,  8., 18.]]))
            >>> BBoxesXYXY(bboxes)[:-1]
            BBoxesXYXY(tensor([[ 5., 15.,  8., 18.],
                    [ 5., 15.,  8., 60.]]))
        """
        tensor = self._bboxes[indices]
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return self.__class__(tensor)

    def __iter__(self) -> Generator[BBox, None, None]:
        """Iterate over bboxes.

        Yields:
            One bbox.

        Examples:

            >>> bboxes = torch.tensor([
            ...     [5.0, 15.0, 8.0, 18.0],
            ...     [5.0, 15.0, 8.0, 60.0],
            ...     [5.0, 15.0, 8.0, 105.0],
            ... ])
            >>> for bbox in BBoxesXYXY(bboxes):
            ...     print(bbox)
            (5.0, 15.0, 8.0, 18.0)
            (5.0, 15.0, 8.0, 60.0)
            (5.0, 15.0, 8.0, 105.0)
        """
        yield from map(tuple, self._bboxes.tolist())

    def __and__(self, other: 'BBoxes') -> torch.Tensor:
        """Intersections.

        Args:
            other: :math:`n' \\times 4`.

        Returns:
            :math:`n \\times n'`.
        """
        lt = torch.maximum(  # [n, n', 2]
            einops.rearrange(self.lt, 'n1 lt -> n1 1 lt'),
            einops.rearrange(other.lt, 'n2 lt -> 1 n2 lt'),
        )
        rb = torch.minimum(  # [n, n', 2]
            einops.rearrange(self.rb, 'n1 rb -> n1 1 rb'),
            einops.rearrange(other.rb, 'n2 rb -> 1 n2 rb'),
        )
        wh = rb - lt
        wh = wh.clamp_min_(0)
        return wh[..., 0] * wh[..., 1]

    def __or__(self, other: 'BBoxes') -> torch.Tensor:
        """Wraps `unions`.

        Args:
            other: :math:`n' \\times 4`.

        Returns:
            :math:`n \\times n'`.
        """
        return self.unions(other, self & other)

    @classmethod
    @abstractmethod
    def _from_bboxes(cls, bboxes: 'BBoxes') -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def left(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def right(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def top(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def bottom(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def width(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def height(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def center_x(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def center_y(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def lt(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def rb(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def wh(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def center(self) -> torch.Tensor:
        pass

    @abstractmethod
    def _round(self) -> torch.Tensor:
        pass

    @abstractmethod
    def _expand(self, ratio_wh: Tuple[float, float]) -> torch.Tensor:
        pass

    @abstractmethod
    def _clamp(self, image_wh: Tuple[int, int]) -> torch.Tensor:
        pass

    @abstractmethod
    def _scale(self, ratio_wh: Tuple[float, float]) -> torch.Tensor:
        pass

    @abstractmethod
    def _translate(self, offset: torch.Tensor) -> torch.Tensor:
        """Translate bboxes.

        Args:
            offset: :math:`n \\times 2` or 2.

        Returns:
            Translated bboxes.
        """
        pass

    @property
    def area(self) -> torch.Tensor:
        return self.width * self.height

    def to_tensor(self) -> torch.Tensor:
        return self._bboxes

    def to(self, cls):
        return cls.from_bboxes(self)

    def unions(
        self,
        other: 'BBoxes',
        intersections: torch.Tensor,
    ) -> torch.Tensor:
        """Unions.

        Args:
            other: :math:`n' \\times 4`.
            intersections: :math:`n \\times n'`

        Returns:
            :math:`n \\times n'`.
        """
        return self.area[:, None] + other.area[None, :] - intersections

    def ious(self, other: 'BBoxes', eps: float = 1e-6) -> torch.Tensor:
        """Intersections over unions.

        Args:
            other: :math:`n' \\times 4`.
            eps: avoid division by zero.

        Returns:
            :math:`n \\times n'`.
        """
        intersections = self & other
        unions = self.unions(other, intersections).clamp_min_(eps)
        return intersections / unions

    @classmethod
    def from_bboxes(cls, *args, **kwargs) -> Self:
        return cls(cls._from_bboxes(*args, **kwargs))

    def round(self, *args, **kwargs) -> Self:
        return self.__class__(self._round(*args, **kwargs))

    def expand(self, *args, **kwargs) -> Self:
        return self.__class__(self._expand(*args, **kwargs))

    def clamp(self, *args, **kwargs) -> Self:
        return self.__class__(self._clamp(*args, **kwargs))

    def scale(self, *args, **kwargs) -> Self:
        return self.__class__(self._scale(*args, **kwargs))

    def translate(self, *args, **kwargs) -> Self:
        return self.__class__(self._translate(*args, **kwargs))

    def indices(
        self,
        *,
        min_area = None,
        min_wh = None,
    ) -> torch.Tensor:
        indices = self._bboxes.new_ones(len(self), dtype=torch.bool)
        if min_area is not None:
            indices &= self.area.ge(min_area)
        if min_wh is not None:
            indices &= self.wh.ge(torch.tensor(min_wh)).all(-1)
        return indices


class BBoxesXY(BBoxes):

    @property
    def left(self) -> torch.Tensor:
        return self._bboxes[:, 0]

    @property
    def top(self) -> torch.Tensor:
        return self._bboxes[:, 1]

    @property
    def lt(self) -> torch.Tensor:
        return self._bboxes[:, :2]

    def _scale(self, ratio_wh) -> torch.Tensor:
        scale = torch.tensor(ratio_wh * 2)
        return self._bboxes * scale


class BBoxesWH(BBoxes):

    @property
    def width(self) -> torch.Tensor:
        return self._bboxes[:, 2]

    @property
    def height(self) -> torch.Tensor:
        return self._bboxes[:, 3]

    @property
    def wh(self) -> torch.Tensor:
        return self._bboxes[:, 2:]

    def _translate(self, offset: torch.Tensor) -> torch.Tensor:
        offset = torch.cat([offset, torch.zeros_like(offset)], dim=-1)
        return self._bboxes + offset


class BBoxesXYXY(BBoxesXY):

    @classmethod
    def _from_bboxes(cls, bboxes: BBoxes) -> torch.Tensor:
        return torch.cat([bboxes.lt, bboxes.rb], dim=-1)

    @property
    def right(self) -> torch.Tensor:
        return self._bboxes[:, 2]

    @property
    def bottom(self) -> torch.Tensor:
        return self._bboxes[:, 3]

    @property
    def width(self) -> torch.Tensor:
        return self.right - self.left

    @property
    def height(self) -> torch.Tensor:
        return self.bottom - self.top

    @property
    def center_x(self) -> torch.Tensor:
        return (self.left + self.right) / 2

    @property
    def center_y(self) -> torch.Tensor:
        return (self.top + self.bottom) / 2

    @property
    def rb(self) -> torch.Tensor:
        return self._bboxes[:, 2:]

    @property
    def wh(self) -> torch.Tensor:
        return self.rb - self.lt

    @property
    def center(self) -> torch.Tensor:
        return (self.lt + self.rb) / 2

    def _round(self) -> torch.Tensor:
        lt = self.lt.floor()
        rb = self.rb.ceil()
        return torch.cat([lt, rb], dim=-1)

    def _expand(self, ratio_wh) -> torch.Tensor:
        delta = self.wh * (torch.tensor(ratio_wh) - 1) / 2
        delta = torch.cat([-delta, delta], dim=-1)
        return self._bboxes + delta

    def _clamp(self, image_wh) -> torch.Tensor:
        lt = self.lt.clamp_min(0)
        rb = self.rb.clamp_max(torch.tensor(image_wh))
        return torch.cat([lt, rb], dim=-1)

    def _translate(self, offset: torch.Tensor) -> torch.Tensor:
        offset = torch.cat([offset, offset], dim=-1)
        return self._bboxes + offset


class BBoxesXYWH(BBoxesXY, BBoxesWH):

    @classmethod
    def _from_bboxes(cls, bboxes: BBoxes) -> torch.Tensor:
        return torch.cat([bboxes.lt, bboxes.wh], dim=-1)

    @property
    def right(self) -> torch.Tensor:
        return self.left + self.width

    @property
    def bottom(self) -> torch.Tensor:
        return self.top + self.height

    @property
    def center_x(self) -> torch.Tensor:
        return self.left + self.width / 2

    @property
    def center_y(self) -> torch.Tensor:
        return self.top + self.height / 2

    @property
    def rb(self) -> torch.Tensor:
        return self.lt + self.wh

    @property
    def center(self) -> torch.Tensor:
        return self.lt + self.wh / 2

    def _round(self) -> torch.Tensor:
        lt = self.lt.floor()
        wh = self.rb.ceil() - lt
        return torch.cat([lt, wh], dim=-1)

    def _expand(self, ratio_wh) -> torch.Tensor:
        wh = self.wh * torch.tensor(ratio_wh)
        lt = self.lt - (wh - self.wh) / 2
        return torch.cat([lt, wh], dim=-1)

    def _clamp(self, image_wh) -> torch.Tensor:
        lt = self.lt.clamp_min(0)
        rb = self.rb.clamp_max(torch.tensor(image_wh))
        return torch.cat([lt, rb - lt], dim=-1)


class BBoxesCXCYWH(BBoxesWH):

    @classmethod
    def _from_bboxes(cls, bboxes: BBoxes) -> torch.Tensor:
        return torch.cat([bboxes.center, bboxes.wh], dim=-1)

    @property
    def center_x(self) -> torch.Tensor:
        return self._bboxes[:, 0]

    @property
    def center_y(self) -> torch.Tensor:
        return self._bboxes[:, 1]

    @property
    def center(self) -> torch.Tensor:
        return self._bboxes[:, :2]

    @property
    def left(self) -> torch.Tensor:
        return self.center_x - self.width / 2

    @property
    def right(self) -> torch.Tensor:
        return self.center_x + self.width / 2

    @property
    def top(self) -> torch.Tensor:
        return self.center_y - self.height / 2

    @property
    def bottom(self) -> torch.Tensor:
        return self.center_y + self.height / 2

    @property
    def lt(self) -> torch.Tensor:
        return self.center - self.wh / 2

    @property
    def rb(self) -> torch.Tensor:
        return self.center + self.wh / 2

    def _round(self) -> torch.Tensor:
        lt = self.lt.floor()
        rb = self.rb.ceil()
        center = (lt + rb) / 2
        wh = (rb - lt) / 2
        return torch.cat([center, wh], dim=-1)

    def _expand(self, ratio_wh) -> torch.Tensor:
        wh = self.wh * torch.tensor(ratio_wh)
        return torch.stack([self.center, wh], dim=-1)

    def _clamp(self, image_wh) -> torch.Tensor:
        lt = self.lt.clamp_min(0)
        rb = self.rb.clamp_max(torch.tensor(image_wh))
        center = (lt + rb) / 2
        wh = (rb - lt) / 2
        return torch.stack([center, wh], dim=-1)

    def _scale(self, ratio_wh) -> torch.Tensor:
        w, h = ratio_wh
        ratio_center = w / 2, h / 2
        scale = torch.tensor(ratio_center + ratio_wh)
        return self._bboxes * scale
