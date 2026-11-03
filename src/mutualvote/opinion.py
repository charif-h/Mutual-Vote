"""Binomial subjective-logic opinions (paper Section 3.1)."""

from __future__ import annotations

from dataclasses import dataclass

#: Non-informative prior weight ``W`` of the initial state ``r = s = 0``.
PRIOR_WEIGHT = 2.0


@dataclass(frozen=True)
class Opinion:
    """Evidence counters ``r`` (positive), ``s`` (negative), weight ``w``, base rate ``a``.

    ``w`` includes the prior weight, so a fresh opinion is ``Opinion(0, 0, 2)``.
    """

    r: float = 0.0
    s: float = 0.0
    w: float = PRIOR_WEIGHT
    a: float = 0.5

    @property
    def _total(self) -> float:
        return self.r + self.s + self.w

    @property
    def belief(self) -> float:
        """Eq. (2)."""
        return self.r / self._total

    @property
    def disbelief(self) -> float:
        """Eq. (3)."""
        return self.s / self._total

    @property
    def uncertainty(self) -> float:
        """Eq. (4)."""
        return self.w / self._total

    def expected(self) -> float:
        """Eq. (5): ``Ex = b + a * u``, in ``[0, 1]``."""
        return self.belief + self.a * self.uncertainty

    def score(self) -> float:
        """Expected value shifted to ``[-1, +1]``: ``2 * Ex - 1``."""
        return 2.0 * self.expected() - 1.0

    def positive(self, score: float) -> Opinion:
        """Add a positive interaction of strength ``score`` in ``[0, 1]`` (eq. 6)."""
        if not 0.0 <= score <= 1.0:
            raise ValueError("interaction score must be in [0, 1]")
        return Opinion(self.r + score, self.s, self.w + (1.0 - score), self.a)

    def negative(self, hostility: float = 0.5) -> Opinion:
        """Add a negative interaction with hostility barometer ``hostility`` (eq. 7)."""
        return self.hostile(hostility, 1)

    def hostile(self, hostility: float, count: float) -> Opinion:
        """Add ``count`` negative interactions at once (closed form of eq. 7)."""
        if not 0.0 <= hostility <= 1.0:
            raise ValueError("hostility must be in [0, 1]")
        return Opinion(
            self.r,
            self.s + hostility * count,
            self.w + (1.0 - hostility) * count,
            self.a,
        )

    def combine(self, other: Opinion) -> Opinion:
        """Sum the evidence of two opinions (the prior weight is counted once)."""
        if self.a != other.a:
            raise ValueError("cannot combine opinions with different base rates")
        return Opinion(
            self.r + other.r,
            self.s + other.s,
            self.w + other.w - PRIOR_WEIGHT,
            self.a,
        )
