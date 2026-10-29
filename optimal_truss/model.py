"""Truss inputs in inches and ounce-force; member and joint indexes are zero based."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Truss:
    """A planar truss with a pin, a vertical roller reaction, and one downward load."""

    name: str
    nodes: np.ndarray
    members: np.ndarray
    pin: int
    roller: int
    load_joint: int

    def __post_init__(self):
        self.nodes = np.array(self.nodes, dtype=float, copy=True)
        raw_members = np.asarray(self.members)
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 2 or len(self.nodes) < 3:
            raise ValueError("nodes must contain at least three [x, y] coordinates")
        if not np.isfinite(self.nodes).all():
            raise ValueError("Joint coordinates must be finite")
        if raw_members.ndim != 2 or raw_members.shape[1] != 2 or not len(raw_members):
            raise ValueError("members must contain [start_joint, end_joint] pairs")
        if raw_members.dtype.kind not in "ifu":
            raise ValueError("Member endpoints must be integer joint indexes")
        if not np.isfinite(raw_members).all() or not np.equal(raw_members, np.floor(raw_members)).all():
            raise ValueError("Member endpoints must be integer joint indexes")
        self.members = raw_members.astype(int, copy=True)
        if np.any(self.members < 0) or np.any(self.members >= len(self.nodes)):
            raise ValueError("A member refers to a joint that does not exist")
        edges = [tuple(sorted(pair)) for pair in self.members.tolist()]
        if any(a == b for a, b in edges) or len(set(edges)) != len(edges):
            raise ValueError("Members must have distinct endpoints and cannot be duplicated")
        for label in ("pin", "roller", "load_joint"):
            index = getattr(self, label)
            if not isinstance(index, (int, np.integer)) or not 0 <= index < len(self.nodes):
                raise ValueError(f"{label} must be an integer joint index")
            setattr(self, label, int(index))
        if self.pin == self.roller:
            raise ValueError("Pin and roller must be different joints")

    def to_dict(self):
        """Return the editable JSON representation, with no rounded coordinates."""
        return {
            "name": self.name,
            "nodes": self.nodes.tolist(),
            "members": self.members.tolist(),
            "pin": self.pin,
            "roller": self.roller,
            "load_joint": self.load_joint,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(**{key: data[key] for key in (
            "name", "nodes", "members", "pin", "roller", "load_joint"
        )})


@dataclass(frozen=True)
class BucklingLaw:
    """Class fit Fb = coefficient * length**(-exponent), in oz-force and inches.

    The 1.52 oz fit uncertainty is a supplied 95% band, not a standard deviation.
    The coefficient retains the precision shown in the class data-fit graphic.
    """

    coefficient: float = 3391.349
    exponent: float = 2.012
    uncertainty_oz: float = 1.52

    def __post_init__(self):
        if not np.isfinite([self.coefficient, self.exponent, self.uncertainty_oz]).all():
            raise ValueError("Buckling-fit parameters must be finite")
        if self.coefficient <= 0 or self.exponent <= 0 or self.uncertainty_oz < 0:
            raise ValueError("Coefficient/exponent must be positive; uncertainty cannot be negative")

    def capacity(self, lengths):
        lengths = np.asarray(lengths, dtype=float)
        if np.any(lengths <= 0) or not np.isfinite(lengths).all():
            raise ValueError("Member lengths must be finite and positive")
        return self.coefficient * lengths ** -self.exponent


@dataclass(frozen=True)
class Constraints:
    """Fall 2026 specification limits; lengths in inches and load in oz-force."""

    span_min: float = 30.0
    span_max: float = 33.0
    member_min: float = 8.5
    member_max: float = 14.0
    load_height_min: float = 0.5
    load_height_max: float = 1.5
    max_total_length: float = 120.0
    stock_lengths: tuple = (48.0, 48.0, 24.0)
    min_load: float = 30.0
    kerf_in: float = 0.0

    def __post_init__(self):
        values = [self.span_min, self.span_max, self.member_min, self.member_max,
                  self.load_height_min, self.load_height_max, self.max_total_length,
                  self.min_load, self.kerf_in]
        if not np.isfinite(values).all():
            raise ValueError("Constraint limits must be finite")
        for lower, upper in ((self.span_min, self.span_max),
                             (self.member_min, self.member_max),
                             (self.load_height_min, self.load_height_max)):
            if lower > upper:
                raise ValueError("Each lower constraint limit must be <= its upper limit")
        if min(self.span_min, self.member_min, self.max_total_length) <= 0:
            raise ValueError("Span, member length, and total material limits must be positive")
        if min(self.load_height_min, self.min_load, self.kerf_in) < 0:
            raise ValueError("Load height, minimum load, and kerf cannot be negative")
        stock = tuple(float(length) for length in self.stock_lengths)
        if not stock or not np.isfinite(stock).all() or min(stock) <= 0:
            raise ValueError("Stock lengths must be positive and finite")
        object.__setattr__(self, "stock_lengths", stock)
