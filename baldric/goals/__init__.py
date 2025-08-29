from typing import Union, Callable
from baldric.configuration import Configuration
from baldric.spaces import Space
from baldric.geometry.aabb import AABB


class GoalBase:
    def satisified(self, q: Configuration):
        return False

    def sample(self):
        raise NotImplementedError


class DiscreteGoal(GoalBase):
    """A goal that is satisifed when a configuration is within a distance of the goal"""

    def __init__(self, location: Configuration, tolerance: float, space: Space):
        super().__init__()
        self.location = location
        self.space = space
        self.tolerance = tolerance

    def satisified(self, q: Configuration):
        return self.space.distance(self.location, q) < self.tolerance

    def sample(self):
        return self.location


class PredicateGoal(GoalBase):
    """A goal that is satisifed when a configuration satisifes a predicate"""

    def __init__(self, location: Configuration, pred: Callable[[Configuration, Configuration], bool]):
        self.location = location
        self.pred = pred

    def satisified(self, q: Configuration):
        return self.pred(self.location, q)

    def sample(self):
        return self.location


class AABBGoal(GoalBase):
    """A goal that is satisifed when a configuration is inside an AABB"""

    def __init__(self, box: AABB):
        self.box = box

    def satisified(self, q: Configuration):
        return self.box.containsPt(q)

    def sample(self):
        return self.box.sample()


Goal = Union[DiscreteGoal, PredicateGoal, AABBGoal]
