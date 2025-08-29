import numpy as np
from baldric.collision import CollisionChecker
from baldric.goals import Goal
from .sampler import FreespaceSampler


class BiasedFreespaceSampler(FreespaceSampler):
    def __init__(self, checker: CollisionChecker, goal: Goal, p_goal_sample: float = 0.05):
        super().__init__(checker)
        self.goal = goal
        assert p_goal_sample >= 0
        assert p_goal_sample <= 1
        self.p_goal_sample = p_goal_sample

    def sample(self):
        """Return a sample biased toward a goal"""
        x = np.random.random()
        if x <= self.p_goal_sample:
            q = self.goal.sample()
            if self.collisionFree(q):
                return q
        return super().sample()
