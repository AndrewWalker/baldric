from typing import List, Literal, Union
from pydantic import BaseModel, Field


class AABBConfig(BaseModel):
    center: List[float]
    limits: List[float]


class Polygon2dConfig(BaseModel):
    pts: List[List[float]] = Field(default=[])


class Polygon2dSetConfig(BaseModel):
    polys: List[Polygon2dConfig] = Field(default=[])


class RigidSpace2dConfig(BaseModel):
    q_min: List[float]
    q_max: List[float]
    kind: Literal["rigid2d"] = "rigid2d"


class DubinsSpaceConfig(BaseModel):
    q_min: List[float]
    q_max: List[float]
    rho: float
    kind: Literal["dubins"] = "dubins"


class VectorSpace2dConfig(BaseModel):
    q_min: List[float] = Field(default=[])
    q_max: List[float] = Field(default=[])
    kind: Literal["VectorSpace2dConfig"] = "VectorSpace2dConfig"


class AABBCheckerConfig(BaseModel):
    obstacles: List[AABBConfig] = Field(default=[])
    collsion_step: float = 1.0
    kind: Literal["AABBCheckerConfig"] = "AABBCheckerConfig"


class Polygon2dCheckerConfig(BaseModel):
    robot: Polygon2dSetConfig
    obstacles: Polygon2dSetConfig | None
    collsion_step: float = 1.0
    kind: Literal["Polygon2dCheckerConfig"] = "Polygon2dCheckerConfig"


class RRTConfig(BaseModel):
    n: int
    eta: float
    kind: Literal["rrt"] = "rrt"


class PRMConfig(BaseModel):
    n: int
    r: float
    kind: Literal["prm"] = "prm"


PlannerConfig = Union[RRTConfig, PRMConfig]

SpaceConfig = Union[RigidSpace2dConfig, VectorSpace2dConfig, DubinsSpaceConfig]

CheckerConfig = Union[AABBCheckerConfig, Polygon2dCheckerConfig]


class DiscreteGoalConfig(BaseModel):
    location: List[float]
    tolerance: float = 1.0
    kind: Literal["discrete"] = "discrete"


class DubinsGoalConfig(BaseModel):
    location: List[float]
    tolerance: float = 1.0
    kind: Literal["dubins"] = "dubins"


GoalConfig = Union[DiscreteGoalConfig, DubinsGoalConfig]


class VectorNearestConfig(BaseModel):
    kind: Literal["vector"] = "vector"


class NaiveNearestConfig(BaseModel):
    kind: Literal["naive"] = "naive"


NearestConfig = Union[VectorNearestConfig, NaiveNearestConfig]


class ProblemConfig(BaseModel):
    planner: PlannerConfig = Field(discriminator="kind")
    space: SpaceConfig = Field(discriminator="kind")
    checker: CheckerConfig = Field(discriminator="kind")
    metric: NearestConfig = Field(discriminator="kind", default=VectorNearestConfig())
    goal: GoalConfig = Field(discriminator="kind")
    initial: List[float]
