from typing import List, Tuple, Literal, Union
from pydantic import BaseModel, Field, conlist


class AABBConfig(BaseModel):
    center: List[float]
    limits: List[float]


class Polygon2dConfig(BaseModel):
    pts: List[List[float]] = Field(default=[])


class Polygon2dSetConfig(BaseModel):
    polys: List[List[float]] = Field(default=[])


class RigidSpace2dConfig(BaseModel):
    kind: Literal["rigid2d"] = "rigid2d"
    q_min: List[float]
    q_max: List[float]


class AABBCheckerConfig(BaseModel):
    kind: Literal["AABBCheckerConfig"] = "AABBCheckerConfig"
    obstacles: List[AABBConfig] = Field(default=[])
    collsion_step: float = 1.0


class Polygon2dCheckerConfig(BaseModel):
    kind: Literal["Polygon2dCheckerConfig"] = "Polygon2dCheckerConfig"
    robot: Polygon2dSetConfig
    obstacles: Polygon2dSetConfig
    collsion_step: float = 1.0


class VectorSpace2dConfig(BaseModel):
    kind: Literal["VectorSpace2dConfig"] = "VectorSpace2dConfig"
    q_min: List[float]
    q_max: List[float]


class RRTConfig(BaseModel):
    kind: Literal["RRTConfig"] = "RRTConfig"
    n: int
    eta: float


class PRMConfig(BaseModel):
    kind: Literal["PRMConfig"] = "PRMConfig"
    n: int
    r: float


PlannerConfig = Union[RRTConfig, PRMConfig]

SpaceConfig = Union[RigidSpace2dConfig, VectorSpace2dConfig]

CheckerConfig = Union[AABBCheckerConfig, Polygon2dCheckerConfig]


class DiscreteGoalConfig(BaseModel):
    kind: Literal["discrete"] = "discrete"
    location: List[float]


GoalConfig = Union[DiscreteGoalConfig]


class ProblemConfig(BaseModel):
    planner: PlannerConfig = Field(discriminator="kind")
    space: SpaceConfig = Field(discriminator="kind")
    checker: CheckerConfig = Field(discriminator="kind")
    goal: GoalConfig = Field(discriminator="kind")
    initial: List[float]
