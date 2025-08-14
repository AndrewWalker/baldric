import numpy as np
from baldric.collision import CollisionChecker
from baldric.collision.aabb_collision import AABB, AABBCollisionChecker
from baldric.collision.convex_collision import (
    ConvexPolygon2d,
    ConvexPolygon2dSet,
    ConvexPolygon2dCollisionChecker,
)
from baldric.problem import Problem
from baldric.metrics import VectorNearest
from baldric.sampler import EmbeddingFreespaceSampler
from baldric.spaces import Space, RigidBody2dSpace, VectorSpace
from baldric.planners import Planner
from baldric.planners.prm import PlannerPRM
from baldric.planners.rrt import PlannerRRT
from baldric.commands.config import (
    AABBConfig,
    Polygon2dSetConfig,
    Polygon2dConfig,
    RigidSpace2dConfig,
    VectorSpace2dConfig,
    RRTConfig,
    PRMConfig,
    ProblemConfig,
    Polygon2dCheckerConfig,
    AABBCheckerConfig,
)


def create_planner(config: ProblemConfig, checker: CollisionChecker):
    sampler = EmbeddingFreespaceSampler(
        low=np.array(config.space.q_min),
        high=np.array(config.space.q_max),
        colltest=checker,
    )
    match config.planner:
        case RRTConfig():
            return PlannerRRT(
                sampler=sampler,
                nearest=VectorNearest(checker._space),
                colltest=checker,
                n=config.planner.n,
                eta=config.planner.eta,
            )
        case PRMConfig():
            return PlannerPRM(
                sampler=sampler,
                nearest=VectorNearest(checker._space),
                colltest=checker,
                n=config.planner.n,
                r=config.planner.r,
            )


def create_space(config: ProblemConfig):
    match config.space:
        case VectorSpace2dConfig():
            return VectorSpace(dimension=2)
        case RigidSpace2dConfig():
            return RigidBody2dSpace()


def create_polygon(config: Polygon2dConfig):
    return ConvexPolygon2d(pts=np.array(config.pts))


def from_file(filename: str):
    return Problem(space=None, collision_checker=None, planner=None)


def create_aabb(config: AABBConfig):
    return AABB(np.array(config.center), np.array(config.limits))


def create_polygon_set(config: Polygon2dSetConfig):
    polys = []
    for poly in config.polys:
        polys.append(ConvexPolygon2d(pts=np.array(poly)))
    return ConvexPolygon2d


def create_checker(config: ProblemConfig, space: Space):
    checker = config.checker
    match checker:
        case Polygon2dCheckerConfig():
            obs = create_polygon(checker.obstacles)
            bot = create_polygon(checker.robot)
            res = ConvexPolygon2dCollisionChecker(
                space=space, obs=obs, bot=bot, step=checker.collsion_step
            )
            return res
        case AABBCheckerConfig():
            aabbs = [create_aabb(aabb) for aabb in checker.obstacles]
            return AABBCollisionChecker(space, boxes=aabbs, step=checker.collsion_step)


def create_problem(config: ProblemConfig):
    p = Problem()
    p.space = create_space(config)
    p.collision_checker = create_checker(config, p.space)
    p.planner = create_planner(config, p.collision_checker)
    match p.space, p.collision_checker:
        case RigidBody2dSpace(), ConvexPolygon2dCollisionChecker():
            p.space.set_weights_from_pts(p.collision_checker.robot.all_points)
    return p
