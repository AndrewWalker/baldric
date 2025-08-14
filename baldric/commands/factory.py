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
from baldric.sampler import FreespaceSampler
from baldric.spaces import Space, RigidBody2dSpace, VectorSpace
from baldric.planners import Planner
from baldric.planners.prm import PlannerPRM
from baldric.planners.rrt import PlannerRRT
from baldric.commands import config


def create_planner(cfg: config.ProblemConfig, checker: CollisionChecker):
    sampler = FreespaceSampler(
        colltest=checker,
    )
    match cfg.planner:
        case config.RRTConfig():
            return PlannerRRT(
                sampler=sampler,
                nearest=VectorNearest(checker._space),
                colltest=checker,
                n=cfg.planner.n,
                eta=cfg.planner.eta,
            )
        case config.PRMConfig():
            return PlannerPRM(
                sampler=sampler,
                nearest=VectorNearest(checker._space),
                colltest=checker,
                n=cfg.planner.n,
                r=cfg.planner.r,
            )


def create_space(cfg: config.ProblemConfig):
    match cfg.space:
        case config.VectorSpace2dConfig():
            return VectorSpace(dimension=2)
        case config.RigidSpace2dConfig():
            return RigidBody2dSpace()


def create_polygon(cfg: config.Polygon2dConfig):
    return ConvexPolygon2d(pts=np.array(cfg.pts))


def create_aabb(cfg: config.AABBConfig):
    return AABB(np.array(cfg.center), np.array(cfg.limits))


def create_polygon_set(cfg: config.Polygon2dSetConfig):
    polys = []
    for poly in cfg.polys:
        polys.append(ConvexPolygon2d(pts=np.array(poly)))
    return ConvexPolygon2d


def create_checker(cfg: config.ProblemConfig, space: Space):
    checker = cfg.checker
    match checker:
        case config.Polygon2dCheckerConfig():
            obs = create_polygon(checker.obstacles)
            bot = create_polygon(checker.robot)
            res = ConvexPolygon2dCollisionChecker(
                space=space, obs=obs, bot=bot, step=checker.collsion_step
            )
            return res
        case config.AABBCheckerConfig():
            aabbs = [create_aabb(aabb) for aabb in checker.obstacles]
            return AABBCollisionChecker(space, boxes=aabbs, step=checker.collsion_step)


def create_problem(cfg: config.ProblemConfig):
    p = Problem()
    p.space = create_space(cfg)
    p.collision_checker = create_checker(cfg, p.space)
    p.planner = create_planner(cfg, p.collision_checker)
    match p.space, p.collision_checker:
        case RigidBody2dSpace(), ConvexPolygon2dCollisionChecker():
            p.space.set_weights_from_pts(p.collision_checker.robot.all_points)
    return p
