import numpy as np
import yaml
from baldric.collision import CollisionChecker
from baldric.collision.aabb_collision import AABB, AABBCollisionChecker
from baldric.collision.convex_collision import (
    ConvexPolygon2d,
    ConvexPolygon2dSet,
    ConvexPolygon2dCollisionChecker,
)
from baldric.problem import Problem
from baldric.metrics import VectorNearest, NaiveNearest, Nearest
from baldric.sampler import FreespaceSampler
from baldric.spaces import Space, RigidBody2dSpace, VectorSpace, DubinsSpace
from baldric.planners import Planner, Goal, DiscreteGoal, PredicateGoal
from baldric.planners.prm import PlannerPRM
from baldric.planners.rrt import PlannerRRT
from baldric.commands import config


def create_planner(cfg: config.ProblemConfig, checker: CollisionChecker, nearest: Nearest):
    sampler = FreespaceSampler(
        checker=checker,
    )
    match cfg.planner:
        case config.RRTConfig():
            return PlannerRRT(
                sampler=sampler,
                nearest=nearest,
                colltest=checker,
                n=cfg.planner.n,
                eta=cfg.planner.eta,
            )
        case config.PRMConfig():
            return PlannerPRM(
                sampler=sampler,
                nearest=nearest,
                colltest=checker,
                n=cfg.planner.n,
                r=cfg.planner.r,
            )


def create_nearest(cfg: config.ProblemConfig, space: Space):
    match cfg.metric:
        case config.NaiveNearestConfig():
            return NaiveNearest(space)
        case config.VectorNearestConfig():
            return VectorNearest(space)


def create_space(cfg: config.ProblemConfig):
    space = cfg.space
    match space:
        case config.VectorSpace2dConfig():
            return VectorSpace(low=np.array(space.q_min), high=np.array(space.q_max))
        case config.RigidSpace2dConfig():
            return RigidBody2dSpace(low=np.array(space.q_min), high=np.array(space.q_max))
        case config.DubinsSpaceConfig():
            return DubinsSpace(low=np.array(space.q_min), high=np.array(space.q_max), rho=space.rho)


def create_aabb(cfg: config.AABBConfig):
    return AABB(np.array(cfg.center), np.array(cfg.limits))


def create_polygon_set(cfg: config.Polygon2dSetConfig):
    polys = []
    for poly in cfg.polys:
        polys.append(ConvexPolygon2d(pts=np.array(poly.pts)))
    return ConvexPolygon2dSet(polys=polys)


def create_checker(cfg: config.ProblemConfig, space: Space):
    checker = cfg.checker
    match checker:
        case config.Polygon2dCheckerConfig():
            obs = create_polygon_set(checker.obstacles)
            bot = create_polygon_set(checker.robot)
            res = ConvexPolygon2dCollisionChecker(space=space, obs=obs, robot=bot, step=checker.collsion_step)
            return res
        case config.AABBCheckerConfig():
            aabbs = [create_aabb(aabb) for aabb in checker.obstacles]
            return AABBCollisionChecker(space, boxes=aabbs, step=checker.collsion_step)


def create_goal(cfg: config.GoalConfig, space: Space):
    match cfg:
        case config.DiscreteGoalConfig():
            return DiscreteGoal(location=np.asarray(cfg.location), tolerance=cfg.tolerance, space=space)
        case config.DubinsGoalConfig():

            def goalfn(q0, q1):
                return np.linalg.norm((q0 - q1)[:2]) < cfg.tolerance

            return PredicateGoal(location=np.asarray(cfg.location), pred=goalfn)


def create_problem(cfg: config.ProblemConfig):
    p = Problem()
    p.space = create_space(cfg)
    p.init = np.asarray(cfg.initial)
    p.goal = create_goal(cfg.goal, p.space)
    p.collision_checker = create_checker(cfg, p.space)
    p.nearest = create_nearest(cfg, p.space)
    p.planner = create_planner(cfg, p.collision_checker, p.nearest)
    match p.space, p.collision_checker:
        case RigidBody2dSpace(), ConvexPolygon2dCollisionChecker():
            # p.space.set_weights_from_pts(p.collision_checker.robot.all_points)
            p.space.set_weights(np.array([1.0, 1.0, 0.0]))
    return p


def load_problem(fname: str):
    obj = yaml.safe_load(open(fname, "r"))
    return create_problem(config.ProblemConfig(**obj))
