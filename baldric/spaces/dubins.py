import enum
import numpy as np
from numpy.typing import ArrayLike
from dataclasses import dataclass


class SegmentType(enum.Enum):
    L_SEG = 0
    S_SEG = 1
    R_SEG = 2


class DubinsPathType(enum.Enum):
    LSL = 0
    LSR = 1
    RSL = 2
    RSR = 3
    RLR = 4
    LRL = 5


@dataclass
class DubinsIntermediateResults:
    alpha: float
    beta: float
    d: float
    sa: float
    sb: float
    ca: float
    cb: float
    c_ab: float
    d_sq: float

    def __init__(self, d, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.d = d
        self.sa = np.sin(alpha)
        self.ca = np.cos(alpha)
        self.sb = np.sin(beta)
        self.cb = np.cos(beta)
        self.c_ab = np.cos(alpha - beta)
        self.d_sq = d * d


@dataclass
class DubinsPath:
    qi: np.ndarray
    param: np.ndarray
    rho: float
    path_type: DubinsPathType
    intermediate: DubinsIntermediateResults | None

    @property
    def segment_types(self):
        match self.path_type:
            case DubinsPathType.LRL:
                return [SegmentType.L_SEG, SegmentType.R_SEG, SegmentType.L_SEG]
            case DubinsPathType.RLR:
                return [SegmentType.R_SEG, SegmentType.L_SEG, SegmentType.R_SEG]
            case DubinsPathType.LSL:
                return [SegmentType.L_SEG, SegmentType.S_SEG, SegmentType.L_SEG]
            case DubinsPathType.LSR:
                return [SegmentType.L_SEG, SegmentType.S_SEG, SegmentType.R_SEG]
            case DubinsPathType.RSL:
                return [SegmentType.R_SEG, SegmentType.S_SEG, SegmentType.L_SEG]
            case DubinsPathType.RSR:
                return [SegmentType.R_SEG, SegmentType.S_SEG, SegmentType.R_SEG]

    @property
    def length(self):
        return np.sum(self.param) * self.rho


def mod2pi(theta: float):
    return theta % (2 * np.pi)


def dubins_intermediate_results(q0: ArrayLike, q1: ArrayLike, rho: float) -> DubinsIntermediateResults:
    q0 = np.asarray(q0)
    q1 = np.asarray(q1).reshape((3,))
    assert q0.shape == q1.shape
    assert rho > 0.0
    dq = q1 - q0
    D = np.linalg.norm(dq[:2])
    d = D / rho
    theta = 0
    if d > 0:
        theta = mod2pi(np.atan2(dq[1], dq[0]))
    alpha = mod2pi(q0[2] - theta)
    beta = mod2pi(q1[2] - theta)
    return DubinsIntermediateResults(d, alpha, beta)


def shortest_path(qi: ArrayLike, qf: ArrayLike, rho: float) -> DubinsPath:
    qi = np.asarray(qi)
    qf = np.asarray(qf)
    best_cost = float("inf")
    state = dubins_intermediate_results(qi, qf, rho)
    path = None
    for e in DubinsPathType:
        params = dubins_word(state, e)
        if params is None:
            continue
        cost = np.sum(params)
        if cost < best_cost:
            path = DubinsPath(qi, params, rho, e, state)
            best_cost = cost
    return path


def dubins_word(state: DubinsIntermediateResults, word: DubinsPathType):
    funcs = {
        DubinsPathType.LRL: dubins_LRL,
        DubinsPathType.RLR: dubins_RLR,
        DubinsPathType.LSL: dubins_LSL,
        DubinsPathType.LSR: dubins_LSR,
        DubinsPathType.RSR: dubins_RSR,
        DubinsPathType.RSL: dubins_RSL,
    }
    return funcs[word](state)


def dubins_LSL(state: DubinsIntermediateResults):
    tmp0 = state.d + state.sa - state.sb
    p_sq = 2 + state.d_sq - (2 * state.c_ab) + (2 * state.d * (state.sa - state.sb))
    if p_sq >= 0:
        tmp1 = np.atan2((state.cb - state.ca), tmp0)
        return np.array(
            [
                mod2pi(tmp1 - state.alpha),
                np.sqrt(p_sq),
                mod2pi(state.beta - tmp1),
            ]
        )


def dubins_RSR(state: DubinsIntermediateResults):
    tmp0 = state.d - state.sa + state.sb
    p_sq = 2 + state.d_sq - (2 * state.c_ab) + (2 * state.d * (state.sb - state.sa))

    if p_sq >= 0:
        tmp1 = np.atan2((state.ca - state.cb), tmp0)
        return np.array(
            [
                mod2pi(state.alpha - tmp1),
                np.sqrt(p_sq),
                mod2pi(tmp1 - state.beta),
            ]
        )


def dubins_LSR(state: DubinsIntermediateResults):
    p_sq = -2 + (state.d_sq) + (2 * state.c_ab) + (2 * state.d * (state.sa + state.sb))
    if p_sq >= 0:
        p = np.sqrt(p_sq)
        tmp0 = np.atan2((-state.ca - state.cb), (state.d + state.sa + state.sb)) - np.atan2(-2.0, p)
        return np.array([mod2pi(tmp0 - state.alpha), p, mod2pi(tmp0 - mod2pi(state.beta))])


def dubins_RSL(state: DubinsIntermediateResults):
    p_sq = -2 + state.d_sq + (2 * state.c_ab) - (2 * state.d * (state.sa + state.sb))
    if p_sq >= 0:
        p = np.sqrt(p_sq)
        tmp0 = np.atan2((state.ca + state.cb), (state.d - state.sa - state.sb)) - np.atan2(2.0, p)
        return np.array(
            [
                mod2pi(state.alpha - tmp0),
                p,
                mod2pi(state.beta - tmp0),
            ]
        )


def dubins_RLR(state: DubinsIntermediateResults):
    tmp0 = (6.0 - state.d_sq + 2 * state.c_ab + 2 * state.d * (state.sa - state.sb)) / 8.0
    phi = np.atan2(state.ca - state.cb, state.d - state.sa + state.sb)
    if np.abs(tmp0) <= 1:
        p = mod2pi((2 * np.pi) - np.acos(tmp0))
        t = mod2pi(state.alpha - phi + mod2pi(p / 2.0))
        return np.array([t, p, mod2pi(state.alpha - state.beta - t + mod2pi(p))])


def dubins_LRL(state: DubinsIntermediateResults):
    tmp0 = (6.0 - state.d_sq + 2 * state.c_ab + 2 * state.d * (state.sb - state.sa)) / 8.0
    phi = np.atan2(state.ca - state.cb, state.d + state.sa - state.sb)
    if np.abs(tmp0) <= 1:
        p = mod2pi(2 * np.pi - np.acos(tmp0))
        t = mod2pi(-state.alpha - phi + p / 2.0)
        return np.array([t, p, mod2pi(mod2pi(state.beta) - state.alpha - t + mod2pi(p))])


def dubins_segment(t: float, qi: np.ndarray, segment_type: SegmentType):
    st = np.sin(qi[2])
    ct = np.cos(qi[2])
    qt = np.zeros((3,))
    match segment_type:
        case SegmentType.L_SEG:
            qt = np.array([np.sin(qi[2] + t) - st, -np.cos(qi[2] + t) + ct, t])
        case SegmentType.R_SEG:
            qt = np.array([-np.sin(qi[2] - t) + st, np.cos(qi[2] - t) - ct, -t])
        case SegmentType.S_SEG:
            qt = np.array([ct * t, st * t, 0.0])
    return qt + qi


def dubins_path_sample(pth: DubinsPath, t: float):
    # tprime is the normalised variant of the parameter t
    assert t >= 0
    if t > pth.length:
        print("length exceeds", t, pth.length)
        t = pth.length
    tprime = t / pth.rho
    segs = pth.segment_types
    qi = np.array([0, 0, pth.qi[2]])
    p1 = pth.param[0]
    p2 = pth.param[1]
    q1 = dubins_segment(p1, qi, segs[0])
    q2 = dubins_segment(p2, q1, segs[1])
    if tprime < p1:
        q = dubins_segment(tprime, qi, segs[0])
    elif tprime < (p1 + p2):
        q = dubins_segment(tprime - p1, q1, segs[1])
    else:
        q = dubins_segment(tprime - p1 - p2, q2, segs[2])

    # scale the target configuration, translate back to the original starting point
    q[0] = q[0] * pth.rho + pth.qi[0]
    q[1] = q[1] * pth.rho + pth.qi[1]
    q[2] = mod2pi(q[2])
    return q


def dubins_path_sample_many(pth: DubinsPath, stepSize: float):
    x = 0.0
    length = pth.length
    lst = []
    while x < length:
        q = dubins_path_sample(pth, x)
        lst.append(q)
        x += stepSize
    return np.vstack(lst)
