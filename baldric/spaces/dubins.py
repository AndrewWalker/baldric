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

    @property
    def length(self):
        return np.sum(self.param) * self.rho


def mod2pi(theta: float):
    return theta % (2 * np.pi)


def dubins_intermediate_results(
    q0: ArrayLike, q1: ArrayLike, rho: float
) -> DubinsIntermediateResults:
    q0 = np.asarray(q0)
    q1 = np.asarray(q1)
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
            path = DubinsPath(qi, params, rho, e)
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
    tmp0 = state.d - state.sa - state.sb
    p_sq = 2 + state.d_sq - (2 * state.c_ab) + (2 * state.d * (state.sa - state.sb))

    if p_sq >= 0:
        tmp1 = np.atan2((state.cb - state.ca), tmp0)
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
        tmp0 = np.atan2(
            (-state.ca - state.cb), (state.d + state.sa + state.sb)
        ) - np.atan2(-2.0, p)
        return np.array(
            [mod2pi(tmp0 - state.alpha), p, mod2pi(tmp0 - mod2pi(state.beta))]
        )


def dubins_RSL(state: DubinsIntermediateResults):
    p_sq = -2 + state.d_sq + (2 * state.c_ab) - (2 * state.d * (state.sa + state.sb))
    if p_sq >= 0:
        p = np.sqrt(p_sq)
        tmp0 = np.atan2(
            (state.ca + state.cb), (state.d - state.sa - state.sb)
        ) - np.atan2(2.0, p)
        return np.array(
            [
                mod2pi(state.alpha - tmp0),
                p,
                mod2pi(state.beta - tmp0),
            ]
        )


def dubins_RLR(state: DubinsIntermediateResults):
    tmp0 = (
        6.0 - state.d_sq + 2 * state.c_ab + 2 * state.d * (state.sa - state.sb)
    ) / 8.0
    phi = np.atan2(state.ca - state.cb, state.d - state.sa + state.sb)
    if np.abs(tmp0) <= 1:
        p = mod2pi((2 * np.pi) - np.acos(tmp0))
        t = mod2pi(state.alpha - phi + mod2pi(p / 2.0))
        return np.array([t, p, mod2pi(state.alpha - state.beta - t + mod2pi(p))])


def dubins_LRL(state: DubinsIntermediateResults):
    tmp0 = (
        6.0 - state.d_sq + 2 * state.c_ab + 2 * state.d * (state.sb - state.sa)
    ) / 8.0
    phi = np.atan2(state.ca - state.cb, state.d + state.sa - state.sb)
    if np.abs(tmp0) <= 1:
        p = mod2pi(2 * np.pi - np.acos(tmp0))
        t = mod2pi(-state.alpha - phi + p / 2.0)
        return np.array(
            [t, p, mod2pi(mod2pi(state.beta) - state.alpha - t + mod2pi(p))]
        )


# double dubins_segment_length( DubinsPath* path, int i )
# {
#     if( (i < 0) || (i > 2) )
#     {
#         return INFINITY;
#     }
#     return path.param[i] * path.rho;
# }

# double dubins_segment_length_normalized( DubinsPath* path, int i )
# {
#     if( (i < 0) || (i > 2) )
#     {
#         return INFINITY;
#     }
#     return path.param[i];
# }

# DubinsPathType dubins_path_type( DubinsPath* path )
# {
#     return path.type;
# }

# void dubins_segment( double t, double qi[3], double qt[3], SegmentType type)
# {
#     double st = sin(qi[2]);
#     double ct = cos(qi[2]);
#     if( type == L_SEG ) {
#         qt[0] = +sin(qi[2]+t) - st;
#         qt[1] = -cos(qi[2]+t) + ct;
#         qt[2] = t;
#     }
#     else if( type == R_SEG ) {
#         qt[0] = -sin(qi[2]-t) + st;
#         qt[1] = +cos(qi[2]-t) - ct;
#         qt[2] = -t;
#     }
#     else if( type == S_SEG ) {
#         qt[0] = ct * t;
#         qt[1] = st * t;
#         qt[2] = 0.0;
#     }
#     qt[0] += qi[0];
#     qt[1] += qi[1];
#     qt[2] += qi[2];
# }

# int dubins_path_sample( DubinsPath* path, double t, double q[3] )
# {
#     /* tprime is the normalised variant of the parameter t */
#     double tprime = t / path.rho;
#     double qi[3]; /* The translated initial configuration */
#     double q1[3]; /* end-of segment 1 */
#     double q2[3]; /* end-of segment 2 */
#     const SegmentType* types = DIRDATA[path.type];
#     double p1, p2;

#     if( t < 0 || t > dubins_path_length(path) ) {
#         return EDUBPARAM;
#     }

#     /* initial configuration */
#     qi[0] = 0.0;
#     qi[1] = 0.0;
#     qi[2] = path.qi[2];

#     /* generate the target configuration */
#     p1 = path.param[0];
#     p2 = path.param[1];
#     dubins_segment( p1,      qi,    q1, types[0] );
#     dubins_segment( p2,      q1,    q2, types[1] );
#     if( tprime < p1 ) {
#         dubins_segment( tprime, qi, q, types[0] );
#     }
#     else if( tprime < (p1+p2) ) {
#         dubins_segment( tprime-p1, q1, q,  types[1] );
#     }
#     else {
#         dubins_segment( tprime-p1-p2, q2, q,  types[2] );
#     }

#     /* scale the target configuration, translate back to the original starting point */
#     q[0] = q[0] * path.rho + path.qi[0];
#     q[1] = q[1] * path.rho + path.qi[1];
#     q[2] = mod2pi(q[2]);

#     return EDUBOK;
# }

# int dubins_path_sample_many(DubinsPath* path, double stepSize,
#                             DubinsPathSamplingCallback cb, void* user_data)
# {
#     int retcode;
#     double q[3];
#     double x = 0.0;
#     double length = dubins_path_length(path);
#     while( x <  length ) {
#         dubins_path_sample( path, x, q );
#         retcode = cb(q, x, user_data);
#         if( retcode != 0 ) {
#             return retcode;
#         }
#         x += stepSize;
#     }
#     return 0;
# }

# int dubins_path_endpoint( DubinsPath* path, double q[3] )
# {
#     return dubins_path_sample( path, dubins_path_length(path) - EPSILON, q );
# }

# int dubins_extract_subpath( DubinsPath* path, double t, DubinsPath* newpath )
# {
#     /* calculate the true parameter */
#     double tprime = t / path.rho;

#     if((t < 0) || (t > dubins_path_length(path)))
#     {
#         return EDUBPARAM;
#     }

#     /* copy most of the data */
#     newpath.qi[0] = path.qi[0];
#     newpath.qi[1] = path.qi[1];
#     newpath.qi[2] = path.qi[2];
#     newpath.rho   = path.rho;
#     newpath.type  = path.type;

#     /* fix the parameters */
#     newpath.param[0] = fmin( path.param[0], tprime );
#     newpath.param[1] = fmin( path.param[1], tprime - newpath.param[0]);
#     newpath.param[2] = fmin( path.param[2], tprime - newpath.param[0] - newpath.param[1]);
#     return 0;
# }
