import numpy as np


def assert_2d_polygon(vertices):
    """Assert that the polygon is 2D with proper shape"""
    vertices = np.asarray(vertices)
    assert vertices.ndim == 2, f"Vertices must be 2D array, got {vertices.ndim}D"
    assert vertices.shape[1] == 2, (
        f"Vertices must have 2 coordinates (x,y), got {vertices.shape[1]}"
    )
    assert vertices.shape[0] >= 3, (
        f"Polygon must have at least 3 vertices, got {vertices.shape[0]}"
    )
    return vertices


def triple_product(a, b, c):
    """Triple product: (a × b) × c for 2D vectors"""
    # In 2D: (a × b) is a scalar (cross product), so we compute (a × b) × c
    ac = np.dot(a, c)
    bc = np.dot(b, c)
    return b * ac - a * bc


def support(shape1, shape2, direction):
    """
    Support function: returns the point on the Minkowski difference
    that is furthest in the given direction
    """
    # Get furthest point in direction for shape1
    p1 = furthest_point(shape1, direction)
    # Get furthest point in opposite direction for shape2
    p2 = furthest_point(shape2, -direction)
    # Return Minkowski difference point
    return p1 - p2


def furthest_point(vertices, direction):
    """Find the vertex of a polygon furthest in the given direction"""
    # Compute dot products for all vertices at once
    dot_products = np.dot(vertices, direction)
    # Return vertex with maximum dot product
    max_index = np.argmax(dot_products)
    return vertices[max_index]


def contains_origin(simplex):
    """
    Check if simplex contains origin and update simplex
    Returns True if origin is contained (collision detected)
    """
    if len(simplex) == 2:
        # Line case - origin cannot be contained in a line
        return False

    elif len(simplex) == 3:
        # Triangle case
        a, b, c = simplex[2], simplex[1], simplex[0]  # a is most recent
        ab = b - a
        ac = c - a
        ao = -a

        # Check which region the origin is in using cross products
        ab_perp = triple_product(ac, ab, ab)
        ac_perp = triple_product(ab, ac, ac)

        if np.dot(ab_perp, ao) > 0:
            # Origin is on AB side - remove c from simplex
            simplex[:] = [b, a]
            return False
        elif np.dot(ac_perp, ao) > 0:
            # Origin is on AC side - remove b from simplex
            simplex[:] = [c, a]
            return False
        else:
            # Origin is inside triangle
            return True

    return False


def get_next_direction(simplex):
    """Get the direction for the next support point"""
    if len(simplex) == 2:
        # Line case
        a, b = simplex[1], simplex[0]  # a is most recent point
        ab = b - a
        ao = -a

        # Get perpendicular to AB towards origin
        ab_perp = triple_product(ab, ao, ab)

        # Normalize direction to avoid numerical issues
        norm = np.linalg.norm(ab_perp)
        if norm > 1e-10:
            return ab_perp / norm
        else:
            return ao / np.linalg.norm(ao)

    # Should not reach here if contains_origin is working correctly
    return np.array([1.0, 0.0])


def gjk(shape1, shape2):
    """
    GJK collision detection algorithm
    Returns True if shapes collide, False otherwise
    """
    # Assert inputs are valid 2D polygons
    shape1 = assert_2d_polygon(shape1)
    shape2 = assert_2d_polygon(shape2)

    # Initial direction (can be arbitrary)
    direction = np.array([1.0, 0.0])

    # Get first support point
    a = support(shape1, shape2, direction)

    # Initialize simplex with first point
    simplex = [a]

    # Next direction is towards origin
    direction = -a

    max_iterations = 20  # Prevent infinite loops

    for _ in range(max_iterations):
        # Get next support point
        a = support(shape1, shape2, direction)

        # If we didn't pass the origin, no collision
        if np.dot(a, direction) < 0:
            return False

        # Add point to simplex
        simplex.append(a)

        # Check if origin is in simplex and update accordingly
        if contains_origin(simplex):
            return True

        # Update direction for next iteration
        direction = get_next_direction(simplex)

    return False
