"""Cut surface outflow from an independently documented closed basin in D8."""

import numpy as np

D8_STEPS = ((1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1))


def close_boundary(direction, inside, nodata=255):
    """Terminate only outgoing edges; preserve other D8 edges and existing sinks.

    No artificial reverse edges or elevated DEM walls are introduced. External
    cells draining into the closed basin also stop contributing downstream;
    callers must report that additional catchment change. This is a constrained
    graph correction, not a reconstruction of internal basin hydrology.
    """
    direction, inside = np.asarray(direction), np.asarray(inside, dtype=bool)
    if direction.ndim != 2 or direction.shape != inside.shape or not inside.any():
        raise ValueError("Expected aligned nonempty closed basin and D8 arrays")
    if nodata in range(8) or not np.isin(direction, [*range(8), nodata]).all():
        raise ValueError("Invalid D8 codes or terminal value")
    result = direction.copy()
    rows, columns = direction.shape
    counts = []
    for code, (dx, dy) in enumerate(D8_STEPS):
        yy, xx = np.nonzero(inside & (direction == code))
        ny, nx = yy + dy, xx + dx
        valid = (ny >= 0) & (ny < rows) & (nx >= 0) & (nx < columns)
        stays = np.zeros(len(yy), dtype=bool)
        stays[valid] = inside[ny[valid], nx[valid]]
        result[yy[~stays], xx[~stays]] = nodata
        counts.append(int((~stays).sum()))
    return result, counts
