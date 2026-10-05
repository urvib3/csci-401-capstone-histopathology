# Lattice primitives for turning a tile into WCSP-sized pieces.
#
# A 224x224 tile has 50176 pixels; one WCSP variable per pixel is possible
# but large. CellGrid partitions a tile into square cells (e.g. 8x8 pixels
# gives a 28x28 lattice of 784 cells) so that each cell can become one
# variable whose domain is the set of tissue classes. Neighboring cells are
# connected by lattice edges, which are the natural scopes for binary
# smoothness/compatibility cost functions.

import numpy as np

from .classes import IGNORE_INDEX


class CellGrid:
    """
    A rows x cols lattice of square cells covering an H x W image.

    Cells are numbered in row-major order: cell (r, c) has index
    r * cols + c. H and W must be divisible by cell_size.
    """

    def __init__(self, height, width, cell_size):
        if height % cell_size or width % cell_size:
            raise ValueError(
                f"{height}x{width} image is not divisible into {cell_size}px cells."
            )
        self.height = height
        self.width = width
        self.cell_size = cell_size
        self.rows = height // cell_size
        self.cols = width // cell_size

    @property
    def num_cells(self):
        return self.rows * self.cols

    def index(self, r, c):
        return r * self.cols + c

    def position(self, index):
        return divmod(index, self.cols)

    def edges(self, connectivity=4):
        """
        Lattice edges between neighboring cells.

        connectivity:
            4 for horizontal/vertical neighbors, 8 to add diagonals.

        returns:
            int array of shape (num_edges, 2) with i < j in each row.
        """
        if connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8.")

        ids = np.arange(self.num_cells).reshape(self.rows, self.cols)
        pairs = [
            (ids[:, :-1], ids[:, 1:]),  # right
            (ids[:-1, :], ids[1:, :]),  # down
        ]
        if connectivity == 8:
            pairs.append((ids[:-1, :-1], ids[1:, 1:]))  # down-right
            pairs.append((ids[:-1, 1:], ids[1:, :-1]))  # down-left

        edges = np.concatenate(
            [np.stack([a.ravel(), b.ravel()], axis=1) for a, b in pairs]
        )
        return np.sort(edges, axis=1)

    def _blocks(self, array):
        # (H, W, ...) -> (rows, cols, cell_size * cell_size, ...)
        s = self.cell_size
        trailing = array.shape[2:]
        blocks = array.reshape(self.rows, s, self.cols, s, *trailing)
        blocks = np.moveaxis(blocks, 2, 1)
        return blocks.reshape(self.rows, self.cols, s * s, *trailing)

    def label_histogram(self, mask, num_classes, ignore_index=IGNORE_INDEX):
        """
        Count the pixels of each class inside every cell.

        mask:
            (H, W) integer label mask.

        returns:
            int array of shape (rows, cols, num_classes). Pixels equal to
            ignore_index are not counted.
        """
        blocks = self._blocks(np.asarray(mask)).reshape(self.num_cells, -1)
        hist = np.zeros((self.num_cells, num_classes), dtype=np.int64)
        valid = blocks != ignore_index
        cell_ids = np.broadcast_to(
            np.arange(self.num_cells)[:, None], blocks.shape
        )
        np.add.at(hist, (cell_ids[valid], blocks[valid].astype(np.int64)), 1)
        return hist.reshape(self.rows, self.cols, num_classes)

    def majority_labels(self, mask, num_classes, ignore_index=IGNORE_INDEX,
                        min_fraction=0.0):
        """
        Reduce a pixel mask to one label per cell by majority vote.

        min_fraction:
            A cell whose labeled pixels make up less than this fraction of
            the cell is marked ignore_index (e.g. mostly outside the ROI).

        returns:
            (rows, cols) array of cell labels.
        """
        hist = self.label_histogram(mask, num_classes, ignore_index)
        labeled = hist.sum(axis=2)
        labels = hist.argmax(axis=2).astype(np.int64)
        empty = labeled <= min_fraction * self.cell_size ** 2
        labels[empty | (labeled == 0)] = ignore_index
        return labels

    def pool_probabilities(self, probs):
        """
        Average per-pixel class probabilities over each cell.

        probs:
            (num_classes, H, W) array, e.g. softmax output of a pixel model.

        returns:
            (rows, cols, num_classes) array of mean probabilities.
        """
        probs = np.moveaxis(np.asarray(probs), 0, -1)
        return self._blocks(probs).mean(axis=2)

    def upsample(self, cell_labels):
        """Expand (rows, cols) cell labels back to an (H, W) pixel mask."""
        s = self.cell_size
        return np.repeat(np.repeat(np.asarray(cell_labels), s, axis=0), s, axis=1)


def unary_costs_from_probabilities(probs, eps=1e-6):
    """
    Negative log-likelihood unary costs, one row per WCSP variable.

    probs:
        (..., num_classes) probabilities.

    returns:
        float array of shape (num_variables, num_classes).
    """
    probs = np.asarray(probs, dtype=np.float64)
    return -np.log(np.clip(probs, eps, 1.0)).reshape(-1, probs.shape[-1])


def to_integer_costs(costs, scale=1000, axis=None):
    """
    Scale float costs to non-negative integers for solvers such as Toulbar2.

    Costs are shifted so their minimum is 0 before scaling; shifting a whole
    cost function by a constant does not change which assignment is optimal.

    axis:
        None treats the array as one cost table (use this for a binary
        K x K table). axis=-1 treats each row as its own table (use this for
        a stacked (num_variables, K) array of unary costs).
    """
    costs = np.asarray(costs, dtype=np.float64)
    shifted = costs - costs.min(axis=axis, keepdims=True)
    return np.rint(shifted * scale).astype(np.int64)
