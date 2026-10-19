"""The energy from the project notes: neighbors, cost tables, unary and pairwise terms."""

from .neighborhood import distance_weights, forward_offsets, neighbor_type, pair_slices
from .pairwise import color_features, contrast_term, instance_term
from .problem import GridProblem, build_problem
from .tables import HANDWRITTEN, POTTS, load_table, save_table
from .unary import calibrate, unary_costs
