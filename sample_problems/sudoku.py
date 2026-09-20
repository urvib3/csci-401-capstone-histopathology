# Sudoku as a WCSP [Rishi]
#
# Sudoku: fill a 9x9 grid with digits 1 through 9 so that:
#   1. Every row contains each digit exactly once.
#   2. Every column contains each digit exactly once.
#   3. Every 3x3 box contains each digit exactly once.
#   4. The original clues remain unchanged.
#
# WCSP signature:
#   variables: 81, one variable for each cell
#   domains:   {1, ..., 9} for empty cells
#              a single fixed value for cells containing clues
#   hard constraints:
#       - all-different for every row
#       - all-different for every column
#       - all-different for every 3x3 box
#   cost function:
#       - valid assignments have cost 0
#       - assignments violating a constraint have cost "big"
#
# Sudoku is primarily a constraint-satisfaction problem rather than an
# optimization problem. A CSP is a special case of a WCSP where every
# constraint is hard, so every valid solution has total cost 0.
#
# Local search:
#   Start with a completed grid and repeatedly swap values within rows or
#   boxes to reduce the number of duplicate values. This can become stuck
#   in a local optimum.
#
# A*:
#   A state is a partially completed grid.
#   g can represent the number of assignments made, while h estimates the
#   remaining conflicts. A* is possible, but constraint propagation and
#   backtracking are generally more natural for Sudoku.

import pytoulbar2


def solve_toulbar2(puzzle):
    """
    Solves a 9x9 Sudoku puzzle using Toulbar2.

    puzzle:
        A list containing nine lists of nine integers.
        Use 0 to represent an empty cell.

    returns:
        solved_grid: completed 9x9 Sudoku grid
        cost: total WCSP cost, which should be 0
    """

    # Validate the dimensions of the puzzle.
    if len(puzzle) != 9 or any(len(row) != 9 for row in puzzle):
        raise ValueError("The puzzle must be a 9x9 grid.")

    if any(value < 0 or value > 9 for row in puzzle for value in row):
        raise ValueError("Cell values must be between 0 and 9.")

    # Any assignment with a cost greater than or equal to big is forbidden.
    #
    # Since valid Sudoku assignments have cost 0, big only needs to be
    # greater than 0.
    big = 1000

    # Create the WCSP / Cost Function Network.
    cfn = pytoulbar2.CFN(big)

    # Convert a row and column into the corresponding variable index.
    #
    # (0, 0) -> 0
    # (0, 1) -> 1
    # ...
    # (8, 8) -> 80
    def cell_index(row, col):
        return row * 9 + col

    # Create one variable for every cell.
    for row in range(9):
        for col in range(9):
            value = puzzle[row][col]

            if value == 0:
                # Empty cells can contain any digit from 1 through 9.
                domain = range(1, 10)
            else:
                # A clue has a one-value domain, so Toulbar2 cannot change it.
                domain = [value]

            cfn.AddVariable(f"cell_{row}_{col}", domain)

    # Add an all-different constraint for every row.
    #
    # For example, row 0 contains variables 0 through 8.
    for row in range(9):
        row_variables = [
            cell_index(row, col)
            for col in range(9)
        ]

        cfn.AddGlobalFunction(
            row_variables,
            "salldiff",
            "var",
            big
        )

    # Add an all-different constraint for every column.
    #
    # For example, column 0 contains variables
    # 0, 9, 18, ..., 72.
    for col in range(9):
        column_variables = [
            cell_index(row, col)
            for row in range(9)
        ]

        cfn.AddGlobalFunction(
            column_variables,
            "salldiff",
            "var",
            big
        )

    # Add an all-different constraint for every 3x3 box.
    #
    # box_row and box_col identify the top-left cell of each box:
    # (0,0), (0,3), (0,6), (3,0), ..., (6,6)
    for box_row in range(0, 9, 3):
        for box_col in range(0, 9, 3):

            box_variables = []

            for row_offset in range(3):
                for col_offset in range(3):
                    row = box_row + row_offset
                    col = box_col + col_offset
                    box_variables.append(cell_index(row, col))

            cfn.AddGlobalFunction(
                box_variables,
                "salldiff",
                "var",
                big
            )

    # Find a complete assignment satisfying all constraints.
    result = cfn.Solve()

    if result is None:
        return None, None

    solution, cost, _ = result

    # Convert Toulbar2's one-dimensional solution into a 9x9 grid.
    solved_grid = []

    for row in range(9):
        solved_row = []

        for col in range(9):
            index = cell_index(row, col)
            solved_row.append(int(solution[index]))

        solved_grid.append(solved_row)

    return solved_grid, cost


def print_grid(grid):
    """Prints a Sudoku grid in an easy-to-read format."""

    for row in range(9):

        if row > 0 and row % 3 == 0:
            print("-" * 21)

        for col in range(9):

            if col > 0 and col % 3 == 0:
                print("|", end=" ")

            print(grid[row][col], end=" ")

        print()


if __name__ == "__main__":

    # Use 0 for an empty cell.
    puzzle = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]

    solved_grid, cost = solve_toulbar2(puzzle)

    if solved_grid is None:
        print("No valid solution exists.")
    else:
        print("Toulbar2 Sudoku:")
        print("Total cost:", cost)
        print_grid(solved_grid)


# EXPECTED OUTPUT:
#
# Toulbar2 Sudoku:
# Total cost: 0.0
#
# 5 3 4 | 6 7 8 | 9 1 2
# 6 7 2 | 1 9 5 | 3 4 8
# 1 9 8 | 3 4 2 | 5 6 7
# ---------------------
# 8 5 9 | 7 6 1 | 4 2 3
# 4 2 6 | 8 5 3 | 7 9 1
# 7 1 3 | 9 2 4 | 8 5 6
# ---------------------
# 9 6 1 | 5 3 7 | 2 8 4
# 2 8 7 | 4 1 9 | 6 3 5
# 3 4 5 | 2 8 6 | 1 7 9
