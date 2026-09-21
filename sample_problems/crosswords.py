# Crosswords as a WCSP [Kashvi]
#
# Crossword: assign one word to each horizontal or vertical slot so that letters
# at every intersection match
#
# WCSP signature:
#   variables:  one variable for each word slot
#   domains:    candidate words with the same length as the slot
#   hard constraint:    intersecting words must have the same letter at their shared cell
#   cost function:    assigning a preferred word can have a lower cost
#   arity:      unary word costs + binary intersection constraints
#   treewidth:  depends on the crossword's slot-intersection graph
#
# Local search: start with one word in each slot. Replace a word with another
#   candidate word and keep the change if it reduces the number of conflicting letters.
#
# A*: fill the crossword one slot at a time.
#   g = number or cost of conflicts created so far
#   h = optimistic estimate of the conflicts remaining in unfilled slots

import pytoulbar2

# Small 2x2 crossword:
#
#     A T
#     N O
#
# Across slots: A1 = AT, A2 = NO
# Down slots:   D1 = AN, D2 = TO

slots = ["A1", "A2", "D1", "D2"]

# Candidate words for each slot.
words = {
    "A1": ["AT", "IN"],
    "A2": ["NO", "UP"],
    "D1": ["AN", "IU"],
    "D2": ["TO", "NP"],
}

# Each intersection is:
# (first slot, letter position, second slot, letter position)
intersections = [
    ("A1", 0, "D1", 0),
    ("A1", 1, "D2", 0),
    ("A2", 0, "D1", 1),
    ("A2", 1, "D2", 1),
]


def solve_toulbar2(slots, words, intersections):
    # Cost used for an invalid letter combination.
    big = 100

    # Create the WCSP / Cost Function Network.
    cfn = pytoulbar2.CFN(big)

    # Map each slot name to its variable index.
    slot_index = {slot: i for i, slot in enumerate(slots)}

    # Create one variable for each crossword slot.
    #
    # The value assigned to a variable is the index of the chosen
    # word in that slot's candidate-word list.
    for slot in slots:
        cfn.AddVariable(slot, range(len(words[slot])))

    # Add one binary hard constraint for each intersection.
    for slot1, pos1, slot2, pos2 in intersections:
        costs = []

        # A pair of words has cost 0 if its intersecting letters match.
        # Otherwise, it receives the invalid cost.
        for word1 in words[slot1]:
            for word2 in words[slot2]:
                if word1[pos1] == word2[pos2]:
                    costs.append(0)
                else:
                    costs.append(big)

        cfn.AddFunction(
            [slot_index[slot1], slot_index[slot2]],
            costs
        )

    # Ask Toulbar2 to find a valid crossword assignment.
    solution, cost, _ = cfn.Solve()

    return solution, cost

def main():
    # Solve the example crossword.
        solution, cost = solve_toulbar2(slots, words, intersections)
     
        print("toulbar2 Crossword:")
        print("total cost:", cost)
     
        # Convert each numeric value back into its chosen word.
        for i, slot in enumerate(slots):
            word_index = int(solution[i])
            print(f"{slot} -> {words[slot][word_index]}")


if __name__ == "__main__":
    main()


# EXAMPLE OUTPUT:

# toulbar2 Crossword:
# total cost: 0.0
# A1 -> AT
# A2 -> NO
# D1 -> AN
# D2 -> TO