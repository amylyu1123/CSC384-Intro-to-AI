#Look for #IMPLEMENT tags in this file.
'''
All models need to return a CSP object, and a list of lists of Variable objects
representing the board. The returned list of lists is used to access the
solution.

For example, after these three lines of code

    csp, var_array = caged_csp_model(board)
    solver = BT(csp)
    solver.bt_search(prop_FC, var_ord)

var_array[0][0].get_assigned_value() should be the correct value in the top left
cell of the FunPuzz puzzle.

The grid-only models do not need to encode the cage constraints.

1. binary_ne_grid (worth 10/100 marks)
    - A model of a FunPuzz grid (without cage constraints) built using only
      binary not-equal constraints for both the row and column constraints.

2. nary_ad_grid (worth 10/100 marks)
    - A model of a FunPuzz grid (without cage constraints) built using only n-ary
      all-different constraints for both the row and column constraints.

3. caged_csp_model (worth 25/100 marks)
    - A model built using your choice of (1) binary binary not-equal, or (2)
      n-ary all-different constraints for the grid.
    - Together with FunPuzz cage constraints.

'''
from cspbase import *
import itertools

def binary_ne_grid(fpuzz_grid):
    N = fpuzz_grid[0][0]
    var_array = []

    dom = []
    for i in range(N):
        dom.append(i + 1)

    vars = []
    for i in range(N):
        row = []
        for j in range(N):
            var = Variable("V{}{}".format(i + 1, j + 1), dom)
            row.append(var)
            vars.append(var)
        var_array.append(row)

    sat_tuples = []
    for t in itertools.product(dom, dom):
        if t[0] != t[1]:
            sat_tuples.append(t)

    csp = CSP("binary not-equal", vars)
    for i in range(N):
        for j in range(N):
            for k in range(j + 1, N):
                # row
                scope = [var_array[i][j], var_array[i][k]]
                con_row = Constraint("C{}-{}".format(scope[0].name,
                                                     scope[1].name), scope)
                con_row.add_satisfying_tuples(sat_tuples)
                csp.add_constraint(con_row)
            for k in range(i + 1, N):
                # col
                scope = [var_array[i][j], var_array[k][j]]
                con_col = Constraint("C{}-{}".format(scope[0].name,
                                                     scope[1].name), scope)

                con_col.add_satisfying_tuples(sat_tuples)
                csp.add_constraint(con_col)

    return csp, var_array

def nary_ad_grid(fpuzz_grid):
    N = fpuzz_grid[0][0]
    var_array = []

    dom = []
    for i in range(N):
        dom.append(i + 1)

    vars = []
    for i in range(N):
        row = []
        for j in range(N):
            var = Variable("V{}{}".format(i + 1, j + 1), dom)
            row.append(var)
            vars.append(var)
        var_array.append(row)

    varDoms = []
    for item in range(N):
        varDoms.append(dom)

    sat_tuples = []
    for t in itertools.product(*varDoms):
        satisfied = True
        for i in range(N):
            for j in range(i + 1, N):
                if t[i] == t[j]:
                    satisfied = False
                    break
        if satisfied:
            sat_tuples.append(t)

    csp = CSP("n-ary all-different", vars)
    for i in range(N):
        scope_row = []
        scope_col = []
        for j in range(N):
            scope_row.append(var_array[i][j])
            scope_col.append(var_array[j][i])

        con_row = Constraint("C-r{}".format(i + 1), scope_row)
        con_row.add_satisfying_tuples(sat_tuples)
        csp.add_constraint(con_row)

        con_col = Constraint("C-c{}".format(i + 1), scope_col)
        con_col.add_satisfying_tuples(sat_tuples)
        csp.add_constraint(con_col)

    return csp, var_array


def caged_csp_model(fpuzz_grid):
    # use binary not-equal constraint for the grid
    csp, var_array = binary_ne_grid(fpuzz_grid)

    N = fpuzz_grid[0][0]
    csp.name = "cage FunPuzz"

    for i in range(1, len(fpuzz_grid)):
        # get current cage info
        curr = fpuzz_grid[i]

        # if there are only two elements
        if len(curr) == 2:
            row = curr[0] // 10
            col = curr[0] % 10
            con = Constraint("C-{}{}".format(row, col), [var_array[row-1][col-1]])
            con.add_satisfying_tuples([(curr[1],)])
            csp.add_constraint(con)

        # more than two elements
        else:
            curr_vars = []
            first_row, first_col = None, None  # record the row, column number
            for j in range(len(curr) - 2):
                row = curr[j] // 10
                col = curr[j] % 10
                if j == 0:
                    first_row = row
                    first_col = col
                curr_vars.append(var_array[row-1][col-1])

            # call get_sat_tuple to get all satisfied tuples for this constraint
            sat_tuples = get_sat_tuples(curr_vars, curr[len(curr)-2], curr[len(curr)-1], N)
            con = Constraint("C-{}{}".format(first_row, first_col), curr_vars)
            con.add_satisfying_tuples(sat_tuples)
            csp.add_constraint(con)

    return csp, var_array


# get all satisfied tuples involving vars to get target value under operation
# for grid with length N
def get_sat_tuples(vars, target, operation, N):
    dom = []
    for i in range(N):
        dom.append(i + 1)

    varDoms = []
    for con in range(len(vars)):
        varDoms.append(dom)

    sat_tuples = []
    for t in itertools.product(*varDoms):
        result = None
        # if the operation is addition or multiplication
        if operation == 0 or operation == 3:
            for i in range(len(t)):
                # +
                if operation == 0:
                    if result is None:
                        result = 0
                    result += t[i]
                # *
                elif operation == 3:
                    if result is None:
                        result = 1
                    result *= t[i]
            if result == target:
                sat_tuples.append(t)
        # if the operation is subtraction or division
        else:
            # find all permutations of t
            permutations = itertools.permutations(t)
            for perm in permutations:
                result = None
                for i in range(len(perm)):
                    # -
                    if operation == 1:
                        if result is None:
                            result = perm[i]
                        else:
                            result -= perm[i]
                    # *
                    elif operation == 2:
                        if result is None:
                            result = perm[i]
                        else:
                            result /= perm[i]
                # if any of the permutation evaluates to target, break the loop
                if result == target:
                    sat_tuples.append(t)
                    break
    return sat_tuples
