"""
This version has passed some sucesfull tests.


The simplex_method function solves the problem
****************
((obj_fuc)^T @ x) -> min

subject to
A @ x = b
****************

The integrality is observed.
All data and variables MUST be eighet Fractions or integers

The simplex_method function accepts the
1) objectibve function: obj_fun
2) correct and verified initial basic feasible solution: init_basic_feas_sol
3) the list of basic variables in relation to initial basic feasible solution: bas_var
4) the list of non-basic variables in relation to initial basic feasible solution: non_bas_var
5) the simplex matrix A: A_eq
6) the right-hand-side of the equations b: b_eq


The simplex_method function returns the
1) optimal solution to the problem: x,
2) value of the objective function at optimum: F_value,
3) list of basic variables: basis,
4) list of non-basic variables: non_basis

"""

from fractions import Fraction
import numpy as np

def simplex_method(obj_fun, init_basic_feas_sol, bas_var, non_bas_var, A_eq, b_eq):
    c, x, A, b, basis, non_basis, F_value = valid_copies_of_input(obj_fun, init_basic_feas_sol, bas_var, non_bas_var, A_eq, b_eq)
    A, b, diagonal_elements = express_A_via_nonbasic(A, b, basis)
    c, x, A, b, basis, non_basis, F_value = express_ObjFunc_via_nonbasic(c, x, A, b, basis, non_basis, F_value, diagonal_elements)
    
    min_c_coef=np.amin(c)
    while min_c_coef<0:
        leading_colomn=get_index_of_element(c, min_c_coef)
        elements_lc=A[:, leading_colomn]
        leading_row=get_leading_row(elements_lc, b)
        
        inserted_item=leading_colomn # inserted into basis 
        removed_item=get_removed_item(diagonal_elements, leading_row) # removed from basis 
        
        x=update_solution(A, leading_colomn, leading_row, b, x, diagonal_elements)
        
        basis.append(inserted_item)
        index = basis.index(removed_item)
        basis.pop(index)
        basis.sort()
        
        non_basis.append(removed_item)
        index = non_basis.index(inserted_item)
        non_basis.pop(index)
        non_basis.sort()
        
        A, b, diagonal_elements = express_A_via_nonbasic(A, b, basis)
        c, x, A, b, basis, non_basis, F_value = express_ObjFunc_via_nonbasic(c, x, A, b, basis, non_basis, F_value, diagonal_elements)
        
        
        min_c_coef=np.amin(c)
    
    #the objective function is minimized to optimality
        
    
    
    return x, F_value, basis, non_basis

def update_solution(A, leading_colomn, leading_row, b, x, diagonal_elements):
    delta=b[leading_row]/A[leading_row][leading_colomn]
    x[leading_colomn]=delta
    
    for row, col in diagonal_elements:
        #if col!=leading_colomn:
        change=-delta*A[row][leading_colomn]/A[row][col]
        x[col]=x[col]+change
    
    return x


def get_removed_item(diagonal_elements, leading_row):
    for row, col in diagonal_elements:
        if row==leading_row:
            removed_item=col
            break
    return removed_item




def get_leading_row(elements_lc, b):
    height=np.size(elements_lc)
    for i in range(height):
        e=elements_lc[i]
        if e>Fraction(0, 1):
            relation=b[i]/e
            idx=i
            break
    for i in range(height):
        e=elements_lc[i]
        if e>Fraction(0, 1):
            if (b[i]/e)<relation:
                relation=b[i]/e
                idx=i
    return idx


def get_index_of_element(c, min_coef):
    tmp_arr = np.where(c == min_coef)
    idx=tmp_arr[0][0]
    return idx
    



def express_ObjFunc_via_nonbasic(c, x, A, b, basis, non_basis, F_value, diagonal_elements):
    A_height, A_length=np.shape(A)
    for row, col in diagonal_elements:
        leading_element=A[row][col]
        coef=c[col]/leading_element
        if coef!=0:
            for j in range(A_length):
                c[j]=c[j]-coef*A[row][j]
        F_value=F_value+coef*b[row]
        
    
    
    return c, x, A, b, basis, non_basis, F_value


def valid_copies_of_input(obj_fun, init_basic_feas_sol, bas_var, non_bas_var, A_eq, b_eq):
    """
    This function
    1) creates the copies of all input structures
    (to keep original input object protected against changes that happen while the itterations of simplex method).
    
    2) It ensures that all the structures use Fractions-type elements as their foundation.
    
    3) Further, it validates the sizes and shapes of the input structures to meet the general requirements of the classical linear problem
    *****
    c @ x -> min
    s.t.
    A @ x = b
    ******
    
    4) It verifies that the init_basic_feas_sol is valid for Ax=b.
    
    5) It verifies that init_basic_feas_sol is coherent with non_basis. (That is, all the non-basic variables are zeros)
    
    """
    c=obj_fun.copy()
    x=init_basic_feas_sol.copy()
    #F_value=np.vdot(c, x)
    F_value=Fraction(0, 1)
    A=A_eq.copy()
    b=b_eq.copy()
    basis=bas_var.copy()
    basis.sort()
    non_basis=non_bas_var.copy()
    non_basis.sort()
    
    c_length=np.size(c)
    A_height, A_length=np.shape(A)
    x_length=np.size(x)
    basis_lenght=np.size(basis)
    non_basis_lenght=np.size(non_basis)
    b_length=np.size(b)
    
    check_dimensions_coherence(c, x, A, b, basis, non_basis)
    
    if not check_Fraction_or_int_type_of_vector(c, c_length):
        raise ValueError('The type of the objective function is neither integer nor Fraction')
    if not check_Fraction_or_int_type_of_vector(x, x_length):
        raise ValueError('The type of the initial feasible basic solution is neither integer nor Fraction')
    if not check_Fraction_or_int_type_of_vector(b, b_length):
        raise ValueError('The type of the RHS (right-hand-side) values is neither integer nor Fraction')
    if not check_Fraction_or_int_type_of_matrix(A, A_height, A_length):
        raise ValueError('The type of the Simplex matrix values is neither integer nor Fraction')
    
    validate_bfs(A, x, b)
    vaildate_nonbasis(x, non_basis)
    return c, x, A, b, basis, non_basis, F_value


def express_A_via_nonbasic(A, b, basis):
    rows_to_go=[i for i in range(np.size(b))]
    colomns_to_go=basis.copy()
    A, b, diagonal_elements = gauss_direct_move(A, b, rows_to_go, colomns_to_go)
    A, b = gauss_reverse_move(A, b, rows_to_go, diagonal_elements)
    return A, b, diagonal_elements

def gauss_reverse_move(A, b, rows_to_go, diagonal_elements):
    A_height, A_length=np.shape(A)
    
    # The list of tuples (diagonal_elements) is sorted in a manner that the coloms from the right-hand side of the simplex martix
    # will appear first. Thus, the iteration process over the diagonal_elements will proceed from the right to the left.
    # It is a pre-requisite for the correct execution of the gauss_reverse_move function
    
    # iterating over the diagonal_elements means that we go from the right-hand side of the simplex table to the left
    for row, col in diagonal_elements:
        leading_element=A[row][col]
        idx = rows_to_go.index(row)
        rows_to_go.pop(idx)
        for r in rows_to_go:
            coef=A[r][col]/leading_element
            if coef!=Fraction(0, 1):
                b[r]=(b[r]-coef*b[row])
                for j in range(A_length):
                    A[r][j]=(A[r][j]-coef*A[row][j])
    
    # this (below) FOR cycle actually makes sure that the simplex matrix is exactly identity matrix (E)
    # over the colomns where the basic elements appear
    for row, col in diagonal_elements:
        leading_element=A[row][col]
        if leading_element!=Fraction(1, 1):
            b[row]=b[row]/leading_element
            for j in range(A_length):
                A[row][j]=A[row][j]/leading_element
        
    
    return A, b


def gauss_direct_move(A, b, rotogo, cotogo):
    rows_to_go=rotogo.copy()
    colomns_to_go=cotogo.copy()
    A_height, A_length=np.shape(A)
    diagonal_elements=[]
    while len(colomns_to_go)>0:
        current_col=colomns_to_go.pop(0)
        # colomns_to_go is a sorted list. Thefore, we go from the left-hand-side
        # towards the right-hand side of the simplex table
        
        current_row_id=0 #We need to iterate over the rows_to_go list to retrieve the non-zero element in the current colomn.
        leading_element=A[rows_to_go[current_row_id], current_col]
        while ((leading_element==Fraction(0, 1)) and (current_row_id<(len(rows_to_go)-1))):
            current_row_id+=1
            leading_element=A[rows_to_go[current_row_id], current_col]
        if ((leading_element==Fraction(0, 1)) and (current_row_id>=(np.size(b)-1))):
            raise ValueError('A matrix appears to be singular. Colomn entirely full of zeros. report from gauss_direct_move')
        # At this point the leading_element is non-zero
        # current_col and rows_to_go[current_row_id] represent the indices where it was found
        
        # Now for each row in rows_to_go, except the current_row
        # we need to calculate the coefficient suitable to zero the the element in the current colomn
        # and then we need to zero the the element in the current colomn
        if (leading_element!=Fraction(0, 1)):
            diagonal_elements.append((rows_to_go[current_row_id], current_col))
        
        for row in rows_to_go:
            if row!=rows_to_go[current_row_id]:
                coef=A[row][current_col]/leading_element
                if coef!=Fraction(0, 1):
                    b[row]=b[row]-coef*b[rows_to_go[current_row_id]]
                    for j in range(A_length):
                        A[row][j]=A[row][j]-coef*A[rows_to_go[current_row_id]][j]                    

        rows_to_go.pop(current_row_id)
        # We remove the current row from the rows_to_go.
        
    custom_sort_list_of_tuples(diagonal_elements, 1, reverse=True)
    # This kind of sorting ensures that the coloms from the right-hand side of the simplex martix
    # will appear first. Thus, the iteration process over the diagonal_elements will proceed from the right to the left.
    # It is a pre-requisite for the correct execution of the gauss_reverse_move function
    return A, b, diagonal_elements


def validate_bfs(A, x, b):
    A_height, A_length=np.shape(A)
    for i in range(A_height):
        z=np.vdot(A[i], x)
        if z!=b[i]:
            raise ValueError('BFS is invalid. It does not fit into A@x=b equation')

    return

def vaildate_nonbasis(x, non_basis):
    non_basis_lenght=np.size(non_basis)
    for i in range(non_basis_lenght):
        if x[non_basis[i]]!=0:
            raise ValueError('non-basic variable ', i, ' does not equal to zero')
    return
    



def check_dimensions_coherence(c, x, A, b, basis, non_basis):
    c_length=np.size(c)
    A_height, A_length=np.shape(A)
    x_length=np.size(x)
    basis_lenght=np.size(basis)
    non_basis_lenght=np.size(non_basis)
    b_length=np.size(b)

    if A_length!=x_length:
        raise ValueError('The dimensions of initial feasible solution and the matrix do not match')
    if A_height!=basis_lenght:
        raise ValueError('The dimensions of the basis and the matrix do not match')
    if A_length!=(basis_lenght+non_basis_lenght):
        raise ValueError('The dimensions of the non_basis and the matrix do not match')
    if A_height!=b_length:
        raise ValueError('The dimensions of the RHS (b) and the matrix do not match')
    return
    


def check_Fraction_or_int_type_of_vector(vector, size):
    for i in range(size):
        validated=isinstance(vector[i], Fraction)
        validated=(validated or (isinstance(vector[i], int)))
        if not validated:
            return validated # which should be False
    return validated # which should be True
        

def check_Fraction_or_int_type_of_matrix(matrix, height, length):
    for i in range(height):
        for j in range(length):
            validated=isinstance(matrix[i][j], Fraction)
            validated=(validated or (isinstance(matrix[i][j], int)))
            if not validated:
                return validated # which should be False
    return validated # which should be True
    
    
    


def custom_sort_list_of_tuples(List, i, reverse=False):
    # if input is
    # List=[("Alice", 25), ("Bob", 20), ("Alex", 5)]
    # i=1
    # ...
    # then the output is
    # [('Alex', 5), ('Bob', 20), ('Alice', 25)]

    if reverse:
        List.sort(key=lambda x: x[i], reverse=True)
    else:
        List.sort(key=lambda x: x[i])
    return(List)




if __name__=="__main__":

    #obj_fun=np.array([Fraction(-3, 1), Fraction(-1, 1), Fraction(-7, 1), Fraction(4, 1), Fraction (5, 1), Fraction(-19, 1)])
    obj_fun=np.array([Fraction(3, 1), Fraction(5, 1), Fraction(6, 1), Fraction(8, 1), Fraction(0, 1), Fraction(-4, 1), Fraction(-3, 1), Fraction(19, 1), Fraction(-17, 1)])
    #This is objective function. It has to be minimized.
    
    #A_eq=np.array([
        #[Fraction(1, 1),  Fraction(-2, 1),  Fraction(0, 1), Fraction(1, 1), Fraction(1, 1), Fraction(0, 1)],
        #[Fraction(-1, 1), Fraction(1, 1), Fraction(2, 1), Fraction(3, 1), Fraction(2, 1), Fraction(1, 1)],
        #[Fraction(0, 1), Fraction(9, 1), Fraction(1, 1), Fraction(1, 1), Fraction(6, 1), Fraction(0, 1)]])
        
    A_eq=np.array([
        [Fraction(4, 1), Fraction(-1, 1), Fraction(0, 1), Fraction(4, 1), Fraction(-78, 1), Fraction(32, 1), Fraction(21, 1), Fraction(31, 1), Fraction(2, 1)],
        [Fraction(6, 1), Fraction(6, 1), Fraction(13, 1), Fraction(5, 1), Fraction(78, 1), Fraction(41, 1), Fraction(0, 1), Fraction(2, 1), Fraction(3, 1)],
        [Fraction(-17, 1), Fraction(2, 1), Fraction(16, 1), Fraction(-8, 1), Fraction(2, 1), Fraction(-22, 1), Fraction(32, 1), Fraction(2, 1), Fraction(4, 1)],
        [Fraction(9, 1), Fraction(6, 1), Fraction(3, 1), Fraction(2, 1), Fraction(-33, 1), Fraction(23, 1), Fraction(0, 1), Fraction(7, 1), Fraction(5, 1)],
        [Fraction(8, 1), Fraction(7, 1), Fraction(1, 1), Fraction(-18, 1), Fraction(11, 1), Fraction(3, 1), Fraction(1, 1), Fraction(45, 1), Fraction(6, 1)]])
        
        
        
    
    #b_eq=np.array([Fraction(10, 1), Fraction(10, 1), Fraction(26, 1)])
    b_eq=np.array([Fraction(-14, 1), Fraction(112, 1), Fraction(18, 1), Fraction(-2, 1), Fraction(98, 1)])
    
    #basis=[5, 1, 0]
    #basis=[0, 3, 4]
    basis=[1, 2, 3, 4, 7]
    
    #non_basis=[4, 3, 2]
    #non_basis=[1, 2, 5]
    non_basis=[0, 5, 6, 8]
    
    #init_basic_feas_sol=np.array([Fraction(142, 9), Fraction(26, 9), Fraction(0, 1), Fraction(0, 1), Fraction(0, 1), Fraction(206, 9)])
    #init_basic_feas_sol=np.array([Fraction(4, 1), Fraction(0, 1), Fraction(0, 1), Fraction(2, 1), Fraction(4, 1), Fraction(0, 1)])
    init_basic_feas_sol=np.array([Fraction(0, 1), Fraction(2, 1), Fraction(1, 1), Fraction(1, 1), Fraction(1, 1), Fraction(0, 1), Fraction(0, 1), Fraction(2, 1), Fraction(0, 1)])
    
    x, TFval, bas_var, non_bas_var=simplex_method(obj_fun, init_basic_feas_sol, basis, non_basis, A_eq, b_eq)
