"""
The linear_matrix_equation function solves the equation of the form
****************
A @ x = b
****************

The integrality is observed.
All data and variables MUST be eighet Fractions or integers

The linear_matrix_equation function accepts the
1) the square matrix A
2) the right-hand-side of the equations b


The simplex_method function returns the vector
solution to the problem: x
"""

from fractions import Fraction
import numpy as np


def linear_matrix_equation(A, b):
    A, b = valid_copies_of_input(A, b)
    A, b = diagonalize(A, b)
    return b

def diagonalize(A, b):
    """
    This function manipulates matix A by multiplying and adding (substarcting) the rows of this maytrix
    to bring it the standard identity form (E) 
    """
    height=np.size(b)
    
    for col in range(height): # we iterate over colomns of matrix A
        row=col
        element=A[row][col]
        while ((element==0) and (row<height-1)): # we look for non-zero element in the colomn, starting from diagonal and down
            row+=1
            if A[row][col]!=0:
                element=A[row][col]
        # At this point the variable element is definately non-zero
        
        if row!=col:      #switch two lines in A: the line with the number equal to col and the line with the number equal to row
            b[col]=b[col]+b[row]
            b[row]=b[col]-b[row]
            b[col]=b[col]-b[row]
            for j in range(height):
                A[col][j]=A[col][j]+A[row][j]
                A[row][j]=A[col][j]-A[row][j]
                A[col][j]=A[col][j]-A[row][j]
        # lines are switched
        
        if col<height-1:
            for i in range(row+1, height): # every line below the current row must obtain zero in the colomn col
                if A[i][col]!=0:
                    coef=A[i][col]/A[col][col]
                    b[i]=b[i]-coef*b[col]
                    for j in range(row, height):
                        A[i][j]=A[i][j]-coef*A[col][j]
            
        # the bottom side (under main diagonal) must become zero
    
    for row in range(height-1, 0, -1):
        col=row
        element=A[row][col]
        for i in range(row):
            if A[i][col]!=0:
                coef=A[i][col]/element            
                b[i]=b[i]-coef*b[row]
                for j in range(height):
                    A[i][j]=A[i][j]-coef*A[row][j]

    # the upper side (under main diagonal) must become zero
    for row in range(height):
        coef=A[row][row]
        b[row]=b[row]/coef
        A[row][row]=A[row][row]/coef
            
    
    return A, b


def valid_copies_of_input(A, b):
    """
    This function
    1) creates the copies of all input structures
    (to keep original input object protected against changes that happen while the itterations of simplex method).
    
    2) It ensures that all the structures use Fractions-type elements as their foundation.
    
    3) Further, it validates the sizes and shapes of the input structures to meet
    the general requirements of the classical matrix linear problem
    *****
    A @ x = b
    ******
    """
    A=A.copy()
    b=b.copy()
    
    A_height, A_length=np.shape(A)
    b_length=np.size(b)
    check_dimensions_coherence(A, b)

    if not check_Fraction_or_int_type_of_vector(b, b_length):
        raise ValueError('The type of the RHS (right-hand-side) values is neither integer nor Fraction')
    if not check_Fraction_or_int_type_of_matrix(A, A_height, A_length):
        raise ValueError('The type of the Simplex matrix values is neither integer nor Fraction')

    return A, b

def check_dimensions_coherence(A, b):
    A_height, A_length=np.shape(A)
    b_length=np.size(b)

    if A_length!=A_height:
        raise ValueError('The matrix is not square')
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

def validate_solution(A, x, b):
    A_height, A_length=np.shape(A)
    for i in range(A_height):
        z=np.vdot(A[i], x)
        if z!=b[i]:
            raise ValueError('BFS is invalid. It does not fit into A@x=b equation')

    return



if __name__=="__main__":
    A=np.array([
        [Fraction(4, 1), Fraction(-1, 1), Fraction(0, 1), Fraction(4, 1), Fraction(-78, 1), Fraction(32, 1)],
        [Fraction(6, 1), Fraction(6, 1), Fraction(13, 1), Fraction(5, 1), Fraction(78, 1), Fraction(41, 1)],
        [Fraction(-17, 1), Fraction(2, 1), Fraction(16, 1), Fraction(-8, 1), Fraction(2, 1), Fraction(-22, 1)],
        [Fraction(9, 1), Fraction(6, 1), Fraction(3, 1), Fraction(2, 1), Fraction(-33, 1), Fraction(23, 1)],
        [Fraction(8, 1), Fraction(7, 1), Fraction(1, 1), Fraction(-18, 1), Fraction(11, 1), Fraction(3, 1)],
        [Fraction(7, 1), Fraction(11, 1), Fraction(15, 1), Fraction(21, 1), Fraction(109, 1), Fraction(23, 1)]])
        
    b=np.array([Fraction(-14, 1), Fraction(112, 1), Fraction(18, 1), Fraction(-2, 1), Fraction(98, 1), Fraction(87, 1)])
    
    x=linear_matrix_equation(A, b)
    
    validate_solution(A, x, b)
    
    
