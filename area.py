import numpy as np
from scipy.optimize import linprog
from scipy import linalg
from math import floor


class Searchable_Area:
    #scipy.optimize.linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None,
    #method='revised simplex', callback=None, options=None, x0=None)
    
    #res.status
    #int
    #An integer representing the exit status of the algorithm.
    #0 : Optimization terminated successfully.
    #1 : Iteration limit reached.
    #2 : Problem appears to be infeasible.
    #3 : Problem appears to be unbounded.    
    
    #x=linalg.solve(A_LinEq, b_LinEq)
    
    def __init__(self, c, A_ineq_restr=None, UpBound=None, A_eq=None, b_eq=None, direct_bounds=None, A_dual=None, optimal_dual_solution=None, auxiliary_dual_solutions=None, dual_bounds=None):
        
        self.precision=0.00001
        self.SearchArea_Is_Empty=False
        self.OptimalSolution_Is_Integer=False
        self.c=c
        self.A_ub=A_ineq_restr
        self.b_ub=UpBound
        self.A_eq=A_eq
        self.b_eq=b_eq
        self.direct_bounds=direct_bounds
        self.A_ub, self.b_ub=retrieve_inequalities_from_direct_bounds(self.A_ub, self.b_ub, self.direct_bounds)
        self.dual_objective_function=create_dual_objective_function(self.b_ub, self.b_eq)
        self.A_dual, self.nr_equations, self.nr_inequalities=create_dual_matrix(A_dual, self.A_eq, self.A_ub)
        self.dual_bounds=create_dual_boundaries(dual_bounds, self.A_eq, self.A_ub)
        self.optimal_dual_solution, self.optimal_dual_slack=create_optimal_dual_solution(optimal_dual_solution, self.dual_objective_function, self.A_dual, self.c, self.dual_bounds)
        self.auxiliary_dual_solutions, self.auxiliary_values=create_auxiliary_dual_solutions(auxiliary_dual_solutions, self.dual_objective_function, self.A_dual, self.A_ub, self.dual_bounds, self.precision)
    
    
    def tighten_boundaries(self):
        changes_occured=True
        while changes_occured:
            changes_occured=False
            for i in range(self.nr_inequalities):
                b_ub=-np.array(self.A_ub[i])
                #print(self.auxiliary_dual_solutions[i])
                res=linprog(self.dual_objective_function, A_ub=self.A_dual, b_ub=b_ub, bounds=self.dual_bounds, method='revised simplex', x0=self.auxiliary_dual_solutions[i])
                
                #res.status==3 : Problem (dual one in this case) appears to be unbounded. #This means that the direct problem is infeasible
                if (res.status==3):
                    self.SearchArea_Is_Empty=True
                elif (res.status==0): 
                    new_rhs=floor(res.fun+self.precision)
                    #int(res.fun+PRECISION)
                    #new_rhs=int(round(res.fun))
                    #if (abs(new_rhs-res.fun) > PRECISION):
                        #if ((new_rhs-res.fun)>0):
                            #new_rhs-=1
                        #self.b_ub[i]=new_rhs
                        #self.dual_objective_function[i+self.nr_equations]=new_rhs
                        #lst=res.x
                        #bring_small_to_zeros(lst)
                        #self.auxiliary_dual_solutions[i]=lst
                        #self.auxiliary_values[i]=new_rhs
                        #changes_occured=True
                    if (abs(new_rhs-res.fun) > self.precision):
                        self.b_ub[i]=new_rhs
                        self.dual_objective_function[i+self.nr_equations]=new_rhs
                        lst=res.x
                        bring_small_to_zeros(lst, self.precision)
                        self.auxiliary_dual_solutions[i]=lst
                        self.auxiliary_values[i]=new_rhs
                        changes_occured=True
                else:
                    print("Unexpected linprog result in method tighten_boundaries")
                    print("res=", res)
                    print("res=", res)
        return
    
    def add_new_inequality(self, restriction, rhs_value):
        if not self.SearchArea_Is_Empty:
            self.A_ub=np.vstack((self.A_ub, np.asarray(restriction)))
            self.b_ub=np.hstack((self.b_ub, np.asarray(rhs_value)))
            self.dual_objective_function=np.hstack((self.dual_objective_function, np.asarray(rhs_value)))
            self.nr_inequalities+=1
            #self.A_dual=np.hstack((self.A_dual, (-np.array(restriction))))
            self.A_dual=np.column_stack((self.A_dual, (-np.array(restriction)).T))
            direct_bound=(0, None)
            self.dual_bounds.append(direct_bound)
            self.optimal_dual_solution=np.concatenate((self.optimal_dual_solution, np.array([0])), axis=None)
            b_ub=self.c
            res=linprog(self.dual_objective_function, A_ub=self.A_dual, b_ub=b_ub, bounds=self.dual_bounds, method='revised simplex', x0=self.optimal_dual_solution)
            
            #res.status==3 : Problem (dual one in this case) appears to be unbounded. #This means that the direct problem is infeasible
            if (res.status==3):
                self.SearchArea_Is_Empty=True     
            elif (res.status==0):
                lst=res.x
                bring_small_to_zeros(lst, self.precision)
                self.optimal_dual_solution=lst
                self.optimal_dual_slack=res.slack
            else:
                print("unexpected linprog termination in method add_new_inequality")
                print(res)
    
            for i in range(self.nr_inequalities-1):
                self.auxiliary_dual_solutions[i]=np.concatenate((self.auxiliary_dual_solutions[i], np.array([0])), axis=None)
                b_ub=-np.array(self.A_ub[i])
                res=linprog(self.dual_objective_function, A_ub=self.A_dual, b_ub=b_ub, bounds=self.dual_bounds, method='revised simplex', x0=self.auxiliary_dual_solutions[i])
                #res.status==3 : Problem (dual one in this case) appears to be unbounded. #This means that the direct problem is infeasible
                if (res.status==3):
                    self.SearchArea_Is_Empty=True
                elif (res.status==0):
                    new_rhs=floor(res.fun+self.precision)
                    #new_rhs=int(res.fun+PRECISION)
                    #new_rhs=int(round(res.fun))
                    #if (abs(new_rhs-res.fun) > PRECISION):
                        #if ((new_rhs-res.fun)>0):
                            #new_rhs-=1
                    self.b_ub[i]=new_rhs
                    self.dual_objective_function[i+self.nr_equations]=new_rhs
                    lst=res.x
                    bring_small_to_zeros(lst, self.precision)
                    self.auxiliary_dual_solutions[i]=lst
                    self.auxiliary_values[i]=new_rhs
                else:
                    print("unexpected linprog termination in method add_new_inequality")
                    print("res=", res)
                    print("x0", self.auxiliary_dual_solutions[i])
    
            b_ub=-np.array(self.A_ub[self.nr_inequalities-1])
            
            #res.status==3 : Problem (dual one in this case) appears to be unbounded. #This means that the direct problem is infeasible
            res=linprog(self.dual_objective_function, A_ub=self.A_dual, b_ub=b_ub, bounds=self.dual_bounds, method='revised simplex')
            if (res.status==3):
                self.SearchArea_Is_Empty=True
            elif (res.status==0):
                new_rhs=floor(res.fun+self.precision)
                #new_rhs=int(res.fun+PRECISION)
                #new_rhs=int(round(res.fun))
                #if (abs(new_rhs-res.fun) > PRECISION):
                    #if ((new_rhs-res.fun)>0):
                        #new_rhs-=1
                self.b_ub[self.nr_inequalities-1]=new_rhs
                self.dual_objective_function[self.nr_inequalities+self.nr_equations-1]=new_rhs
                lst=res.x
                bring_small_to_zeros(lst, self.precision)
                self.auxiliary_dual_solutions.append(lst)
                self.auxiliary_values.append(new_rhs)
            else:
                print("unexpected linprog termination in method add_new_inequality")
                print(res)
        else:
            print("SearchArea_Is_Empty=", self.SearchArea_Is_Empty)
        return
    
    def get_solution_of_direct_lp(self):
        if not self.SearchArea_Is_Empty:
            NonZeroPositions=list()
            for i in range(len(self.optimal_dual_slack)):
                if abs(self.optimal_dual_slack[i])>self.precision:
                    NonZeroPositions.append(i)
            
            L=len(self.optimal_dual_slack)
            for i in range(len(self.optimal_dual_solution)):
                if abs(self.optimal_dual_solution[i])>self.precision:
                    NonZeroPositions.append((i+L))
            #print (NonZeroPositions)
            identity_size=self.nr_equations+self.nr_inequalities
            identity_matrix=np.identity(identity_size, dtype=int)
            #print(identity_matrix)
            
            
            upper_layer_matrix=np.hstack(((-self.A_dual).T, identity_matrix))
            #print(upper_layer_matrix)
    
            #h_size=self.nr_inequalities
            h_size=len(self.c)
            v_size=h_size+identity_size
            bottom_matrix=np.zeros((h_size, v_size), dtype=int)
            #print(bottom_matrix)
            
            for i in range(len(NonZeroPositions)):
                bottom_matrix[i, NonZeroPositions[i]]=1
            #print(bottom_matrix)
            
            Z_matrix=np.vstack((upper_layer_matrix, bottom_matrix))
            #print(Z_matrix)
            
            upper_rhs_matrix_equation=np.array(self.dual_objective_function)
            #print(upper_rhs_matrix_equation)
            #lower_rhs_matrix_equation=np.zeros(self.nr_inequalities, dtype=int)
            lower_rhs_matrix_equation=np.zeros(h_size, dtype=int)
            #print(lower_rhs_matrix_equation)
            
            
            rhs_matrix_equation=np.hstack((upper_rhs_matrix_equation, lower_rhs_matrix_equation))
            #print(rhs_matrix_equation)
            
            x=linalg.solve(Z_matrix, rhs_matrix_equation)
            #print(x)
            x=x[:len(self.c)]
            #print(x)
            integrality_flag=True
            for i in range(len(x)):
                if (abs(floor(x[i]+self.precision)-x[i])<self.precision):
                    x[i]=floor(x[i]+self.precision)
                else:
                    integrality_flag=False
            self.OptimalSolution_Is_Integer=integrality_flag
            return x
        else:
            print("SearchArea_Is_Empty=", self.SearchArea_Is_Empty)
        return None
        




def create_dual_objective_function(a, b):
    if ((a is not None) and (b is not None)):
        x=np.hstack((b, a))
    elif (a is not None):
        x=a
    else:
        x=b
    return x


def create_dual_matrix(A_dual, Matrix_equations, Matrix_ineqalities):
    if not A_dual:
        if ((Matrix_equations is not None) and (Matrix_ineqalities is not None)):
            A_dual=-(np.vstack((np.asanyarray(Matrix_equations), np.asanyarray(Matrix_ineqalities)))).T
            nr_eqations=len(Matrix_equations)
            nr_inequalities=len(Matrix_ineqalities)
        elif (Matrix_equations is not None):
            A_dual=-(np.asanyarray(Matrix_equations)).T
            nr_eqations=len(Matrix_equations)
            nr_inequalities=0
        else:
            A_dual=-(np.asanyarray(Matrix_ineqalities)).T
            nr_eqations=0
            nr_inequalities=len(Matrix_ineqalities)
    return A_dual, nr_eqations, nr_inequalities
    
def create_dual_boundaries(dual_bounds, Matrix_equations, Matrix_ineqalities):
    if not dual_bounds:
        unbound_tuple=(None, None)
        positive_tuple=(0, None)
        lst_unbound=list()
        lst_positive=list()
        lst_unbound.append(unbound_tuple)
        lst_positive.append(positive_tuple)
        
        dual_boundaries=list()
        
        if (Matrix_equations is not None):
            unbounded=lst_unbound*len(Matrix_equations)
        else:
            unbounded=list()
        
        if (Matrix_ineqalities is not None):
            positive=lst_positive*len(Matrix_ineqalities)
        else:
            positive=list()
        
        if ((Matrix_equations is not None) or (Matrix_ineqalities is not None)):
            dual_boundaries=unbounded+positive #FIX ME
        return dual_boundaries
    return dual_bounds

def create_optimal_dual_solution(optimal_dual_solution, dual_objective_function, A_dual, c, dual_bounds):
    #x=linalg.solve(A_dual, c)
    #scipy.optimize.linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None,
    #method='revised simplex', callback=None, options=None, x0=None)
    
    #res.status==3 : Problem (dual one in this case) appears to be unbounded. #This means that the direct problem is infeasible
    
    b_ub=np.array(c)
    if not optimal_dual_solution:
        res=linprog(dual_objective_function, A_ub=A_dual, b_ub=b_ub, bounds=dual_bounds, method='revised simplex')
        if (res.status==3):
            return None
        elif (res.status==0):
            return res.x, res.slack
        else:
            print("Unexpected linprog termination in function create_optimal_dual_solution")
            print(res)
            return None
    return None

def create_auxiliary_dual_solutions(auxiliary_dual_solutions, dual_objective_function, A_dual, A_ineq_restr, dual_bounds, precision):
    if not auxiliary_dual_solutions:
        auxiliary_dual_solutions=list()
        auxiliary_values=list()
        if (A_ineq_restr is not None):
            for k in range(len(A_ineq_restr)):
                b_ub=-np.array(A_ineq_restr[k])
                
                #res.status==3 : Problem (dual one in this case) appears to be unbounded. #This means that the direct problem is infeasible
                res=linprog(dual_objective_function, A_ub=A_dual, b_ub=b_ub, bounds=dual_bounds, method='revised simplex')
                if (res.status==3):
                    return (None, None)
                elif (res.status==0):
                    lst=res.x
                    bring_small_to_zeros(lst, precision)
                    auxiliary_dual_solutions.append(lst)
                    auxiliary_values.append(res.fun)
                else:
                    print("Unexpected linprog termination in create_auxiliary_dual_solutions")
                    print(res)
                    return (None, None)
        return auxiliary_dual_solutions, auxiliary_values
    else:
        print("Unexpcted call to create_auxiliary_dual_solutions. auxiliary_dual_solutions already existed before")
        return (auxiliary_dual_solutions, None)

        
def bring_small_to_zeros(lst, precision):
    #global PRECISION
    for i in range(len(lst)):
        if ((abs(lst[i]))<precision):
            lst[i]=0

def retrieve_inequalities_from_direct_bounds(A, b, dir_bound):
    if (dir_bound is not None):
        if A is not None:
            A=np.asarray(A)
            b=np.asarray(b)
        
        size=len(dir_bound)
        for i in range(size):
            LB, UB = dir_bound[i]
            if UB is not None:
                new_inequality=np.zeros(size, dtype=int)
                new_inequality[i]=UB
                if A is None:
                    A=new_inequality
                    new_bound=np.array([UB])
                    b=new_bound
                else:
                    A=np.vstack((A, new_inequality))
                    new_bound=np.array([UB])
                    b=np.hstack((b, new_bound))
        
        return A, b
    else:
        return np.asarray(A), np.asarray(b)



#executable programm

if __name__=="__main__":
    #PRECISION=0.0000001
    c=[-3, -1, -7, 4, 5, -19]    #This is objective function. It has to be minimized.
    A_eq=[[1, -2, 0, 1, 1, 0], [-1, 1, 2, 3, 2, 1]]
    b_eq=[10, 10]
    A_ineq_restr=[[-1, -1, 4, 1, 0, 3], [1, 1, 3, 0, 2, 0], [4, 7, 0, 1, 0, 1], [4, 11, -4, 0, -1, 4]]
    UpBound=[-1, 23, 24, 46]
    positive_bound=[(0, None)]
    direct_bounds=5*positive_bound
    
    area=Searchable_Area(c, A_ineq_restr=A_ineq_restr, UpBound=UpBound, A_eq=A_eq, b_eq=b_eq, direct_bounds=direct_bounds)
    area.tighten_boundaries()
    x=area.get_solution_of_direct_lp()
    print(x)
    print("Optimal solution is integer =", area.OptimalSolution_Is_Integer)
    
    
    restriction=[0, 9, 1, 1, 6, 0]
    rhs_value=37
    area.add_new_inequality(restriction, rhs_value)
    area.tighten_boundaries()
    x=area.get_solution_of_direct_lp()
    print(x)
    print("Optimal solution is integer =", area.OptimalSolution_Is_Integer)
    
    restriction=[1, 0, 0, 0, 0, 0]
    rhs_value=6
    area.add_new_inequality(restriction, rhs_value)
    area.tighten_boundaries()
    x=area.get_solution_of_direct_lp()
    print(x)
    print("Optimal solution is integer =", area.OptimalSolution_Is_Integer)
    
    restriction=[0, 1, 0, 0, 0, 0]
    rhs_value=6
    area.add_new_inequality(restriction, rhs_value)
    area.tighten_boundaries()
    x=area.get_solution_of_direct_lp()
    print(x)
    print("Optimal solution is integer =", area.OptimalSolution_Is_Integer)
    
    restriction=[0, 0, 1, 0, 0, 0]
    rhs_value=10
    area.add_new_inequality(restriction, rhs_value)
    area.tighten_boundaries()
    x=area.get_solution_of_direct_lp()
    print(x)
    print("Optimal solution is integer =", area.OptimalSolution_Is_Integer)
    
    restriction=[0, 0, 0, 1, 0, 0]
    rhs_value=3
    area.add_new_inequality(restriction, rhs_value)
    area.tighten_boundaries()
    x=area.get_solution_of_direct_lp()
    print(x)
    print("Optimal solution is integer =", area.OptimalSolution_Is_Integer)
    
    restriction=[0, 0, 0, 0, 1, 0]
    rhs_value=10
    area.add_new_inequality(restriction, rhs_value)
    area.tighten_boundaries()
    x=area.get_solution_of_direct_lp()
    print(x)
    print("Optimal solution is integer =", area.OptimalSolution_Is_Integer)
    
    restriction=[0, 0, 0, 0, 0, 1]
    rhs_value=1
    area.add_new_inequality(restriction, rhs_value)
    area.tighten_boundaries()
    x=area.get_solution_of_direct_lp()
    print(x)
    print("Optimal solution is integer =", area.OptimalSolution_Is_Integer)
