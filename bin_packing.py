"""
This class generate a Gurobi model object with the optimisation model based
on "Models and bounds for two dimensional level packing problems", Lodi, Martello and Vigo, 2004.
Model (2LBP) page 366, equations 1-13. It adds the functionality from the original paper
to constraint items to a same or level.

Assumptions model: 
    
    - packing is done in levels
    - items sorted and renumbered by non-increasing width values 
    - at each level the farthest item is the widest one
    - in each bin the leftmost level is the widest one
    - no rotation allowed

Inputs:  
    - The class requires a dataframe with items and another one with the 
    information of the different possible bins to use. Each item contains width, depth and
    height (although height is not directly used in the model as it is assumed that 
    there is not allowed to store items on top of other items). Similarly bins
    contains each bin with width, depth and height as well as a cost per unit
    and number of bins per unit (eg, a cabinet with bins type a might have 4 
    bins per cabinet and costs 10).
    
    It is assumed that 
    the columns are called ['index','width','depth','height','shared_bin', 'shared_level','bin']
    for the items (note that the index is assumed to be in a column and not 
    the index of the dataframe). The bin dataframe is ['index','width','depth',
    'height','cost_per_unit','bins_per_unit'].

Example:
    items
      index  width  depth  height shared_level shared_bin  bin
   0      7  333.0    733     650          None       None  None
   1      9  261.0    664     640         None       None  None
   2      6  234.5    768     740         None       None  None
   3      1  152.0    505     740         None       None  None
   4      2  139.0    809     695         None       None  None
   5      0  128.5    331     740         None      None  None
   6      3  116.0    973     700         None       None  None
   7      8  110.5    527     520         None     [1, 6]  None
   8      4  103.5    223     740       [0, 8]       None  None
   9      5   93.0    691     720         None       None  None
   
bins
   index  width  depth  height  cost_per_unit bins_per_unit
0      1    150    340     750       10          3
1      2    400   1000     750       20          4
 
    - In this case the model will constraint the solutions according to the
    columns shared_bin and shared_level, eg, items 7 and 2 should be in the 
    same bin and level, while items 8, 1 and 6
    should be in the same bin.
    
    - Finally a parameter weights to calculate the objective function.
    If weights are {'space':a, 'cost':b] then the objective funciotn is:
            min a*total_cost + b*total_space
    Where tha space is calculated as the sum of the total number of bins 
    multiplied by its area (width * depth) for each type of bin.

Outputs:
    - After instantiated the new object checks the input, creates a Guroby model, and optimise it
    - The output is saved in the dataframe self.items, where for each item (row), besides the original
    data, it adds 3 columns with the bin, type of bin and level where the item must be stored for each solution found.
    - The info about the objective function for each solution found is stored in self.bin_pool_solutions
    - The member self.model contains the information of the optimisation, e.g., self.model.MIPGap
    contains the gap of the best solution found.

Use:
# Import class
from bin_packing import bin_packing
# Create the object 
# If not other option is give the model will have the defaults specified in the set_parameters method
problem = bin_packing(items,bins, {'space':1,'cost':1})
# Create gurobi problem
problem.create_model()
# Solve the problem
problem.optimize()    
# To create a figure of the bin where an item (e.g., 7) is stored in solution 8 use:
problem.display_bin(index_sol = 1, index_item = 7)
# If want to visualize a particular bin (e.g., bin 2) use:
problem.display_bin(index_sol = 2, index_bin = 2)
# To change the parameters of the optimisation problem use dictionary with
# parameters and the method set_parameters, and optimize again. Example
options = {'TimeLimit':10*60, 'FeasibilityTol':1e-7, 'PoolSolutions': 3)}
problem.set_parameters(options)

"""
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import matplotlib.pyplot as plt



class bin_packing:
    def __init__(self, items, bins, weights = {'space':1,'cost':1}, solver_parameters = None):
        self.items = items
        self.bins = bins
        self.solver_parameters = solver_parameters
        self.weights = weights
        self.model = None
        self.n = items.shape[0]
        self.x = []
        self.y = []
        self.q = []
        self.z = []
        self.binary_shared = None
        self.shared_bin = []
        self.shared_level = []
        self.bin_pool_solutions = {}
        self.check_input_data()
        
        
    def check_input_data(self):
        '''
        Check and sort the input data.
        Create auxiliary sets for shared bins and levels.
        
        Returns: None
        '''
        
        # Check if the dimensions of the items fit any bin without rotation
        
        for idx_item, row in self.items.iterrows():    
            aux_set = list(self.bins.loc[:, 'index'])
            if row.loc['bin'] != None:
                aux_set = row.loc['bin']
            w, d, h = row.loc[['width', 'depth', 'height']]
            final_set = []
            for idx_bin in aux_set:
                wb, db, hb = self.bins.loc[self.bins.loc[:,'index'] == idx_bin, ['width', 'depth', 'height']].values[0]
                if (w <= wb) and (d <= db) and (h <= hb):
                    final_set.append(idx_bin)
            assert(len(final_set) > 0), f'Item {self.items.loc[idx_item, "index"]} does not fit in any of the bins with index {aux_set}'
            self.items.at[idx_item, 'bin'] = final_set
            
        # Sorting the data 
        self.items.sort_values(by = 'width', ascending = False, inplace = True)
        self.items.reset_index(drop = True, inplace = True)
        
        # Creating shared sets for items in same level or bin
        self.create_sets_shared()
        
    def create_sets_shared(self): 
        '''
        Creating sets for the items to be in the same level or bin.
        This is used in the added functionality of the Lodi etal 2004 model.
        In particular to allow for constraints that force items to be in 
        the same bin or the same level.
        
        Returns: None
        '''
        
        shared_dict = {'shared_level': self.shared_level, 'shared_bin': self.shared_bin}
        
        for label in ['shared_level', 'shared_bin']:
            for idx, row in self.items[self.items[label].notna()].iterrows():
                
                if label == 'shared_level':
                    # If the item needs to be in the same level it would be in the same bin
                    # Therefore we delete the item if it exsits in the shared_bin list
                    if row['shared_bin']:
                        self.items.at[idx, 'shared_bin'] = [x for x in row['shared_bin'] if x not in row['shared_level']]
                        if len(self.items.at[idx, 'shared_bin']) == 0:
                            self.items.loc[idx, 'shared_bin'] = None
                else:
                    # Check that items in shared bin are not already shared in a level set
                    for i in row['shared_bin']:
                        for shared_level in self.shared_level:
                            if i in shared_level and row['index'] in shared_level:
                                self.items.loc[idx, 'shared_bin'].remove(i)
                    
                # Adding the items to an existing set or creating a new one
                added = False                
                set_aux = set([row['index']] + row[label])
                for set_i in shared_dict[label]:
                    if any(x in set_i for x in set_aux):
                        set_i.update(set_aux)
                        added = True
                        break
                if not added:
                    shared_dict[label].append(set_aux)
                    
        # The original indeces are not the same as the indeces in the model
        # (The original dataframe was sorted and the indeces of the model 
        # correspond to that order)
        # Renumbering the indeces in the shared sets to correspond to the 
        # final indexing
        
        for label in ['shared_level', 'shared_bin']:
            aux_set = []
            for set_i in shared_dict[label]:
                new_set = set()
                for i in set_i:                    
                    new_idx = self.items[self.items.loc[:,'index'] == i].index
                    assert(len(new_idx) == 1),f'Index {i} is repeated in the dataframe'
                    new_set.add(new_idx[0] + 1) # we add one so the number of the items starts at 1
                    
                aux_set.append(new_set)
            shared_dict[label] = aux_set
        
        # Update self.shared_level and self.shared_bin with the updated lists from shared_dict
        self.shared_level = shared_dict['shared_level']
        self.shared_bin = shared_dict['shared_bin']
            
        
    def create_model(self): 
        '''
        Creates Gurobi model.

        Returns: None.
        '''
        
        self.model = gp.Model('2LBP')
        bins_keys = self.bins.loc[:, 'index'].values
        # Variables:
        # y_i_b: 1 if item i initializes level i for bin type b, 0 other wise
        idx_aux = [(i, j) for i in range(1, self.n + 1) for j in bins_keys]
        self.y = self.model.addVars(idx_aux, vtype = GRB.BINARY, name = 'y')
        
        # q_i_b: 1 if item i initializes bin i of type b, 0 other wise
        self.q = self.model.addVars(idx_aux, vtype = GRB.BINARY, name = 'q')
        
        # x_i_j_b:  1 if item j is packed in level i in type bin b, 0 other wise
        # Original model has i<j, but we will add i<=j if self.shared_bin or 
        # self.shared_bin is not empty (ie if we need to place items in the
        # same bin or level). This will facilitate the modelling of the 
        # constraints.
        
        #add_var = 1 if len(self.shared_level) + len(self.shared_bin) > 1 else 0
        add_var = 1
        indices = [(i, j, k) for i in range(1, self.n + add_var) for j in range(i + 1 - add_var, self.n + 1) for k in bins_keys]
        self.x = self.model.addVars(indices, vtype=GRB.BINARY, name = 'x')
        
        # z_k_i_b:  1 if level i is allocated to bin k of type b, 0 other wise
        self.z = self.model.addVars(indices, vtype=GRB.BINARY, name = 'z')
        self.model.update()
        
        # Objective function
        
        self.int_units = self.model.addVars(bins_keys, vtype = GRB.INTEGER, name = 'int_units')
        self.units = self.model.addVars(bins_keys, lb = 0, name = 'total_units')
        self.model.addConstrs((self.units[j] == gp.quicksum(self.q[(i,j)] for i in range(1, self.n + 1))/self.bins.loc[self.bins.loc[:,'index'] == j, 'bins_per_unit'].values[0] for j in bins_keys), name = 'total_units')
        self.model.addConstrs((self.int_units[i] >= self.units[i] for i in bins_keys), name = 'ceil_1')
        self.model.addConstrs((self.int_units[i] <= self.units[i] + 1 - self.model.Params.IntFeasTol for i in bins_keys), name = 'ceil_2')
                
        expr_space = gp.quicksum(self.q[(i,b)]*(self.bins.loc[self.bins.loc[:,'index'] == b,'width'].values*self.bins.loc[self.bins.loc[:,'index'] == b,'depth'].values) for i,b in idx_aux)
        expr_cost = gp.quicksum(self.int_units[i]*self.bins.loc[self.bins.loc[:,'index'] == i,'cost_per_unit'].values for i in bins_keys)
        
        self.model.setObjective(self.weights['space']*expr_space + self.weights['cost']*expr_cost, GRB.MINIMIZE)
        
        # Constraints
        # Constraints for items to be stored once
        self.model.addConstrs((gp.quicksum(gp.quicksum(self.x[i, j, b] for i in range(1,j)) + self.y[j,b] for b in bins_keys) == 1 for j in range(1, self.n + 1)), name = 'c1')
        # Constraints to not exceeed the depth of the bin
        self.model.addConstrs((gp.quicksum(self.x[i, j, b]*self.items.depth.iloc[j-1] for j in range(i + 1, self.n + 1)) <= (self.bins.loc[self.bins.loc[:, 'index'] == b, 'depth'].values[0] - self.items.depth.iloc[i - 1])*self.y[i, b] for i in range(1, self.n) for b in bins_keys), name = 'c2')
        
        # Constraints for each level is allocated to exactly one bin
        self.model.addConstrs((gp.quicksum(self.z[k, i, b] for k in range(1, i)) + self.q[i, b] == self.y[i, b] for i in range(1, self.n + 1) for b in bins_keys), name = 'c3')
        # Constraints to not exceeed the width of the bin
        self.model.addConstrs((gp.quicksum(self.z[k, i, b]*self.items.width.iloc[i - 1] for i in range(k + 1,self.n + 1)) <= (self.bins.loc[self.bins.loc[:, 'index'] == b, 'width'].values[0] - self.items.width.iloc[k - 1])*self.q[k, b] for k in range(1, self.n) for b in bins_keys), name = 'c4')
        # If add_var == 1 then we need to add constraints for the added variables to the model
        if add_var:
            self.model.addConstrs((self.x[(i, i, b)] == self.y[i, b] for i in range(1, self.n + 1) for b in bins_keys), name = 'c5_1')
            self.model.addConstrs((self.z[(i, i, b)] == self.q[i, b] for i in range(1, self.n + 1) for b in bins_keys), name = 'c5_2')
        
        # Added functionality outside the original model of Lodi etal 2004   
        # Constraints for items to be in the same level
        self.create_constraints_same_level()
        # Constraints for items to be in the same bin
        self.create_constraints_same_bin()
        # Constraints for items not allowed in certain kind of bins
        self.create_constraints_valid_bin()
        
        
        # Set parameters of optimisation
        self.set_parameters(self.solver_parameters)
        self.model.update()
        
        print('Model created')
    
    def create_constraints_same_level(self):
        '''
        Create constraints given by column 'shared_level'.
        
        return: None
        '''
        bins_keys = self.bins.loc[:,'index'].values
        for set_shared in self.shared_level:
            sorted_set = sorted(set_shared)
            i = sorted_set[0]            
            self.model.addConstrs((self.x[(l, i, b)] == self.x[(l, j, b)] for j in sorted_set[1:] for l in range(1, i + 1) for b in bins_keys), name = f'shared_level_{i}')
              
    def create_constraints_same_bin(self):
        '''
        Create constraints given by column 'shared_bin'.
        
        return: None
        '''
        bins_keys = self.bins.loc[:,'index'].values
        # Constraints for items to be in the same bin
        self.binary_shared = [[]]*len(self.shared_bin)
        count = 0
        
        for set_shared in self.shared_bin:  
            big_M = len(set_shared)
            sorted_set = sorted(set_shared)  
            i = sorted_set[0]
            idx_aux = [(i,j) for i in range(1, i + 1) for j in bins_keys]
            self.binary_shared[count] = self.model.addVars(idx_aux, vtype = GRB.BINARY, name = f'shared_bin_var_{i}')            
            self.model.addConstrs((self.x[(l, j, b)] <= self.z[(k, l, b)] + 1 - self.binary_shared[count][(k,b)] for k,b in idx_aux for j in sorted_set for l in range(k, j + 1)), f'shared_bin_{i}')
            
            self.model.addConstrs((gp.quicksum(self.x[(l,j,b)] for j in sorted_set for l in range(k, j+1))  <= big_M*(1 - self.binary_shared[count][(k,curr_bin)]) for curr_bin in bins_keys for b in bins_keys if b != curr_bin for k in range(1,sorted_set[0] +1)), name = f'shared_bin_type_{i}')
            
            self.model.addConstrs((gp.quicksum(self.x[(l,j,b)]  for l in range(1, k) for j in sorted_set for b in bins_keys) <= big_M*(1 - gp.quicksum(self.binary_shared[count][(k,b)] for b in bins_keys)) for k in range(2, sorted_set[0] + 1)), name = f'shared_level_bin_{i}')
            
            self.model.addConstr(gp.quicksum(self.binary_shared[count][(k, b)] for k in range(1, i + 1) for b in bins_keys) == 1, name = f'binary_shared_eq_1_{i}')
            count += 1
     
    def create_constraints_valid_bin(self):
        '''
        Create constraints given by column 'bins' in items.
        
        return: None
        '''
        bins_keys = self.bins.loc[:,'index'].values
        add_var = 1 if len(self.shared_level) + len(self.shared_bin) > 1 else 0
        indices = [(i, j, k) for i in range(1, self.n + add_var) for j in range(i + 1 - add_var, self.n + 1) for k in bins_keys]
        self.model.addConstrs((self.y[(i,k)] == 0 for i in range(1, self.n + 1) for k in bins_keys if k not in self.items.bin.iloc[i-1]), name = 'valid_bins_y')
        self.model.addConstrs((self.x[(i,j,k)] == 0  for i,j,k in indices if k not in self.items.bin.iloc[j-1]), name = 'valid_bins_x')
        self.model.addConstrs((self.q[(i,k)] == 0 for i in range(1, self.n + 1) for k in bins_keys if k not in self.items.bin.iloc[i-1]), name = 'valid_bins_q')
        self.model.addConstrs((self.z[(i,j,k)] == 0  for i,j,k in indices if k not in self.items.bin.iloc[j-1]), name = 'valid_bins_z')
        
    def set_parameters(self, options = None):
        '''
        Method to change Gurobi parameters.
        
        Input: dictionary with key = parameter to change
               and value the value to assigned.
               
        Return: None.
        '''
        if options == None:
            #self.model.setParam('MIPGap', 1e-8)
            #self.model.setParam('FeasibilityTol', 1e-8)
            self.model.setParam('TimeLimit', 3600)
            #self.model.setParam('PoolGap', 0.0)
            self.model.setParam('PoolSearchMode', 2)
            self.model.setParam('PoolSolutions', 10) 
            self.model.setParam('PoolGapAbs', 1e-5)    
        else:
            keys_to_check = ['PoolSearchMode', 'PoolSolutions', 'PoolGap', 'PoolGapAbs']
            if any(key in options.keys() for key in keys_to_check):
                print('The model will be reset (all info about previous solutions will be lost).') 
                self.model.reset()
                self.items = self.items.loc[:,['index', 'width', 'depth', 'height', 'shared_bin', 'shared_level','bin']]
                self.bin_pool_solutions = {}
                           
            for key, val in options.items():
                self.model.setParam(key, val)
        self.model.update()
                
    def optimize(self):
        '''
        Optimise the model after self.create_model().
        Generate solutions and add them into self.items.
        Display solutions message to log.
        
        Returns: None.
        '''
        self.model.optimize()
        self.generate_solution()
        self.display_info_optimisation()
          
    def generate_solution(self):
        '''
        After self.optimize(), if solutions were found, 
        add them to self.items.

        Returns: None.
        '''
        tol_int_feas = self.model.Params.IntFeasTol
        total_solutions = self.model.SolCount
        # If no solution or infeasible, unbounded
        if self.model.Status in [1,3,4,5,6]:
            status_dict = self.gurobi_status_dict()
            print(f'\n\n{"_"*45} SOLUTION {"_"*45}\n\n')
            print(f'{status_dict[self.model.Status]}')
            __import__('sys').exit()  # Exit the script
            
        obj_opt = self.model.objVal 
        
        # Finding indeces of solutions that satisfy the required gap with respect to the best solution found
        sol_index = [True]*total_solutions
        s = 1
        bins_keys = self.bins.loc[:, 'index'].values       
        idx_aux = [(i, j) for i in range(1, self.n + 1) for j in bins_keys]
        for count in range(total_solutions):
            
            self.model.Params.SolutionNumber = count            
            expr_space = gp.quicksum(self.q[(i,j)].Xn*(self.bins.loc[self.bins.loc[:,'index'] == j,'width'].values*self.bins.loc[self.bins.loc[:,'index'] == j,'depth'].values) for i,j in idx_aux)            
            expr_cost = gp.quicksum(self.int_units[i].Xn*self.bins.loc[self.bins.loc[:,'index'] == i,'cost_per_unit'].values for i in bins_keys)
           
            
            obj_aux = (self.weights['space']*expr_space + self.weights['cost']*expr_cost).getValue()
            # Finding solutions that satisfy the required gap with respect to the best solution found
            if obj_opt == 0:
                rel_gap = 0
            else:
                rel_gap = obj_aux/obj_opt - 1
            if rel_gap > self.model.Params.PoolGap or abs(obj_aux - obj_opt) > self.model.Params.PoolGapAbs:
                sol_index[count] = False
                continue
            self.bin_pool_solutions[s] = {'obj':obj_aux, 'space': expr_space, 'cost': expr_cost}
            s += 1
        
        cols = [f'{x}_{y}' for y in range(1, s) for x in ['bin', 'level','bin_type']]
        self.items = pd.concat([self.items, pd.DataFrame(columns = cols)], axis = 1)
        
        # Returning all the solutions in sol_index
        s = 1
        for count in range(total_solutions):
            if not sol_index[count]:
                continue
            self.model.Params.SolutionNumber = s - 1
            self.items[f'bin_{s}'] = 0
            self.items[f'level_{s}'] = 0
            self.items[f'bin_type_{s}'] = 0
            for i in range(1, self.n + 1):
                idx = self.items.index[i - 1]
                aux = [k for k in bins_keys if abs(self.y[i,k].Xn - 1) <= tol_int_feas]
                assert(len(aux) <= 1)
                if len(aux) == 1:                
                    self.items.loc[idx, f'level_{s}'] = i   
                    self.items.loc[idx, f'bin_type_{s}'] = aux[0]  
                    if abs(self.q[i, aux[0]].Xn - 1) <= tol_int_feas:       
                        self.items.loc[idx, f'bin_{s}'] = i                    
                    else:
                        z_nonzero = [j for j in range(1, i) if abs(self.z[j, i, aux[0]].Xn - 1) <= tol_int_feas]
                        assert(len(z_nonzero) <= 1)
                        self.items.loc[idx, f'bin_{s}'] = z_nonzero[0]                   
                else:
                    x_nonzero = [(j, k) for j in range(1, i) for k in bins_keys if abs(self.x[j, i, k].Xn - 1) <= tol_int_feas]
                    assert(len(x_nonzero) <= 1)
                    self.items.loc[idx, f'level_{s}'] = x_nonzero[0][0]            
                    self.items.loc[idx, f'bin_{s}'] = self.items.loc[:,f'bin_{s}'].iloc[x_nonzero[0][0] - 1]
                    self.items.loc[idx, f'bin_type_{s}'] = x_nonzero[0][1]
                    
            # Renumber levels within bin
            self.items.loc[:, f'level_{s}'] = self.items.groupby([f'bin_{s}', f'bin_type_{s}'])[f'level_{s}'].transform(lambda x: x.rank(method='dense').astype(int))

            # Renumber bins
            self.items.loc[:, f'bin_{s}'] = self.items.loc[:,f'bin_{s}'].rank(method = 'dense').astype('int')
            s += 1
            
        # Verifying that the solution is correct       
        for i in range(1, s):
            self.check_solution_fit(i)
            self.check_solution_shared(i)
            self.check_solution_valid_bin(i)
            
    def check_solution_fit(self, index_sol):  
        '''
        Check if the solution provided fits in the bins
        Return: error if solution does not fit
        '''
        bin_col = f'bin_{index_sol}'
        level_col = f'level_{index_sol}'
        bin_type_col = f'bin_type_{index_sol}'
        df = self.items.loc[:,['width','depth', bin_col, level_col, bin_type_col]]
        grouped = df.groupby([bin_col, level_col, bin_type_col])
        depth_sum = grouped.depth.sum()
        width_sum = grouped.width.max().groupby([bin_col, bin_type_col]).sum()
        depth_sum = depth_sum.reset_index().merge(self.bins, left_on=bin_type_col, right_on='index', how='left')
        width_sum = width_sum.reset_index().merge(self.bins, left_on=bin_type_col, right_on='index', how='left')

        # Check that the items fit within the bins
        assert all(depth_sum['depth_x'] <= depth_sum['depth_y']), f'The items of Solution {index_sol} do not fit (depth)'
        assert all(width_sum['width_x'] <= width_sum['width_y']), f'The items of Solution {index_sol} do not fit (width)'

    def check_solution_shared(self, index_sol):
        bin_col = f'bin_{index_sol}'
        level_col = f'level_{index_sol}'
        bin_type_col = f'bin_type_{index_sol}'
        df = self.items[['index', 'shared_bin', 'shared_level', bin_col, level_col, bin_type_col]]
        idx_shared_bin = df['shared_bin'].notna()
        idx_shared_level = df['shared_level'].notna()
        # Check shared_bin
        if idx_shared_bin.any():
            df_shared_bin = df[idx_shared_bin].explode('shared_bin')
            df_shared_bin = df_shared_bin.merge(df[['index', bin_col]], left_on='shared_bin', right_on='index', suffixes=('', '_shared'))
            mismatches_bin = df_shared_bin[df_shared_bin[bin_col] != df_shared_bin[bin_col + '_shared']]
            assert(mismatches_bin.empty), f'For solution {index_sol}, items {mismatches_bin.loc[:, "index"].values} mismatches bin with items {mismatches_bin.loc[:, "index_shared"].values}'
        
        # Check shared_level
        if idx_shared_level.any():
            df_shared_level = df[idx_shared_level].explode('shared_level')
            df_shared_level = df_shared_level.merge(df[['index', bin_col, level_col]], left_on='shared_level', right_on='index', suffixes=('', '_shared'))
            mismatches_level = df_shared_level[(df_shared_level[bin_col] != df_shared_level[bin_col + '_shared']) | (df_shared_level[level_col] != df_shared_level[level_col + '_shared'])]
            assert(mismatches_level.empty), f'For solution {index_sol}, items {mismatches_level.loc[:, "index"].values} mismatches level with items {mismatches_level.loc[:, "index_shared"].values}'
    
    def check_solution_valid_bin(self, index_sol):
          '''
          Check if the solution provided each item is in the right bin
          Return: error if solution does not fit
          '''
          
          bin_type_col = f'bin_type_{index_sol}'
          df = self.items[['index', 'bin', bin_type_col]]
          def check_bin_type(row):
              return row[bin_type_col] in row['bin']
          # Apply the function row-wise
          assert(df.apply(check_bin_type, axis=1).sum() == df.shape[0]), f'An error in solution {index_sol} was found: not all the items were packed in a bin of the type determined by the column "bin" in the items dataframe'

          
          
    def display_info_optimisation(self):
        '''
        Display info after self.optimize to log.
        
        Returns: None
        '''
        status = self.model.Status       
        status_dict = self.gurobi_status_dict()
        bins_keys = self.bins.loc[:, 'index'].values       
        print(f'\n\n{"_"*45} SOLUTION {"_"*45}\n\n')
        print(f'{status_dict[status]}')
        print(f'The gap is: {self.model.MIPGap}')
        #print(f'The number of bins in the solution is: {gp.quicksum(self.q[i,k].X for i in range(1, self.n + 1) for k in bins_keys)}')
        print(f'A total of {self.model.SolCount} solutions were found with a relative gap of {self.model.Params.PoolGap*100}% and absolute gap of {self.model.Params.PoolGapAbs} with respect to the bins of the best solution found.')
    
    def  gurobi_status_dict(self):
        '''
        Returns: dictionary with Gurobi status definitions.
        '''
        status_dict = {
    1: "Model is loaded, but no solution information is available.",
    2: "Model was solved to optimality (subject to tolerances), and an optimal solution is available.",
    3: "Model was proven to be infeasible.",
    4: "Model was proven to be either infeasible or unbounded. To obtain a more definitive conclusion, set the DualReductions parameter to 0 and reoptimize.",
    5: "Model was proven to be unbounded. Important note: an unbounded status indicates the presence of an unbounded ray that allows the objective to improve without limit. It says nothing about whether the model has a feasible solution. If you require information on feasibility, you should set the objective to zero and reoptimize.",
    6: "Optimal objective for model was proven to be worse than the value specified in the Cutoff parameter. No solution information is available.",
    7: "Optimization terminated because the total number of simplex iterations performed exceeded the value specified in the IterationLimit parameter, or because the total number of barrier iterations exceeded the value specified in the BarIterLimit parameter.",
    8: "Optimization terminated because the total number of branch-and-cut nodes explored exceeded the value specified in the NodeLimit parameter.",
    9: "Optimization terminated because the time expended exceeded the value specified in the TimeLimit parameter.",
    10: "Optimization terminated because the number of solutions found reached the value specified in the SolutionLimit parameter.",
    11: "Optimization was terminated by the user.",
    12: "Optimization was terminated due to unrecoverable numerical difficulties.",
    13: "Unable to satisfy optimality tolerances; a sub-optimal solution is available.",
    14: "An asynchronous optimization call was made, but the associated optimization run is not yet complete.",
    15: "User specified an objective limit (a bound on either the best objective or the best bound), and that limit has been reached.",
    16: "Optimization terminated because the work expended exceeded the value specified in the WorkLimit parameter.",
    17: "Optimization terminated because the total amount of allocated memory exceeded the value specified in the SoftMemLimit parameter."
} 
        return status_dict
    
    
    def display_bin(self, index_sol = 1, index_item = None, index_bin = None):
        '''
        Generate 3d figures of bin after model.optimize() 
        if solutions were found.
        For a given solution (index_sol):
            - If index_item is given creates a figure of the bin where the item with index is located. 
            - If index_bin is given creates a figure of the bin. 
                
        Return: figure of bin.
        '''
        
        if not index_bin or index_item:
            assert(index_sol in range(1, len(self.bin_pool_solutions)+1)), 'The index_sol is not an index in the pool of solutions'                
            index_bin = self.items.loc[self.items.loc[:, 'index'] == index_item, f'bin_{index_sol}'].values
            assert(len(index_bin) == 1), 'The index provided does not exist in the dataframe self.items or is repeated'
            index_bin = index_bin[0]       
        
        df_bin = self.items.loc[self.items.loc[:, f'bin_{index_sol}'] == index_bin, ['index', 'width', 'depth', 'height', f'bin_{index_sol}', f'level_{index_sol}', f'bin_type_{index_sol}']]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        levels = df_bin.loc[:, f'level_{index_sol}'].unique()
        
        # Starting position x axis
        x_start = 0
        bin_type = df_bin.loc[:, f'bin_type_{index_sol}'].unique()
        assert(bin_type.shape[0] == 1), f'In solution {index_sol} and bin {index_bin} the bin belong to more than one type of bin'
        bin_type = bin_type[0]
        bin_depth = self.bins.loc[self.bins.loc[:,'index'] == bin_type, 'depth'].values[0]
        bin_width = self.bins.loc[self.bins.loc[:,'index'] == bin_type, 'width'].values[0]
        bin_height = self.bins.loc[self.bins.loc[:,'index'] == bin_type, 'height'].values[0]
        
        for l in levels:
            idx_level = df_bin.loc[:, f'level_{index_sol}'] == l
            df_bin_level = df_bin.loc[idx_level, :].sort_values(by = ['width'], ascending = False)
            max_width = df_bin_level.width.max()
            # Starting position y axis
            y_start = bin_depth
            for idx, row in df_bin_level.iterrows():  
                # Updating starting position for current box
                y_start -= row['depth']
                color_box = 'b' if row['index'] == index_item else 'r'
                ax.bar3d(x_start, y_start, 0, row['width'], row['depth'], row['height'], color = color_box, alpha = 0.5, edgecolor = 'k', linewidth = 1)
                
                ax.text(x_start + row['width'] / 2, y_start + row['depth'] / 2, row['height'], str(row['index']), color='black', weight = 'bold', ha = 'center')
                
            # Updating starting position for next level
            x_start += max_width
                    
        # Creating figure
        ax.bar3d(0, 0, 0, bin_width, bin_depth, bin_height, color=(0, 0, 1, 0), edgecolor='black')
        ax.set_xlabel('Width')
        ax.set_ylabel('Depth')
        ax.set_zlabel('Height')
        title_str = ''
        if index_item:
            title_str = f', where item index: {index_item} is stored'
            
        ax.set_title(f'Solution: {index_sol} \n Layout of bin: {index_bin}{title_str} \n Bin type: {bin_type}')
        
        # Show plot
        plt.show()
        
        return ax
    def return_sol(self, index_sol = 1):
            df_bin = self.items.loc[:, ['index', 'width', 'depth', 'height', f'bin_{index_sol}', f'level_{index_sol}', f'bin_type_{index_sol}']].dropna(subset = f'bin_{index_sol}')
            return df_bin
    
    
