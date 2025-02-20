# Bin-Packing-Optimization-Storage

Implement and extend the capability of the bin packing formulation in "Models and bounds for two-dimensional level packing problems", by Lodi A, Martello S, Vigo D, Journal of
Combinatorial Optimization 8:363–379, 2004.

The problem corresponds to a 2-dimensional bin packing problem by levels: it assumes that items and bins are rectangular, that items need to be packed in non-overlapping levels, and that rotation of the items is not allowed. This basic model is extended by:
- Adding different types of bins and associated costs: it is assumed that there are different types of furnitures and each piece of furniture contains a fixed number of uniform bins (e.g., furniture type A contains 3 bins of dimensions 10x20 each one, and the cost corresponds to the full piece of type A with the 3 bins).
- Allowing constraints that restrict items to be in the same level or the same bin in the final solution.
- Allowing constraints to restrict items to a specific type of bin.

All the codes are formulated in Python and use Gurobi as solver.

## Requirements
The code has been tested in a MacBook Pro with 2.3 GHz 8-Core Intel Core i9 and 32 GB, with Gurobi 11.0. The code requires:
- Python
- Gurobipy solver
- Pandas
- matplotlib

## Files and folders
1. bin_packing.py: this file contains the main file to run the bin packing problem.
2. example_bin_packing.ipybn: Jupyter notebook file containing an example with artificial data using the bin_packing.py file.
	

## Author
---------------------

## License
All the codes are released under the MIT License. Please refer to the LICENSE file for details.

## Acknowledgements
--------------------- 

## References
- Lodi A, Martello S, Vigo D (2004) Models and bounds for two-dimensional level packing problems. Journal of
Combinatorial Optimization 8:363–379.
