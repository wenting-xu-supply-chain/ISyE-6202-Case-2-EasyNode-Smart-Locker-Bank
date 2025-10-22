Task 4: Dynamic Assignment Heuristics Output

This directory contains the output of two different assignment strategies:

1. strawman_assignment_results.csv:
   - Heuristic: Simple 'First-Fit'. Assigns to the first available locker (by ID) that fits.
   - Characteristics: Myopic, easy to implement, not optimized.

2. smarter_assignment_results.csv:
   - Heuristic: 'Cost Minimization'. Selects the locker that minimizes a total cost function.
   - Cost Components: 
     a) Ergonomic Cost: Based on the formula from Faugere & Montreuil (2020).
     b) Oversize Opportunity Cost: A penalty for using a locker larger than required.
   - Characteristics: Smarter, considers both user comfort and space efficiency.
