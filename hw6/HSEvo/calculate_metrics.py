import json
import numpy as np
import os
import ast

def get_cyclomatic_complexity(source_code):
    class ComplexityVisitor(ast.NodeVisitor):
        def __init__(self):
            self.complexity = 1
            
        def visit_If(self, node):
            self.complexity += 1
            self.generic_visit(node)
            
        def visit_For(self, node):
            self.complexity += 1
            self.generic_visit(node)
            
        def visit_While(self, node):
            self.complexity += 1
            self.generic_visit(node)
            
        def visit_And(self, node):
            self.complexity += 1
            self.generic_visit(node)
            
        def visit_Or(self, node):
            self.complexity += 1
            self.generic_visit(node)

    try:
        tree = ast.parse(source_code)
        visitor = ComplexityVisitor()
        visitor.visit(tree)
        return visitor.complexity
    except:
        return None

complexities = []
for root, dirs, files in os.walk('.'):
    for file in files:
        if file == 'gpt.py':
            with open(os.path.join(root, file), 'r') as f:
                code = f.read()
                c = get_cyclomatic_complexity(code)
                if c:
                    complexities.append(c)

print(f"\nExplainability (Cyclomatic Complexity):")
if complexities:
    print(f"Average V(G): {np.mean(complexities):.2f} (Min: {np.min(complexities)}, Max: {np.max(complexities)})")
else:
    print("No algorithm files found.")

