"""
Verifier: Validates generated code for logical errors and correctness
"""

import ast
import re
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import nbformat

from agent_v6.core.agent import Agent
from agent_v6.core.tools import ToolRegistry, BashTool, ReadTool, WriteTool


class Verifier:
    """
    Verifies notebook code by:
    - Checking for syntax errors
    - Validating imports
    - Checking for logical fallacies
    - Ensuring correct data paths
    - Validating model architecture
    - Checking output format
    """
    
    def __init__(self, logs_dir: Path):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
    async def verify(self, notebook_path: Path, iteration: int, attempt: int = 0) -> Dict:
        """
        Verify notebook for errors
        
        Args:
            notebook_path: Path to notebook to verify
            iteration: Current iteration
            attempt: Current fix attempt
            
        Returns:
            Dictionary with verification results
        """
        print(f"🔍 Verifying notebook (iteration {iteration}, attempt {attempt})...")
        
        # Load notebook
        with open(notebook_path, 'r') as f:
            notebook = nbformat.read(f, as_version=4)
        
        # Initialize results
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'syntax_errors': [],
            'import_errors': [],
            'logic_errors': [],
            'data_errors': [],
            'output_errors': []
        }
        
        # Run verification checks
        self._check_syntax(notebook, results)
        await self._check_imports(notebook, results)
        self._check_logic(notebook, results)
        self._check_data_handling(notebook, results)
        self._check_output_format(notebook, results)
        
        # Determine overall validity
        total_errors = len(results['errors'])
        results['is_valid'] = (total_errors == 0)
        
        # Save verification log
        log_file = self.logs_dir / f"verification_iter{iteration}_attempt{attempt}.json"
        with open(log_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        if results['is_valid']:
            print(f"✅ Verification PASSED")
        else:
            print(f"❌ Verification FAILED: {total_errors} errors found")
            for i, error in enumerate(results['errors'][:5], 1):  # Show first 5 errors
                print(f"   {i}. {error}")
        
        return results
    
    def _check_syntax(self, notebook: nbformat.NotebookNode, results: Dict):
        """Check for syntax errors in code cells"""
        
        for cell_idx, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                try:
                    ast.parse(cell.source)
                except SyntaxError as e:
                    error_msg = f"Syntax error in cell {cell_idx}: {e.msg} at line {e.lineno}"
                    results['syntax_errors'].append(error_msg)
                    results['errors'].append(error_msg)
    
    async def _check_imports(self, notebook: nbformat.NotebookNode, results: Dict):
        """Check if all imports are valid"""
        
        # Extract all import statements
        imports = []
        for cell in notebook.cells:
            if cell.cell_type == "code":
                lines = cell.source.split('\n')
                for line in lines:
                    if line.strip().startswith('import ') or line.strip().startswith('from '):
                        imports.append(line.strip())
        
        if not imports:
            results['warnings'].append("No import statements found")
            return
        
        # Create test script
        test_script = "#!/usr/bin/env python\n"
        test_script += "import sys\n\n"
        test_script += "errors = []\n"
        
        for imp in imports:
            test_script += f"try:\n"
            test_script += f"    {imp}\n"
            test_script += f"except ImportError as e:\n"
            test_script += f"    errors.append('{imp}: ' + str(e))\n"
            test_script += f"except Exception as e:\n"
            test_script += f"    errors.append('{imp}: ' + str(e))\n"
        
        test_script += "\nif errors:\n"
        test_script += "    for error in errors:\n"
        test_script += "        print('IMPORT_ERROR:', error)\n"
        test_script += "else:\n"
        test_script += "    print('ALL_IMPORTS_OK')\n"
        
        # Write and execute test script
        test_file = self.logs_dir / "test_imports.py"
        test_file.write_text(test_script)
        
        try:
            process = await asyncio.create_subprocess_shell(
                f"python {test_file}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            output = stdout.decode()
            
            # Parse results
            for line in output.split('\n'):
                if line.startswith('IMPORT_ERROR:'):
                    error_msg = line.replace('IMPORT_ERROR:', '').strip()
                    results['import_errors'].append(error_msg)
                    results['errors'].append(f"Import failed: {error_msg}")
        
        except Exception as e:
            results['warnings'].append(f"Could not verify imports: {e}")
    
    def _check_logic(self, notebook: nbformat.NotebookNode, results: Dict):
        """Check for logical errors in the code"""
        
        # Combine all code cells
        full_code = []
        for cell in notebook.cells:
            if cell.cell_type == "code":
                full_code.append(cell.source)
        
        combined_code = '\n\n'.join(full_code)
        
        # Check for common logical issues
        logical_checks = [
            {
                'pattern': r'X_train.*=.*X_test',
                'error': 'Potential data leakage: X_train assigned from X_test'
            },
            {
                'pattern': r'y_train.*=.*y_test',
                'error': 'Potential data leakage: y_train assigned from y_test'
            },
            {
                'pattern': r'fit\([^)]*test[^)]*\)',
                'error': 'Model might be fitting on test data'
            },
            {
                'pattern': r'model\.predict\(\s*\)',
                'error': 'Model predict called without data'
            },
            {
                'pattern': r'train_test_split.*test_size\s*=\s*0\.?0',
                'error': 'Test size is 0 in train_test_split'
            },
            {
                'pattern': r'train_test_split.*test_size\s*=\s*1\.?0',
                'error': 'Test size is 1.0 in train_test_split (no training data)'
            },
            {
                'pattern': r'for\s+.*\s+in\s+range\(0\)',
                'error': 'Loop with range(0) will not execute'
            },
            {
                'pattern': r'if\s+True\s*==\s*False',
                'error': 'Impossible condition: True == False'
            }
        ]
        
        for check in logical_checks:
            if re.search(check['pattern'], combined_code, re.IGNORECASE):
                results['logic_errors'].append(check['error'])
                results['errors'].append(f"Logic error: {check['error']}")
        
        # Check for undefined variables (basic)
        # This is a simplified check - a proper one would need AST analysis
        variable_definitions = set()
        variable_uses = set()
        
        # Find assignments (simplified)
        for match in re.finditer(r'^([a-zA-Z_]\w*)\s*=', combined_code, re.MULTILINE):
            variable_definitions.add(match.group(1))
        
        # Find uses in function calls (simplified)
        for match in re.finditer(r'(?:fit|predict|transform|score)\(([^)]+)\)', combined_code):
            args = match.group(1)
            for var in re.findall(r'\b([a-zA-Z_]\w*)\b', args):
                if var not in ['True', 'False', 'None', 'self']:
                    variable_uses.add(var)
        
        # Check for undefined variables
        undefined = variable_uses - variable_definitions
        common_vars = {'X_train', 'y_train', 'X_test', 'y_test', 'X_val', 'y_val', 
                      'train_df', 'test_df', 'model', 'np', 'pd', 'plt'}
        undefined = undefined - common_vars  # Exclude commonly imported/defined vars
        
        if undefined:
            for var in list(undefined)[:3]:  # Limit to first 3
                results['warnings'].append(f"Potentially undefined variable: {var}")
    
    def _check_data_handling(self, notebook: nbformat.NotebookNode, results: Dict):
        """Check data handling and paths"""
        
        combined_code = '\n\n'.join(cell.source for cell in notebook.cells 
                                   if cell.cell_type == "code")
        
        # Check for data loading
        if 'read_csv' not in combined_code and 'load' not in combined_code.lower():
            results['data_errors'].append("No data loading detected")
            results['errors'].append("No data loading found (read_csv, load, etc.)")
        
        # Check for proper file paths
        if '/home/data' in combined_code or 'Config.DATA_DIR' in combined_code:
            # Expected paths for MLE-bench
            pass
        elif '../data' in combined_code or './data' in combined_code:
            # Relative paths are ok
            pass
        else:
            # Check if any data paths are specified
            if not re.search(r'["\'].*\.(csv|json|parquet|txt)["\']', combined_code):
                results['warnings'].append("No clear data file paths found")
        
        # Check for train/test split
        if 'train_test_split' not in combined_code:
            if 'KFold' not in combined_code and 'StratifiedKFold' not in combined_code:
                results['warnings'].append("No train/validation split detected")
        
        # Check for missing value handling
        if 'fillna' not in combined_code and 'dropna' not in combined_code:
            if 'SimpleImputer' not in combined_code and 'KNNImputer' not in combined_code:
                results['warnings'].append("No missing value handling detected")
        
        # Check for feature scaling (for certain models)
        if any(model in combined_code for model in ['SVC', 'SVR', 'LogisticRegression', 'NeuralNetwork']):
            if 'StandardScaler' not in combined_code and 'MinMaxScaler' not in combined_code:
                results['warnings'].append("No feature scaling detected (recommended for SVM/NN models)")
    
    def _check_output_format(self, notebook: nbformat.NotebookNode, results: Dict):
        """Check if output format is correct"""
        
        combined_code = '\n\n'.join(cell.source for cell in notebook.cells 
                                   if cell.cell_type == "code")
        
        # Check for VALIDATION_SCORE output
        if 'VALIDATION_SCORE' not in combined_code:
            results['output_errors'].append("Missing VALIDATION_SCORE output")
            results['errors'].append("Code must print 'VALIDATION_SCORE: X.XXX' for tracking")
        else:
            # Check format
            if not re.search(r'print.*VALIDATION_SCORE.*[{:f\d.]', combined_code):
                results['warnings'].append("VALIDATION_SCORE might not be formatted correctly")
        
        # Check for model saving
        if 'save' not in combined_code.lower() and 'dump' not in combined_code.lower():
            if 'torch.save' not in combined_code and 'joblib.dump' not in combined_code:
                results['warnings'].append("Model saving not detected")
        
        # Check for submission generation
        if 'submission' in combined_code.lower():
            if 'to_csv' not in combined_code:
                results['warnings'].append("Submission might not be saved to CSV")
        
        # Check for proper evaluation metrics
        metric_functions = [
            'accuracy_score', 'roc_auc_score', 'mean_squared_error',
            'log_loss', 'f1_score', 'precision_score', 'recall_score'
        ]
        
        if not any(metric in combined_code for metric in metric_functions):
            results['warnings'].append("No evaluation metrics found")
    
    def generate_trace_log(self, notebook_path: Path, results: Dict, iteration: int) -> Path:
        """Generate detailed trace log for debugging"""
        
        trace_log = self.logs_dir / f"trace_iter{iteration}.txt"
        
        with open(trace_log, 'w') as f:
            f.write(f"VERIFICATION TRACE LOG - Iteration {iteration}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Notebook: {notebook_path}\n")
            f.write(f"Valid: {results['is_valid']}\n")
            f.write(f"Total Errors: {len(results['errors'])}\n")
            f.write(f"Total Warnings: {len(results['warnings'])}\n\n")
            
            if results['syntax_errors']:
                f.write("SYNTAX ERRORS:\n")
                for error in results['syntax_errors']:
                    f.write(f"  - {error}\n")
                f.write("\n")
            
            if results['import_errors']:
                f.write("IMPORT ERRORS:\n")
                for error in results['import_errors']:
                    f.write(f"  - {error}\n")
                f.write("\n")
            
            if results['logic_errors']:
                f.write("LOGIC ERRORS:\n")
                for error in results['logic_errors']:
                    f.write(f"  - {error}\n")
                f.write("\n")
            
            if results['data_errors']:
                f.write("DATA HANDLING ERRORS:\n")
                for error in results['data_errors']:
                    f.write(f"  - {error}\n")
                f.write("\n")
            
            if results['output_errors']:
                f.write("OUTPUT FORMAT ERRORS:\n")
                for error in results['output_errors']:
                    f.write(f"  - {error}\n")
                f.write("\n")
            
            if results['warnings']:
                f.write("WARNINGS:\n")
                for warning in results['warnings']:
                    f.write(f"  - {warning}\n")
                f.write("\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("END OF TRACE LOG\n")
        
        print(f"📝 Trace log saved to {trace_log}")
        return trace_log
