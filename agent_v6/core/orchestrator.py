"""
Orchestrator: Router agent that classifies competitions into 16 MLE-bench types + general-else
"""

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from agent_v6.core.agent import Agent
from agent_v6.core.tools import ToolRegistry, BashTool, ReadTool, WriteTool


class Orchestrator:
    """Router orchestrator for ML competitions"""
    
    COMPETITION_TYPES = {
        "image-classification": "Image classification tasks",
        "text-classification": "Text classification tasks", 
        "tabular-regression": "Tabular data regression",
        "tabular-classification": "Tabular data classification",
        "time-series-forecasting": "Time series forecasting",
        "object-detection": "Object detection in images",
        "image-segmentation": "Image segmentation tasks",
        "image-to-text": "Image captioning/OCR tasks",
        "text-generation": "Text generation tasks",
        "recommender-systems": "Recommendation systems",
        "reinforcement-learning": "RL/game playing tasks",
        "clustering": "Unsupervised clustering",
        "anomaly-detection": "Anomaly/outlier detection",
        "graph-ml": "Graph neural networks",
        "audio-processing": "Audio/speech tasks",
        "video-processing": "Video analysis tasks",
        "general-else": "General/unclassified tasks"
    }
    
    def __init__(
        self,
        competition_id: str,
        data_dir: str,
        submission_dir: str,
        workspace_dir: str,
        instructions_path: str,
        time_limit_minutes: int = 240
    ):
        self.competition_id = competition_id
        self.data_dir = Path(data_dir)
        self.submission_dir = Path(submission_dir)
        self.workspace_dir = Path(workspace_dir)
        self.instructions_path = Path(instructions_path)
        self.time_limit_minutes = time_limit_minutes
        
        self.competition_type = None
        self.evaluation_metric = None
        self.data_description = {}
        
        self.notebooks_dir = self.workspace_dir / "notebooks"
        self.logs_dir = self.workspace_dir / "logs"
        self.augmented_data_dir = self.workspace_dir / "augmented_data"
        
        self.notebooks_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.augmented_data_dir.mkdir(parents=True, exist_ok=True)
        
        self.score_file = self.workspace_dir / "score.txt"
        
        from agent_v6.core.fallback import FallbackManager
        self.fallback = FallbackManager(workspace_dir, data_dir, submission_dir)
        
    async def run(self):
        """Main orchestration flow"""
        print("\n" + "="*60)
        print(f"ORCHESTRATOR: {self.competition_id}")
        print("="*60 + "\n")
        
        # Import clock for time management
        from agent_v6.utils.clock import Clock
        clock = Clock(self.time_limit_minutes)
        clock.start()
        
        try:
            # Phase 1: Explore and classify
            print("🔍 Phase 1: Exploration and Classification")
            await self._explore_and_classify()
            print(f"   Competition Type: {self.competition_type}")
            print(f"   Evaluation Metric: {self.evaluation_metric}")
            
            iteration = 0
            max_iterations = 10  # Prevent infinite loops
            
            while iteration < max_iterations:
                iteration += 1
                print(f"\n🔄 Iteration {iteration}")
                
                # Phase 2: Generate solution with prompts
                print("📝 Phase 2: Solution Generation")
                notebook_path = await self._generate_solution(iteration)
                
                # Phase 3: Data augmentation
                print("🔧 Phase 3: Data Augmentation")
                augmented_notebook = await self._augment_data(notebook_path, iteration)
                
                # Phase 4: Improve solution
                print("✨ Phase 4: Solution Improvement")
                improved_notebook = await self._improve_solution(augmented_notebook, iteration)
                
                # Phase 5: Verification loop
                print("✅ Phase 5: Verification")
                final_notebook = await self._verify_and_fix(improved_notebook, iteration)
                
                # Phase 6: Execute and score
                print("🏃 Phase 6: Execution")
                score = await self._execute_and_score(final_notebook, iteration)
                
                # Log score
                if score is not None:
                    self._log_score(iteration, score, final_notebook)
                    print(f"   Score: {score}")
                
                # Check time and submission status
                clock.checkpoint()
                
                # Check if we should stop or continue
                if clock.should_stop():
                    # Check if we have a submission
                    submission_file = self.submission_dir / "submission.csv"
                    if not submission_file.exists():
                        # No submission yet, check if we should continue in overtime
                        if not self.fallback.extend_time_if_needed(clock):
                            break  # Max overtime exceeded
                        print("⏰ Continuing in OVERTIME mode to create submission...")
                    else:
                        break  # We have submission and time is up
                
        except Exception as e:
            print(f"❌ Error in orchestration: {e}")
            import traceback
            traceback.print_exc()
            
            # Try to create emergency submission on crash
            try:
                await self.fallback.create_emergency_submission()
            except:
                pass
        
        finally:
            # Phase 7: Submit best solution or create emergency submission
            print("\n📤 Phase 7: Final Submission")
            
            # Check if submission already exists
            submission_file = self.submission_dir / "submission.csv"
            if submission_file.exists():
                print(f"✅ Submission already exists: {submission_file}")
                return
            
            # Try to submit best scored solution
            submission_created = await self._submit_best()
            
            # If no submission yet, try any notebook
            if not submission_created:
                print("⚠️ No scored submission, trying any available notebook...")
                submission_created = await self._submit_any_available_notebook()
            
            # Last resort: create emergency submission
            if not submission_created:
                print("🆘 Creating emergency submission...")
                await self.fallback.create_emergency_submission()
    
    async def _explore_and_classify(self):
        """Run exploration to classify competition type"""
        try:
            from agent_v6.exploration.explore import Explorer
            
            explorer = Explorer(
                self.data_dir,
                self.workspace_dir,
                self.instructions_path
            )
            
            # Run exploration with timeout
            exploration_results = await asyncio.wait_for(
                explorer.explore(),
                timeout=300  # 5 minute timeout
            )
            
            # Extract classification
            self.competition_type = exploration_results.get("competition_type", "general-else")
            self.evaluation_metric = exploration_results.get("evaluation_metric", "unknown")
            self.data_description = exploration_results.get("data_description", {})
            
            # Save exploration results
            results_file = self.logs_dir / "exploration.json"
            with open(results_file, 'w') as f:
                json.dump(exploration_results, f, indent=2)
                
        except asyncio.TimeoutError:
            print("⚠️ Exploration timed out, defaulting to general-else")
            self.competition_type = "general-else"
            self.evaluation_metric = "unknown"
            self.data_description = {"timeout": True}
            
        except Exception as e:
            print(f"⚠️ Exploration failed: {e}, defaulting to general-else")
            self.competition_type = "general-else"
            self.evaluation_metric = "unknown"
            self.data_description = {"error": str(e)}
    
    async def _generate_solution(self, iteration: int) -> Path:
        """Generate solution using appropriate prompts"""
        # Import prompt modules
        from agent_v6.exploration.prompt import GeneralPrompt
        
        # Get specialized prompt based on competition type
        specialized_prompt = self._get_specialized_prompt()
        
        # Combine prompts
        general = GeneralPrompt(
            self.competition_type,
            self.data_description,
            self.evaluation_metric
        )
        
        # Generate notebook
        notebook_content = await general.generate(
            specialized_context=specialized_prompt.get_context(),
            iteration=iteration
        )
        
        # Save as Jupyter notebook
        notebook_path = self.notebooks_dir / f"solution_v{iteration}.ipynb"
        self._save_notebook(notebook_content, notebook_path)
        
        return notebook_path
    
    def _get_specialized_prompt(self):
        """Get specialized prompt module based on competition type"""
        # Map competition types to prompt modules
        prompt_mapping = {
            "image-classification": "image_prompt",
            "text-classification": "text_prompt",
            "tabular-regression": "tabular_prompt",
            "tabular-classification": "tabular_prompt",
            "time-series-forecasting": "timeseries_prompt",
            "object-detection": "object_detection_prompt",
            "image-segmentation": "segmentation_prompt",
            "image-to-text": "image_text_prompt",
            "text-generation": "text_generation_prompt",
            "recommender-systems": "recommender_prompt",
            "reinforcement-learning": "rl_prompt",
            "clustering": "clustering_prompt",
            "anomaly-detection": "anomaly_prompt",
            "graph-ml": "graph_prompt",
            "audio-processing": "audio_prompt",
            "video-processing": "video_prompt",
            "general-else": "general_specialized_prompt"
        }
        
        prompt_module = prompt_mapping.get(self.competition_type, "general_specialized_prompt")
        
        # Dynamic import
        try:
            module = __import__(f"agent_v6.prompts.{prompt_module}", fromlist=["SpecializedPrompt"])
            return module.SpecializedPrompt(self.data_description)
        except (ImportError, AttributeError) as e:
            print(f"⚠️ Could not load specialized prompt {prompt_module}: {e}")
            # Fallback to general
            try:
                from agent_v6.prompts.general_specialized_prompt import SpecializedPrompt
                return SpecializedPrompt(self.data_description)
            except Exception as e:
                print(f"⚠️ Even fallback failed: {e}")
                # Return dummy prompt
                class DummyPrompt:
                    def __init__(self, data_desc):
                        pass
                    def get_context(self):
                        return "# No specialized context available\n# Using basic template\n"
                return DummyPrompt(self.data_description)
    
    async def _augment_data(self, notebook_path: Path, iteration: int) -> Path:
        """Run data augmentation"""
        from agent_v6.augmentation.augment import DataAugmenter
        
        augmenter = DataAugmenter(
            self.data_dir,
            self.augmented_data_dir,
            self.competition_type
        )
        
        # Augment data
        augmented_paths = await augmenter.augment(iteration)
        
        # Update notebook to use augmented data
        augmented_notebook = self.notebooks_dir / f"augmented_v{iteration}.ipynb"
        self._update_notebook_data_paths(notebook_path, augmented_notebook, augmented_paths)
        
        return augmented_notebook
    
    async def _improve_solution(self, notebook_path: Path, iteration: int) -> Path:
        """Improve the solution"""
        from agent_v6.augmentation.improve import SolutionImprover
        
        improver = SolutionImprover(
            self.competition_type,
            self.evaluation_metric
        )
        
        # Load notebook
        notebook_content = self._load_notebook(notebook_path)
        
        # Improve
        improved_content = await improver.improve(notebook_content, iteration)
        
        # Save improved notebook
        improved_path = self.notebooks_dir / f"improved_v{iteration}.ipynb"
        self._save_notebook(improved_content, improved_path)
        
        return improved_path
    
    async def _verify_and_fix(self, notebook_path: Path, iteration: int) -> Path:
        """Verification and fixing loop"""
        from agent_v6.verification.verify import Verifier
        from agent_v6.verification.fixer import Fixer
        
        verifier = Verifier(self.logs_dir)
        fixer = Fixer()
        
        max_fix_attempts = 3
        current_path = notebook_path
        
        for attempt in range(max_fix_attempts):
            # Verify
            verification_result = await verifier.verify(current_path, iteration, attempt)
            
            if verification_result["is_valid"]:
                print(f"   ✅ Verification passed")
                return current_path
            
            print(f"   ⚠️ Issues found, attempting fix {attempt + 1}/{max_fix_attempts}")
            
            # Fix
            fixed_content = await fixer.fix(
                self._load_notebook(current_path),
                verification_result["errors"]
            )
            
            # Save fixed notebook
            fixed_path = self.notebooks_dir / f"fixed_v{iteration}_attempt{attempt + 1}.ipynb"
            self._save_notebook(fixed_content, fixed_path)
            current_path = fixed_path
        
        print(f"Max fix attempts reached, using best effort")
        return current_path
    
    async def _execute_and_score(self, notebook_path: Path, iteration: int) -> Optional[float]:
        """Execute notebook and extract score"""
        # Convert notebook to Python script
        script_path = self.workspace_dir / f"execute_v{iteration}.py"
        self._notebook_to_script(notebook_path, script_path)
        
        # Execute
        try:
            process = await asyncio.create_subprocess_shell(
                f"cd {self.workspace_dir} && python {script_path.name}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            # Extract score from output
            output = stdout.decode() + stderr.decode()
            score_match = re.search(r"VALIDATION_SCORE:\s*(\d+\.?\d*)", output)
            
            if score_match:
                return float(score_match.group(1))
            
            print(f"   ⚠️ No score found in output")
            
        except Exception as e:
            print(f"   ❌ Execution failed: {e}")
        
        return None
    
    def _log_score(self, iteration: int, score: float, notebook_path: Path):
        """Log score to file"""
        with open(self.score_file, 'a') as f:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "iteration": iteration,
                "score": score,
                "notebook": str(notebook_path),
                "competition_type": self.competition_type,
                "evaluation_metric": self.evaluation_metric
            }
            f.write(json.dumps(entry) + "\n")
    
    async def _submit_best(self) -> bool:
        """Submit the best solution. Returns True if submission created."""
        if not self.score_file.exists():
            print("⚠️ No scores found")
            return False
        
        # Read all scores
        scores = []
        try:
            with open(self.score_file, 'r') as f:
                for line in f:
                    try:
                        scores.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # Skip malformed lines
        except Exception as e:
            print(f"⚠️ Error reading scores: {e}")
            return False
        
        if not scores:
            print("⚠️ No valid scores")
            return False
        
        # Find best score
        best_entry = max(scores, key=lambda x: x.get("score", 0))
        print(f"🏆 Best score: {best_entry['score']} from iteration {best_entry['iteration']}")
        
        # Load best notebook
        best_notebook = Path(best_entry["notebook"])
        
        if not best_notebook.exists():
            print(f"⚠️ Best notebook not found: {best_notebook}")
            return False
        
        # Generate submission
        success = await self._generate_submission(best_notebook)
        return success
    
    async def _generate_submission(self, notebook_path: Path):
        """Generate submission from notebook"""
        # Convert to submission script
        submission_script = self.workspace_dir / "generate_submission.py"
        
        # Add submission generation code
        script_content = self._notebook_to_script(notebook_path, None, return_content=True)
        script_content += f"""
# Generate submission
import pandas as pd

# Load test data
test_data = pd.read_csv('{self.data_dir}/test.csv')

# Generate predictions (model should be defined above)
predictions = model.predict(test_data)

# Create submission
submission = pd.DataFrame({{
    'id': test_data.index,
    'prediction': predictions
}})

submission.to_csv('{self.submission_dir}/submission.csv', index=False)
print(f"✅ Submission saved to {self.submission_dir}/submission.csv")
"""
        
        submission_script.write_text(script_content)
        
        # Execute submission generation
        process = await asyncio.create_subprocess_shell(
            f"cd {self.workspace_dir} && python {submission_script.name}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        # Check if submission was created
        submission_file = self.submission_dir / "submission.csv"
        if submission_file.exists():
            print(f"✅ Submission created successfully")
            return True
        else:
            print(f"❌ Submission generation failed")
            if stderr:
                print(f"Error: {stderr.decode()[:1000]}")
            return False
    
    # Helper methods for notebook handling
    
    def _save_notebook(self, content: Dict, path: Path):
        """Save content as Jupyter notebook"""
        import nbformat
        
        if isinstance(content, dict) and "cells" in content:
            # Already a notebook format
            notebook = nbformat.from_dict(content)
        else:
            # Create new notebook with content
            notebook = nbformat.v4.new_notebook()
            if isinstance(content, str):
                # Single code cell
                notebook.cells = [nbformat.v4.new_code_cell(content)]
            elif isinstance(content, list):
                # Multiple cells
                notebook.cells = content
        
        nbformat.write(notebook, str(path))
    
    def _load_notebook(self, path: Path) -> Dict:
        """Load Jupyter notebook"""
        import nbformat
        
        with open(path, 'r') as f:
            notebook = nbformat.read(f, as_version=4)
        
        return notebook
    
    def _update_notebook_data_paths(self, input_path: Path, output_path: Path, new_paths: Dict):
        """Update data paths in notebook"""
        notebook = self._load_notebook(input_path)
        
        # Update cells with new paths
        for cell in notebook.cells:
            if cell.cell_type == "code":
                for old_path, new_path in new_paths.items():
                    cell.source = cell.source.replace(str(old_path), str(new_path))
        
        self._save_notebook(notebook, output_path)
    
    def _notebook_to_script(self, notebook_path: Path, script_path: Optional[Path], 
                           return_content: bool = False) -> Optional[str]:
        """Convert notebook to Python script"""
        notebook = self._load_notebook(notebook_path)
        
        # Extract code cells
        code_lines = []
        for cell in notebook.cells:
            if cell.cell_type == "code":
                code_lines.append(cell.source)
                code_lines.append("\n")
        
        script_content = "\n".join(code_lines)
        
        if return_content:
            return script_content
        
        if script_path:
            script_path.write_text(script_content)
        
        return None
    
    async def _submit_any_available_notebook(self) -> bool:
        """Try to submit any available notebook"""
        notebooks_dir = self.notebooks_dir
        
        # Find any notebook
        notebook_path = await self.fallback.find_any_notebook(notebooks_dir)
        
        if notebook_path:
            print(f"📓 Attempting to submit: {notebook_path.name}")
            return await self._generate_submission(notebook_path)
        else:
            # Create a basic notebook and try to submit it
            print("🔨 No notebooks found, creating basic solution...")
            fallback_notebook = await self.fallback.create_basic_notebook()
            return await self._generate_submission(fallback_notebook)