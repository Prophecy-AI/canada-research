"""
KaggleAgent - Extends IDEAgent with Kaggle competition capabilities

Based on Operand Quant architecture:
- Single-agent, IDE-based system
- Deep-Thinking Ensemble (GPT-5, Claude Opus, Grok-4, Gemini 2.5 Pro + O3)
- Non-blocking execution with monitoring
- Jupyter notebook first-class support
"""
from agent_v6.agent import IDEAgent
from kaggle_prompt import create_kaggle_system_prompt


class KaggleAgent(IDEAgent):
    """
    Kaggle competition agent using Operand Quant architecture

    Extends IDEAgent with Kaggle-specific:
    - Competition-focused system prompt
    - MLE lifecycle guidance (EDA, modeling, evaluation, submission)
    - Ensemble consultation best practices
    - Resource optimization for medal achievement
    """

    def __init__(
        self,
        session_id: str,
        workspace_dir: str,
        data_dir: str,
        submission_dir: str,
        instructions_path: str,
        enable_memory_compaction: bool = True
    ):
        """
        Initialize Kaggle agent

        Args:
            session_id: Unique session identifier (usually competition name)
            workspace_dir: Working directory for scripts and analysis
            data_dir: Directory containing competition data
            submission_dir: Directory where submission.csv must be saved
            instructions_path: Path to competition instructions file
            enable_memory_compaction: Enable hierarchical memory compaction (default: True)
        """
        self.data_dir = data_dir
        self.submission_dir = submission_dir
        self.instructions_path = instructions_path

        # Generate Kaggle-specific system prompt
        system_prompt = create_kaggle_system_prompt(
            instructions_path,
            data_dir,
            submission_dir
        )

        # Initialize parent IDEAgent with Kaggle prompt
        # IDEAgent provides:
        # - Deep-Thinking Ensemble (via EnsembleTool)
        # - Jupyter notebooks (via NotebookTool)
        # - Non-blocking execution (via ExecuteScriptTool)
        # - Process monitoring (via CheckProcessTool, InterruptProcessTool)
        # - Memory compaction (via MemoryCompactor)
        # - IDE workspace state tracking (via IDEWorkspace)
        super().__init__(
            session_id=session_id,
            workspace_dir=workspace_dir,
            system_prompt=system_prompt,
            enable_memory_compaction=enable_memory_compaction
        )
