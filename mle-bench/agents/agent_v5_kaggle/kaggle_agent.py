"""
KaggleAgent - Extends ResearchAgent with Kaggle competition system prompt
"""
from pathlib import Path
from agent_v5.agent import ResearchAgent
from datetime import datetime
import os


def create_kaggle_system_prompt(
    instructions_path: str,
    data_dir: str,
    submission_dir: str,
    tool_registry=None
) -> str:
    """
    Generate Kaggle-specific system prompt from modular components

    Args:
        instructions_path: Path to competition instructions
        data_dir: Path to competition data directory
        submission_dir: Path where submission.csv must be written
        tool_registry: Optional ToolRegistry for auto-generating tool docs

    Returns:
        Formatted system prompt string
    """
    # Read competition instructions
    try:
        instructions = Path(instructions_path).read_text()
    except Exception as e:
        instructions = f"(Could not read instructions: {e})"

    current_date = datetime.now().strftime("%Y-%m-%d")

    # Read core system prompt template
    prompt_dir = Path(__file__).parent / "prompts"
    template = (prompt_dir / "system_prompt.md").read_text()

    # Generate tool documentation
    if tool_registry:
        tools_doc = _generate_tool_documentation(tool_registry)
    else:
        tools_doc = "(Tools will be registered at runtime)"

    # Fill in template variables
    system_prompt = template.format(
        data_dir=data_dir,
        submission_dir=submission_dir,
        instructions=instructions,
        current_date=current_date,
        tools=tools_doc
    )

    return system_prompt


def _generate_tool_documentation(tool_registry) -> str:
    """Generate formatted tool documentation from ToolRegistry"""
    docs = []
    for tool_name, tool in sorted(tool_registry.tools.items()):
        schema = tool.schema
        description = schema.get('description', 'No description available')
        docs.append(f"**{tool_name}**: {description}")
    return "\n\n".join(docs)


class KaggleAgent(ResearchAgent):
    """Kaggle competition agent - extends ResearchAgent with Kaggle-specific prompt"""

    def __init__(
        self,
        session_id: str,
        workspace_dir: str,
        data_dir: str,
        submission_dir: str,
        instructions_path: str
    ):
        """
        Initialize Kaggle agent

        Args:
            session_id: Unique session identifier (usually competition name)
            workspace_dir: Working directory for scripts and analysis
            data_dir: Directory containing competition data
            submission_dir: Directory where submission.csv must be saved
            instructions_path: Path to competition instructions file
        """
        self.data_dir = data_dir
        self.submission_dir = submission_dir
        self.instructions_path = instructions_path

        # Initialize parent ResearchAgent (registers tools)
        super().__init__(
            session_id=session_id,
            workspace_dir=workspace_dir,
            system_prompt=""  # Temporary, will be set below
        )

        # Generate Kaggle-specific system prompt with tool documentation
        system_prompt = create_kaggle_system_prompt(
            instructions_path,
            data_dir,
            submission_dir,
            tool_registry=self.tools  # Pass registry for auto-generating tool docs
        )

        # Update system prompt with full documentation
        self.system_prompt = system_prompt
