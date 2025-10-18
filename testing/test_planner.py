import asyncio
import json
import re
import os
import sys
from pathlib import Path
from typing import Dict, List
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent_v6.prompts import format_planning_prompt


class PlannerTester:
    def __init__(self, input_file: str, output_file: str):
        self.input_file = input_file
        self.output_file = output_file
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    
    async def test_all_competitions(self):
        with open(self.input_file, 'r') as f:
            experiments_data = json.load(f)
        
        total_competitions = len(experiments_data)
        output_data = {}
        
        print(f"\n{'='*60}")
        print(f"Starting planner test for {total_competitions} competitions")
        print(f"{'='*60}\n")
        
        for idx, (competition_id, data) in enumerate(experiments_data.items(), 1):
            print(f"\n{'='*60}")
            print(f"[{idx}/{total_competitions}] Testing: {competition_id}")
            print(f"{'='*60}")
            
            eda_text = data.get('EDA', '')
            if not eda_text:
                print(f"⚠️  No EDA found for {competition_id}, skipping...")
                output_data[competition_id] = {
                    "EDA": eda_text,
                    "Plan": "ERROR: No EDA found"
                }
                continue
            
            try:
                plan = await self.run_planner(competition_id, eda_text)
                output_data[competition_id] = {
                    "EDA": eda_text,
                    "Plan": plan
                }
                print(f"\n✓ Successfully generated plan for {competition_id}")
                print(f"  Generated {len(plan)} experiments")
                
                print(f"\n💾 Saving intermediate results...")
                with open(self.output_file, 'w') as f:
                    json.dump(output_data, f, indent=2)
                print(f"  Saved to {self.output_file}")
            except Exception as e:
                print(f"\n❌ Error generating plan for {competition_id}: {str(e)}")
                import traceback
                traceback.print_exc()
                output_data[competition_id] = {
                    "EDA": eda_text,
                    "Plan": f"ERROR: {str(e)}"
                }
        
        with open(self.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Results saved to {self.output_file}")
        print(f"{'='*60}")
    
    async def run_planner(self, competition_id: str, eda_text: str) -> List[Dict]:
        prompt = format_planning_prompt(
            competition_id=competition_id,
            context=eda_text,
            round_num=1,
            best_score=0.0
        )
        
        print(f"\nCalling LLM for {competition_id}...")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                output = ""
                char_count = 0
                print(f"  Streaming response: ", end="", flush=True)
                
                with self.client.messages.stream(
                    model="claude-sonnet-4-5-20250929",
                    max_tokens=60000,
                    system=prompt,
                    messages=[{"role": "user", "content": "Output JSON array of experiments"}],
                    temperature=0
                ) as stream:
                    for event in stream:
                        if event.type == "content_block_delta":
                            if event.delta.type == "text_delta":
                                output += event.delta.text
                                char_count += len(event.delta.text)
                                if char_count % 500 == 0:
                                    print(".", end="", flush=True)
                    
                    final_message = stream.get_final_message()
                    
                    print()
                    
                    if final_message.stop_reason == "max_tokens":
                        print(f"  ⚠️  WARNING: Response hit max_tokens limit!")
                
                print(f"  ✓ Response length: {len(output)} characters")
                
                experiments = self._parse_experiments(output)
                return experiments
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  ⚠️  API error (attempt {attempt + 1}/{max_retries}): {str(e)[:100]}")
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    print(f"  ❌ API error after {max_retries} attempts: {str(e)}")
                    raise
    
    def _parse_experiments(self, output: str) -> List[Dict]:
        json_match = re.search(r'\[[\s\S]*\]', output)
        if not json_match:
            print(f"  ⚠️  No JSON array found in planning output")
            print(f"  Output preview: {output[:500]}")
            return []
        
        json_str = json_match.group(0)
        
        json_str = re.sub(r'(?<!\\)\n', ' ', json_str)
        json_str = re.sub(r'(?<!\\)\r', ' ', json_str)
        json_str = re.sub(r'(?<!\\)\t', ' ', json_str)
        
        try:
            experiments = json.loads(json_str)
            if not isinstance(experiments, list):
                print(f"  ⚠️  JSON is not a list")
                return []
            
            if not experiments:
                print(f"  ⚠️  Empty experiment list")
                return []
            
            print(f"  ✓ Parsed {len(experiments)} experiments:")
            for exp in experiments:
                print(f"    • {exp.get('id', '?')}: {exp.get('model', exp.get('models', '?'))}")
            
            return experiments
        except json.JSONDecodeError as e:
            print(f"  ⚠️  JSON parse error: {e}")
            print(f"  JSON preview (first 500): {json_str[:500]}")
            print(f"  JSON suffix (last 200): ...{json_str[-200:]}")
            return []


async def main():
    script_dir = Path(__file__).parent
    input_file = script_dir / "experiments_parsed.json"
    output_file = script_dir / "experiments_output.json"
    
    tester = PlannerTester(str(input_file), str(output_file))
    await tester.test_all_competitions()


if __name__ == "__main__":
    asyncio.run(main())

