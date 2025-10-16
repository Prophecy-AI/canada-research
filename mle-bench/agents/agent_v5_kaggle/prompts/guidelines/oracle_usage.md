# Oracle Consultation Guidelines

The Oracle tool provides access to expert ML guidance (OpenAI o3) with full conversation history context.

## When to Consult Oracle

### Always Consult

1. **Initial strategy** (after data exploration): Get competition-specific recommendations
2. **Code review** (before training >2 min): Catch bugs before wasting compute
3. **CV/leaderboard mismatch**: Diagnose data leakage or evaluation issues

### Recommended to Consult

4. **Unexpected results**: Model performs worse than baseline
5. **Stuck after 2-3 iterations**: Need fundamental strategy pivot
6. **Complex debugging**: Label encoding, column order, or preprocessing bugs

### Do Not Consult

- During active training (wait for completion)
- For basic syntax questions (use documentation)
- For minor hyperparameter tweaks (iterate yourself)

## How to Consult Effectively

### Provide Context

Share what you've tried and what happened:
- Data characteristics you discovered
- Models and approaches you tested
- Cross-validation vs leaderboard scores
- Error messages or unexpected behaviors

### Ask Specific Questions

Good questions:
- "Why might my CV be 0.85 but leaderboard score 0.62?"
- "Should I use gradient boosting or neural networks for this tabular time-series problem?"
- "What's wrong with this label encoding code?"

Vague questions:
- "How do I improve my score?"
- "What should I try next?"
- "Is this good?"

### Trust Oracle's Reasoning

Oracle uses deep reasoning (o3 model) and has full conversation context.
If Oracle identifies issues or suggests pivots, take them seriously - the model has seen patterns from thousands of competitions.

## Oracle Limitations

- Cannot see your files directly (share relevant code snippets)
- Cannot execute code or run experiments
- May not know about brand new packages or techniques
- Uses reasoning tokens, so responses may take 30-60 seconds
