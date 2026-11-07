# AgentLab Architecture Notes

Fill this in as you build, don't write it all upfront speculatively,
document what you actually built, not what the original spec imagined.

## Sections to fill in as each phase completes

- High-level pipeline diagram (Mermaid), update this after
  graph/workflow.py is wired in week 11
- Per-agent input/output contract (copy from each agent's docstring
  once finalized, docstrings drift from reality if you don't keep
  them in sync)
- Critic Agent decision logic, this is the section that matters most
  for your report, document the exact thresholds you land on and why
- Comparison results: critic-active vs critic-stubbed-out runs across
  your test datasets, this is your actual evidence for the research
  question in the abstract
