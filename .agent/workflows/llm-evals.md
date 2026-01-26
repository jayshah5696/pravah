---
description: LLM Evaluation Best Practices (Based on Hamel Husain & Eugene Yan)
---

# LLM Evaluation Best Practices

This skill provides a comprehensive guide for evaluating LLM-powered applications, synthesized from Hamel Husain and Eugene Yan's research.

## Core Principles

### 1. Start with Manual Trace Review (Most Important!)
> "Manually review 100+ AI conversations before building any automated evals" - Hamel

- Review actual traces/conversations, not just outputs
- Look at: inputs, AI responses, tool calls, errors
- Domain experts should do the review, not outsourced

### 2. Use Binary Pass/Fail, Not Scales
```
❌ Bad:  "Rate helpfulness 1-5"
✅ Good: "Did the response answer the user's question? Pass/Fail"
```

Always include a **critique** explaining WHY something passed/failed.

### 3. Store Full Trajectories (Traces)
A trace should capture:
- User query
- All tool calls (inputs + outputs)
- Intermediate reasoning
- Final response
- Latency
- Errors

Example structure:
```json
{
  "query": "What is the weather in SF?",
  "tool_calls": [
    {"name": "web_search", "input": "weather SF", "output": "..."}
  ],
  "final_response": "The weather in SF is...",
  "latency_seconds": 2.3,
  "errors": []
}
```

---

## Evaluation Checklist

### Phase 1: Setup
- [ ] Define 3-5 **specific** failure modes (not generic like "helpfulness")
- [ ] Create a seed dataset of 30-50 representative queries
- [ ] Build a trace storage system (JSON files or DB)

### Phase 2: Manual Review
- [ ] Review 100+ traces manually
- [ ] Categorize failures into 5-6 buckets (open coding)
- [ ] Create a spreadsheet with Pass/Fail + Critique columns

### Phase 3: Automated Checks
```python
# Priority order for automated checks:

# 1. Code-based assertions (cheapest)
def check_has_citations(response: str) -> bool:
    """Check if response contains citations [Source](URL)"""
    import re
    return bool(re.search(r'\[.+?\]\(https?://.+?\)', response))

# 2. Structural checks
def check_tool_was_called(trace: dict, tool_name: str) -> bool:
    """Check if a specific tool was invoked"""
    return any(t["name"] == tool_name for t in trace["tool_calls"])

# 3. LLM-as-Judge (only for complex cases)
def llm_judge(query: str, response: str, criteria: str) -> bool:
    """Use LLM to evaluate if response meets criteria"""
    # Binary pass/fail, not scales!
    pass
```

### Phase 4: Continuous Monitoring
- [ ] Run evals on 5-10% of production traffic
- [ ] Weekly review of new failure modes
- [ ] Update judges when product changes

---

## Specific Checks for Search/RAG Agents

### Citation Checks
```python
def eval_citations(response: str, tool_results: list) -> dict:
    """
    Check if citations in response match actual sources.
    Returns: {"has_citations": bool, "valid_citations": int, "total_links": int}
    """
    import re
    links_in_response = re.findall(r'\[.+?\]\((https?://[^\)]+)\)', response)
    sources_from_tools = [r.get('url', '') for t in tool_results for r in t.get('results', [])]
    
    valid = sum(1 for link in links_in_response if link in sources_from_tools)
    return {
        "has_citations": len(links_in_response) > 0,
        "valid_citations": valid,
        "total_links": len(links_in_response)
    }
```

### Tool Usage Checks
```python
def eval_tool_usage(trace: dict, expected_behavior: str) -> bool:
    """
    Check if tools were used appropriately.
    - "No Search" queries should not trigger web_search
    - "Search" queries should trigger web_search
    """
    tool_names = [t["name"] for t in trace["tool_calls"]]
    
    if expected_behavior == "No Search":
        return "web_search" not in tool_names
    elif expected_behavior in ["Web Search", "Search"]:
        return "web_search" in tool_names
    return True  # Default pass
```

---

## Eugene Yan's Scientific Method for Evals

1. **Observe** → Look at traces, find failures
2. **Annotate** → Label failures with categories
3. **Hypothesize** → Why is this failing?
4. **Experiment** → Try a fix (prompt, retrieval, model)
5. **Measure** → Did the fix improve the metric?
6. **Iterate** → Repeat

---

## References
- [Hamel Husain - LLM Evals: Everything You Need to Know](https://hamel.dev/blog/posts/evals/)
- [Eugene Yan - LLM Evaluation](https://eugeneyan.com/)
- [Hamel's AI Evals Course](https://maven.com/applied-llms/ai-evals)
