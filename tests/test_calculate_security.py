import sys
from unittest.mock import MagicMock

# Mocking missing dependencies before importing pravah.tools to allow unit tests in restricted environment
sys.modules["langchain_core"] = MagicMock()
def mock_tool(fn):
    return fn
sys.modules["langchain_core.tools"] = MagicMock()
sys.modules["langchain_core.tools"].tool = mock_tool
sys.modules["langchain_core.runnables"] = MagicMock()
sys.modules["pravah.search"] = MagicMock()
sys.modules["pravah.llm"] = MagicMock()
sys.modules["pravah.memory"] = MagicMock()

from pravah.tools import calculate

def test_calculate_valid_expressions():
    """Verify that standard math expressions work as expected."""
    assert "2 + 2 = 4" in calculate("2 + 2")
    assert "sqrt(16) = 4.0" in calculate("sqrt(16)")
    assert "2**3 = 8" in calculate("2**3")
    assert "2^3 = 8" in calculate("2^3")
    assert "sin(pi/2) = 1.0" in calculate("sin(pi/2)")

def test_calculate_security_blocked():
    """Verify that attempts to escape the sandbox are blocked."""
    # Attribute access should be blocked
    payload = "(1).__class__.__mro__[1].__subclasses__()"
    result = calculate(payload)
    assert "Could not calculate" in result
    assert "Unsupported operation: Attribute" in result

    # List comprehension should be blocked
    payload = "[x for x in (1, 2, 3)]"
    result = calculate(payload)
    assert "Could not calculate" in result
    assert "Unsupported operation: ListComp" in result

    # Builtin access should be blocked
    payload = "__builtins__"
    result = calculate(payload)
    assert "Could not calculate" in result
    assert "Unknown name: __builtins__" in result

if __name__ == "__main__":
    # Manual test runner
    print("Running security tests for 'calculate' tool...")

    try:
        test_calculate_valid_expressions()
        print("test_calculate_valid_expressions: PASSED")
    except AssertionError as e:
        print(f"test_calculate_valid_expressions: FAILED: {e}")
        sys.exit(1)

    try:
        test_calculate_security_blocked()
        print("test_calculate_security_blocked: PASSED (Exploits blocked)")
    except AssertionError as e:
        print(f"test_calculate_security_blocked: FAILED: {e}")
        sys.exit(1)

    print("All security tests passed.")
