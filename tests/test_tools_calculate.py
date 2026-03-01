"""Tests for the calculate tool in pravah.tools."""

import pytest

try:
    from pravah.tools import calculate
except ImportError:
    pytest.skip("langchain_core not installed", allow_module_level=True)


class TestCalculate:
    def test_basic_addition(self):
        result = calculate.invoke({"expression": "2 + 3"})
        assert "= 5" in result

    def test_multiplication(self):
        result = calculate.invoke({"expression": "6 * 7"})
        assert "= 42" in result

    def test_division(self):
        result = calculate.invoke({"expression": "10 / 3"})
        assert "3.333" in result

    def test_power(self):
        result = calculate.invoke({"expression": "2 ** 10"})
        assert "= 1024" in result

    def test_caret_power(self):
        result = calculate.invoke({"expression": "2 ^ 10"})
        assert "= 1024" in result

    def test_sqrt(self):
        result = calculate.invoke({"expression": "sqrt(16)"})
        assert "= 4" in result

    def test_pi(self):
        result = calculate.invoke({"expression": "pi"})
        assert "3.14" in result

    def test_complex_expression(self):
        result = calculate.invoke({"expression": "2 * (3 + 4)"})
        assert "= 14" in result

    def test_invalid_expression(self):
        result = calculate.invoke({"expression": "undefined_func(5)"})
        assert "Could not calculate" in result

    def test_division_by_zero(self):
        result = calculate.invoke({"expression": "1 / 0"})
        assert "Could not calculate" in result

    def test_negative_numbers(self):
        result = calculate.invoke({"expression": "-5 + 3"})
        assert "= -2" in result

    def test_trig_functions(self):
        result = calculate.invoke({"expression": "sin(0)"})
        assert "= 0" in result

    def test_log(self):
        result = calculate.invoke({"expression": "log(1)"})
        assert "= 0" in result

    def test_no_builtins_access(self):
        # Should not allow access to builtins
        result = calculate.invoke({"expression": "__import__('os')"})
        assert "Could not calculate" in result
