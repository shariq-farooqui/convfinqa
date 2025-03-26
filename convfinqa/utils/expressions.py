from py_expression_eval import Parser

from convfinqa import logger


class ExpressionEvaluator:
    """A safe evaluator for mathematical expressions using py_expression_eval."""

    def __init__(self):
        """Initialise the expression parser."""
        self.parser = Parser()

    def evaluate(
        self, expression: str, decimal_precision: int | None = None
    ) -> float:
        """Evaluate a mathematical expression using py_expression_eval.

        Args:
            expression: A string containing a mathematical expression
            decimal_precision: Determines how many decimal places to keep after rounding

        Returns:
            The calculated result as a float
        """
        try:
            cleaned_expr = expression.strip()
            answer = self.parser.parse(cleaned_expr).evaluate({})
            return (
                round(answer, decimal_precision)
                if decimal_precision is not None
                else answer
            )
        except Exception as e:
            logger.error(
                "Error evaluating expression",
                extra={"expression": expression, "error": str(e)},
            )
            return float("nan")
