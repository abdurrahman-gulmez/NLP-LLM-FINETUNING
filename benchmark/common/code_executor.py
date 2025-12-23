"""
Code Execution and Evaluation Utilities

Author: naholav

Provides safe code execution with timeout and resource limits.
Used for evaluating generated solutions against test cases.
"""

import sys
import subprocess
import tempfile
import os
import resource
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    output: str
    error: str
    returncode: int
    timed_out: bool
    execution_time: float


def execute_code_subprocess(
    code: str,
    input_data: str = "",
    timeout: float = 10.0,
    memory_limit_mb: int = 512
) -> ExecutionResult:
    """
    Execute Python code in a subprocess with timeout and memory limits.

    Args:
        code: Python code to execute
        input_data: Input to provide via stdin
        timeout: Timeout in seconds
        memory_limit_mb: Maximum memory in MB

    Returns:
        ExecutionResult with output, errors, and status
    """
    import time

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        code_file = f.name

    try:
        start_time = time.time()

        process = subprocess.Popen(
            [sys.executable, code_file],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=lambda: resource.setrlimit(
                resource.RLIMIT_AS,
                (memory_limit_mb * 1024 * 1024, memory_limit_mb * 1024 * 1024)
            ) if sys.platform != 'win32' else None
        )

        try:
            stdout, stderr = process.communicate(input=input_data, timeout=timeout)
            execution_time = time.time() - start_time

            return ExecutionResult(
                success=(process.returncode == 0),
                output=stdout,
                error=stderr,
                returncode=process.returncode,
                timed_out=False,
                execution_time=execution_time
            )

        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            return ExecutionResult(
                success=False,
                output="",
                error="Execution timed out",
                returncode=-1,
                timed_out=True,
                execution_time=timeout
            )

    except Exception as e:
        return ExecutionResult(
            success=False,
            output="",
            error=str(e),
            returncode=-1,
            timed_out=False,
            execution_time=0.0
        )

    finally:
        try:
            os.unlink(code_file)
        except:
            pass


def compare_outputs(expected: str, actual: str, strict: bool = False) -> bool:
    """
    Compare expected and actual outputs.

    Args:
        expected: Expected output
        actual: Actual output from code execution
        strict: If True, require exact match; if False, allow whitespace differences

    Returns:
        True if outputs match
    """
    if strict:
        return expected == actual

    expected_normalized = '\n'.join(line.strip() for line in expected.strip().split('\n'))
    actual_normalized = '\n'.join(line.strip() for line in actual.strip().split('\n'))

    return expected_normalized == actual_normalized


def evaluate_solution(
    code: str,
    test_cases: List[Dict[str, str]],
    timeout_per_case: float = 5.0,
    memory_limit_mb: int = 512
) -> Dict[str, Any]:
    """
    Evaluate a solution against multiple test cases.

    Args:
        code: Python code to evaluate
        test_cases: List of {"input": str, "output": str} dicts
        timeout_per_case: Timeout per test case in seconds
        memory_limit_mb: Memory limit in MB

    Returns:
        Evaluation results including pass rate
    """
    results = {
        "total": len(test_cases),
        "passed": 0,
        "failed": 0,
        "timeout": 0,
        "error": 0,
        "details": []
    }

    for i, test_case in enumerate(test_cases):
        input_data = test_case.get("input", "")
        expected_output = test_case.get("output", "")

        exec_result = execute_code_subprocess(
            code,
            input_data=input_data,
            timeout=timeout_per_case,
            memory_limit_mb=memory_limit_mb
        )

        case_result = {
            "test_case": i,
            "input_preview": input_data[:100] + "..." if len(input_data) > 100 else input_data,
            "expected_preview": expected_output[:100] + "..." if len(expected_output) > 100 else expected_output,
            "actual_preview": exec_result.output[:100] + "..." if len(exec_result.output) > 100 else exec_result.output,
            "execution_time": exec_result.execution_time
        }

        if exec_result.timed_out:
            results["timeout"] += 1
            case_result["status"] = "timeout"
        elif not exec_result.success:
            results["error"] += 1
            case_result["status"] = "error"
            case_result["error_message"] = exec_result.error
        elif compare_outputs(expected_output, exec_result.output):
            results["passed"] += 1
            case_result["status"] = "passed"
        else:
            results["failed"] += 1
            case_result["status"] = "failed"

        results["details"].append(case_result)

    results["pass_rate"] = results["passed"] / results["total"] if results["total"] > 0 else 0.0
    results["all_passed"] = results["passed"] == results["total"]

    return results


def batch_evaluate(
    solutions: List[Dict[str, Any]],
    timeout_per_case: float = 5.0,
    memory_limit_mb: int = 512,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate multiple solutions in batch.

    Args:
        solutions: List of {"question_id": str, "code": str, "test_cases": List} dicts
        timeout_per_case: Timeout per test case
        memory_limit_mb: Memory limit
        verbose: Print progress

    Returns:
        Batch evaluation results with pass@1 score
    """
    from tqdm import tqdm

    results = {
        "total_problems": len(solutions),
        "problems_passed": 0,
        "problems_failed": 0,
        "problem_results": []
    }

    iterator = tqdm(solutions, desc="Evaluating") if verbose else solutions

    for solution in iterator:
        question_id = solution.get("question_id", "unknown")
        code = solution.get("code", "")
        test_cases = solution.get("test_cases", [])

        if not test_cases:
            results["problem_results"].append({
                "question_id": question_id,
                "status": "no_tests",
                "passed": False
            })
            results["problems_failed"] += 1
            continue

        eval_result = evaluate_solution(
            code, test_cases, timeout_per_case, memory_limit_mb
        )

        problem_result = {
            "question_id": question_id,
            "passed": eval_result["all_passed"],
            "pass_rate": eval_result["pass_rate"],
            "tests_passed": eval_result["passed"],
            "tests_total": eval_result["total"]
        }

        if eval_result["all_passed"]:
            results["problems_passed"] += 1
            problem_result["status"] = "passed"
        else:
            results["problems_failed"] += 1
            problem_result["status"] = "failed"

        results["problem_results"].append(problem_result)

    results["pass_at_1"] = (
        results["problems_passed"] / results["total_problems"]
        if results["total_problems"] > 0 else 0.0
    )

    return results
