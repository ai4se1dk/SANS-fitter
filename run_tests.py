"""
Test runner script for SANS Fitter unit tests.

Usage:
    python run_tests.py           # Run all tests
    python run_tests.py -v        # Run with verbose output
    python run_tests.py -k test_name  # Run specific test
"""

import os
import sys
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_tests(verbosity=2, pattern='test*.py'):
    """
    Run all unit tests.

    Args:
        verbosity: Test output verbosity (0=quiet, 1=normal, 2=verbose)
        pattern: Pattern for test file discovery

    Returns:
        TestResult object
    """
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.join(os.path.dirname(__file__), 'tests')
    suite = loader.discover(start_dir, pattern=pattern)

    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run SANS Fitter unit tests')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('-q', '--quiet', action='store_true', help='Quiet output')
    parser.add_argument(
        '-p', '--pattern', default='test*.py', help='Test file pattern (default: test*.py)'
    )
    parser.add_argument('-k', '--test', default=None, help='Run specific test (pattern matching)')

    args = parser.parse_args()

    # Determine verbosity
    if args.quiet:
        verbosity = 0
    elif args.verbose:
        verbosity = 2
    else:
        verbosity = 1

    # Run specific test if requested
    if args.test:
        loader = unittest.TestLoader()
        start_dir = os.path.join(os.path.dirname(__file__), 'tests')
        suite = loader.discover(start_dir, pattern=args.pattern)

        # Filter tests based on pattern
        filtered_suite = unittest.TestSuite()
        for test_group in suite:
            for test_case in test_group:
                if hasattr(test_case, '__iter__'):
                    for test in test_case:
                        if args.test in str(test):
                            filtered_suite.addTest(test)
                elif args.test in str(test_case):
                    filtered_suite.addTest(test_case)

        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(filtered_suite)
    else:
        result = run_tests(verbosity=verbosity, pattern=args.pattern)

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
