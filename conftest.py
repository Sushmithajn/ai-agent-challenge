# conftest.py
import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--target",
        action="store",
        default="icici",
        help="Bank target name (e.g. icici, sbi)"
    )

@pytest.fixture
def target(request):
    """Provide the bank target specified on the CLI."""
    return request.config.getoption("--target")
