import os

import pytest
from webdriver_manager.chrome import ChromeDriverManager


def test_driver():
    if not os.environ.get("DPB_RUN_CHROMEDRIVER_TESTS"):
        pytest.skip("chromedriver download test is disabled by default")
    driver_path = ChromeDriverManager().install()
    print(f"Installed driver_path: {driver_path}")
    assert driver_path is not None
