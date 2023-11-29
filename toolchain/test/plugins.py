"""
Tests focused on TurnkeyML plugins
"""

import os
import unittest
from unittest.mock import patch
import sys
from turnkeyml.cli.cli import main as turnkeycli
import turnkeyml.common.filesystem as filesystem
import turnkeyml.common.build as build
from helpers import common

# Create a cache directory a directory with test models
cache_dir, corpus_dir = common.create_test_dir("plugins")

class Testing(unittest.TestCase):
    def setUp(self) -> None:
        filesystem.rmdir(cache_dir)

        return super().setUp()

    def test_001_device_naming(self):
        """
        Ensure that the device name is correctly assigned
        """
        test_script = "linear.py"
        testargs = [
            "turnkey",
            "benchmark",
            os.path.join(corpus_dir, test_script),
            "--device",
            "example_family",
            "--build-only",
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", testargs):
            turnkeycli()

        _, build_state = common.get_stats_and_state(test_script, cache_dir)

        # Check if build was successful
        assert build_state.build_status == build.Status.SUCCESSFUL_BUILD

        # Check if default part and config were assigned
        expected_device = "example_family::part1::config1"
        assert build_state.config.device == expected_device, f"Got {build_state.config.device}, expected {expected_device}"

if __name__ == "__main__":
    unittest.main()
