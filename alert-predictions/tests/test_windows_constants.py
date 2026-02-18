"""
Unit tests for Windows constants module.

Tests that regex patterns compile correctly and match expected Windows CBS/CSI strings.
"""

import re
import pytest
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.windows.common.constants import (
    WINDOWS_LOG_PATTERN,
    KW_PATTERNS,
    KW_PATTERNS_COMPILED,
    LABELING_RULES,
)


class TestWindowsLogPattern:
    """Tests for WINDOWS_LOG_PATTERN regex."""

    def test_basic_cbs_line(self):
        """Test parsing a standard CBS log line."""
        line = "2016-09-28 04:30:30, Info                  CBS    Starting TrustedInstaller initialization."
        match = WINDOWS_LOG_PATTERN.match(line)

        assert match is not None
        assert match.group("ts").strip() == "2016-09-28 04:30:30"
        assert match.group("level") == "Info"
        assert match.group("proc") == "CBS"

    def test_csi_line(self):
        """Test parsing a CSI log line."""
        line = "2016-09-28 04:30:31, Info                  CSI    00000001@2016/9/27:20:30:31.455 WcpInitialize called"
        match = WINDOWS_LOG_PATTERN.match(line)

        assert match is not None
        assert match.group("proc") == "CSI"

    def test_error_level(self):
        """Test parsing a line with Error level."""
        line = "2016-10-01 12:00:00, Error                 CBS    Package installation failed"
        match = WINDOWS_LOG_PATTERN.match(line)

        assert match is not None
        assert match.group("level") == "Error"

    def test_warning_in_message(self):
        """Test parsing a Warning line."""
        line = "2016-09-28 04:30:31, Info                  CBS    Warning: Unrecognized packageExtended attribute."
        match = WINDOWS_LOG_PATTERN.match(line)

        assert match is not None
        assert "Warning" in match.group("msg")

    def test_invalid_line(self):
        """Test that non-CBS lines don't match."""
        line = "This is not a Windows CBS log line"
        match = WINDOWS_LOG_PATTERN.match(line)

        assert match is None


class TestWindowsKeywordPatterns:
    """Tests for Windows KW_PATTERNS regex patterns."""

    def test_kw_error_matches(self):
        """Test kw_error pattern matches Windows error keywords."""
        pattern = re.compile(KW_PATTERNS["kw_error"], re.IGNORECASE)

        assert pattern.search("This is an error message")
        assert pattern.search("Package installation failed")
        assert pattern.search("[HRESULT = 0x80004005 - E_FAIL]")
        assert pattern.search("HRESULT = 0x800f080d")

    def test_kw_update_matches(self):
        """Test kw_update pattern matches Windows Update keywords."""
        pattern = re.compile(KW_PATTERNS["kw_update"], re.IGNORECASE)

        assert pattern.search("Session initialized by client WindowsUpdateAgent")
        assert pattern.search("Windows Update started scanning")

    def test_kw_manifest_matches(self):
        """Test kw_manifest pattern matches manifest error keywords."""
        pattern = re.compile(KW_PATTERNS["kw_manifest"], re.IGNORECASE)

        assert pattern.search("CBS_E_MANIFEST_INVALID_ITEM")
        assert pattern.search("INVALID_ITEM detected")

    def test_kw_service_matches(self):
        """Test kw_service pattern matches service keywords."""
        pattern = re.compile(KW_PATTERNS["kw_service"], re.IGNORECASE)

        assert pattern.search("TrustedInstaller service starts successfully")
        assert pattern.search("service failed to start")

    def test_kw_corruption_matches(self):
        """Test kw_corruption pattern matches corruption keywords."""
        pattern = re.compile(KW_PATTERNS["kw_corruption"], re.IGNORECASE)

        assert pattern.search("Component store corrupted")
        assert pattern.search("SFC found issues")
        assert pattern.search("DISM repair failed")

    def test_compiled_patterns_exist(self):
        """Test that compiled patterns are available."""
        assert len(KW_PATTERNS_COMPILED) == len(KW_PATTERNS)
        for name in KW_PATTERNS:
            assert name in KW_PATTERNS_COMPILED


class TestWindowsLabelingRules:
    """Tests for Windows LABELING_RULES patterns."""

    def test_cbs_fail_rule(self):
        """Test CBS_FAIL rule matches."""
        for name, pattern in LABELING_RULES:
            if name == "CBS_FAIL":
                assert pattern.search("CBS failed to process package")
                assert pattern.search("CBS error during initialization")
                break

    def test_manifest_error_rule(self):
        """Test MANIFEST_ERROR rule matches."""
        for name, pattern in LABELING_RULES:
            if name == "MANIFEST_ERROR":
                assert pattern.search("CBS_E_MANIFEST_INVALID_ITEM")
                assert pattern.search("manifest invalid attribute detected")
                break

    def test_hresult_error_rule(self):
        """Test HRESULT_ERROR rule matches."""
        for name, pattern in LABELING_RULES:
            if name == "HRESULT_ERROR":
                assert pattern.search("[HRESULT = 0x80004005 - E_FAIL]")
                assert pattern.search("HRESULT = 0x800f080d")
                break

    def test_update_fail_rule(self):
        """Test UPDATE_FAIL rule matches."""
        for name, pattern in LABELING_RULES:
            if name == "UPDATE_FAIL":
                assert pattern.search("Windows Update failed to install")
                assert pattern.search("update failure detected")
                break

    def test_all_rules_compile(self):
        """Test that all labeling rules are valid regex."""
        for name, pattern in LABELING_RULES:
            assert hasattr(pattern, 'search'), f"Rule {name} is not a compiled regex"
