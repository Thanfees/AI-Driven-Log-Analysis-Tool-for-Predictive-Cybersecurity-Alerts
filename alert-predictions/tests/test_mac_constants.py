"""
Unit tests for Mac constants module.

Tests that regex patterns compile correctly and match expected macOS syslog strings.
"""

import re
import pytest
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mac.common.constants import (
    MAC_SYSLOG_PATTERN,
    KW_PATTERNS,
    KW_PATTERNS_COMPILED,
    LABELING_RULES,
)


class TestMacSyslogPattern:
    """Tests for MAC_SYSLOG_PATTERN regex."""

    def test_basic_kernel_line(self):
        """Test parsing a standard macOS kernel log line."""
        line = "Jul  1 09:00:55 calvisitor-10-105-160-95 kernel[0]: Wake reason: ?"
        match = MAC_SYSLOG_PATTERN.match(line)

        assert match is not None
        assert match.group("ts") == "Jul  1 09:00:55"
        assert match.group("host") == "calvisitor-10-105-160-95"
        assert match.group("proc") == "kernel"
        assert match.group("pid") == "0"
        assert match.group("msg") == "Wake reason: ?"

    def test_dotted_process_name(self):
        """Test parsing macOS-specific dotted process names."""
        line = "Jul  1 09:00:55 myhost com.apple.xpc.launchd[1]: Service exited"
        match = MAC_SYSLOG_PATTERN.match(line)

        assert match is not None
        assert match.group("proc") == "com.apple.xpc.launchd"
        assert match.group("pid") == "1"

    def test_symptomsd_line(self):
        """Test parsing symptomsd log entry."""
        line = "Jul  1 09:00:55 host symptomsd[215]: Hashing of the primary key failed."
        match = MAC_SYSLOG_PATTERN.match(line)

        assert match is not None
        assert match.group("proc") == "symptomsd"
        assert match.group("pid") == "215"

    def test_safari_line(self):
        """Test parsing Safari log entry."""
        line = "Jul  8 08:51:52 host Safari[9852]: tcp_connection_tls_session_error_callback_imp 2538"
        match = MAC_SYSLOG_PATTERN.match(line)

        assert match is not None
        assert match.group("proc") == "Safari"
        assert match.group("pid") == "9852"

    def test_mDNSResponder_line(self):
        """Test parsing mDNSResponder log entry."""
        line = "Jul  8 08:52:03 host mDNSResponder[91]: mDNS_DeregisterInterface: Frequent transitions"
        match = MAC_SYSLOG_PATTERN.match(line)

        assert match is not None
        assert match.group("proc") == "mDNSResponder"

    def test_invalid_line(self):
        """Test that invalid lines don't match."""
        line = "This is not a syslog line"
        match = MAC_SYSLOG_PATTERN.match(line)

        assert match is None


class TestMacKeywordPatterns:
    """Tests for Mac KW_PATTERNS regex patterns."""

    def test_kw_error_matches(self):
        """Test kw_error pattern matches error keywords."""
        pattern = re.compile(KW_PATTERNS["kw_error"], re.IGNORECASE)

        assert pattern.search("This is an error message")
        assert pattern.search("Connection failed")
        assert pattern.search("Permission denied")

    def test_kw_sandbox_matches(self):
        """Test kw_sandbox pattern matches sandbox denial keywords."""
        pattern = re.compile(KW_PATTERNS["kw_sandbox"], re.IGNORECASE)

        assert pattern.search("Sandbox: com.apple.Addres(39276) deny(1) network-outbound")
        assert pattern.search("Sandbox violation detected")

    def test_kw_power_matches(self):
        """Test kw_power pattern matches power/wake keywords."""
        pattern = re.compile(KW_PATTERNS["kw_power"], re.IGNORECASE)

        assert pattern.search("Wake reason: ?")
        assert pattern.search("System Sleep")
        assert pattern.search("thermal pressure state: 2")
        assert pattern.search("powerChange: System Sleep")

    def test_kw_wifi_matches(self):
        """Test kw_wifi pattern matches WiFi event keywords."""
        pattern = re.compile(KW_PATTERNS["kw_wifi"], re.IGNORECASE)

        assert pattern.search("AirPort: Link Down on en0. Reason 8")
        assert pattern.search("Link Down on awdl0")
        assert pattern.search("channel changed to 1")

    def test_kw_memory_matches(self):
        """Test kw_memory pattern matches memory pressure keywords."""
        pattern = re.compile(KW_PATTERNS["kw_memory"], re.IGNORECASE)

        assert pattern.search("Memory pressure state: 2")
        assert pattern.search("jetsam kill detected")

    def test_kw_bluetooth_matches(self):
        """Test kw_bluetooth pattern matches Bluetooth keywords."""
        pattern = re.compile(KW_PATTERNS["kw_bluetooth"], re.IGNORECASE)

        assert pattern.search("Bluetooth -- LE is supported")

    def test_compiled_patterns_exist(self):
        """Test that compiled patterns are available."""
        assert len(KW_PATTERNS_COMPILED) == len(KW_PATTERNS)
        for name in KW_PATTERNS:
            assert name in KW_PATTERNS_COMPILED


class TestMacLabelingRules:
    """Tests for Mac LABELING_RULES patterns."""

    def test_sandbox_deny_rule(self):
        """Test SANDBOX_DENY rule matches."""
        for name, pattern in LABELING_RULES:
            if name == "SANDBOX_DENY":
                assert pattern.search("Sandbox: com.apple.Addres(39276) deny(1) network-outbound")
                break

    def test_kernel_panic_rule(self):
        """Test KERNEL_PANIC rule matches."""
        for name, pattern in LABELING_RULES:
            if name == "KERNEL_PANIC":
                assert pattern.search("Kernel panic - not syncing: Fatal exception")
                break

    def test_wifi_failure_rule(self):
        """Test WIFI_FAILURE rule matches."""
        for name, pattern in LABELING_RULES:
            if name == "WIFI_FAILURE":
                assert pattern.search("AirPort: Link Down on en0. Reason 8 (Disassociated)")
                break

    def test_app_crash_rule(self):
        """Test APP_CRASH rule matches."""
        for name, pattern in LABELING_RULES:
            if name == "APP_CRASH":
                assert pattern.search("Application crashed unexpectedly")
                assert pattern.search("Segmentation fault at 0x0")
                break

    def test_security_violation_rule(self):
        """Test SECURITY_VIOLATION rule matches."""
        for name, pattern in LABELING_RULES:
            if name == "SECURITY_VIOLATION":
                assert pattern.search("access denied for user")
                assert pattern.search("Failed password for root")
                break

    def test_pm_response_slow_rule(self):
        """Test PM_RESPONSE_SLOW rule matches."""
        for name, pattern in LABELING_RULES:
            if name == "PM_RESPONSE_SLOW":
                assert pattern.search("PM response took 1999 ms (54, powerd)")
                break

    def test_all_rules_compile(self):
        """Test that all labeling rules are valid regex."""
        for name, pattern in LABELING_RULES:
            assert hasattr(pattern, 'search'), f"Rule {name} is not a compiled regex"
