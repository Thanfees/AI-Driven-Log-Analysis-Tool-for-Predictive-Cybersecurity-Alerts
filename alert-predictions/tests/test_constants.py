"""
Unit tests for constants module.

Tests that regex patterns compile correctly and match expected strings.
"""

import re
import pytest
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.linux.common.constants import (
    SYSLOG_PATTERN,
    KW_PATTERNS,
    KW_PATTERNS_COMPILED,
    LABELING_RULES,
)


class TestSyslogPattern:
    """Tests for SYSLOG_PATTERN regex."""
    
    def test_basic_syslog_line(self):
        """Test parsing a standard syslog line."""
        line = "Jan 15 10:23:45 myhost sshd[1234]: Accepted password for user"
        match = SYSLOG_PATTERN.match(line)
        
        assert match is not None
        assert match.group("ts") == "Jan 15 10:23:45"
        assert match.group("host") == "myhost"
        assert match.group("proc") == "sshd"
        assert match.group("pid") == "1234"
        assert match.group("msg") == "Accepted password for user"
    
    def test_syslog_without_pid(self):
        """Test parsing syslog line without PID."""
        line = "Feb  2 08:00:00 server kernel: Something happened"
        match = SYSLOG_PATTERN.match(line)
        
        assert match is not None
        assert match.group("proc") == "kernel"
        assert match.group("pid") is None
        assert match.group("msg") == "Something happened"
    
    def test_invalid_line(self):
        """Test that invalid lines don't match."""
        line = "This is not a syslog line"
        match = SYSLOG_PATTERN.match(line)
        
        assert match is None


class TestKeywordPatterns:
    """Tests for KW_PATTERNS regex patterns."""
    
    def test_kw_error_matches(self):
        """Test kw_error pattern matches error keywords."""
        pattern = re.compile(KW_PATTERNS["kw_error"], re.IGNORECASE)
        
        # Should match
        assert pattern.search("This is an error message")
        assert pattern.search("Connection failed")
        assert pattern.search("Permission denied")
        assert pattern.search("critical failure occurred")
        
        # Should not match
        assert not pattern.search("Everything is fine")
        assert not pattern.search("Normal operation")
    
    def test_kw_auth_matches(self):
        """Test kw_auth pattern matches authentication failures."""
        pattern = re.compile(KW_PATTERNS["kw_auth"], re.IGNORECASE)
        
        assert pattern.search("Failed password for user admin")
        assert pattern.search("invalid user test from 192.168.1.1")
        assert pattern.search("authentication failure; logname=")
    
    def test_kw_segfault_matches(self):
        """Test kw_segfault pattern matches segfaults."""
        pattern = re.compile(KW_PATTERNS["kw_segfault"], re.IGNORECASE)
        
        assert pattern.search("segfault at 0 ip 00007f")
        assert pattern.search("core dumped")
    
    def test_kw_kernel_matches(self):
        """Test kw_kernel pattern matches kernel issues."""
        pattern = re.compile(KW_PATTERNS["kw_kernel"], re.IGNORECASE)
        
        assert pattern.search("Out of memory: Kill process")
        assert pattern.search("invoked oom-killer")
        assert pattern.search("Kernel panic - not syncing")
    
    def test_compiled_patterns_exist(self):
        """Test that compiled patterns are available."""
        assert len(KW_PATTERNS_COMPILED) == len(KW_PATTERNS)
        for name in KW_PATTERNS:
            assert name in KW_PATTERNS_COMPILED


class TestLabelingRules:
    """Tests for LABELING_RULES patterns."""
    
    def test_auth_fail_rule(self):
        """Test AUTH_FAIL rule matches."""
        for name, pattern in LABELING_RULES:
            if name == "AUTH_FAIL":
                assert pattern.search("Failed password for user admin from 192.168.1.1")
                assert pattern.search("invalid user test from 10.0.0.1 port 22")
                break
    
    def test_segfault_rule(self):
        """Test SEGFAULT rule matches."""
        for name, pattern in LABELING_RULES:
            if name == "SEGFAULT":
                assert pattern.search("app[1234]: segfault at 0")
                assert pattern.search("core dumped")
                break
    
    def test_oom_kill_rule(self):
        """Test OOM_KILL rule matches."""
        for name, pattern in LABELING_RULES:
            if name == "OOM_KILL":
                assert pattern.search("kernel: Out of memory: Kill process 1234")
                assert pattern.search("invoked oom-killer")
                break
    
    def test_ufw_block_rule(self):
        """Test UFW_BLOCK rule matches."""
        for name, pattern in LABELING_RULES:
            if name == "UFW_BLOCK":
                assert pattern.search("[UFW BLOCK] IN=eth0 OUT= MAC=")
                break
    
    def test_all_rules_compile(self):
        """Test that all labeling rules are valid regex."""
        for name, pattern in LABELING_RULES:
            assert hasattr(pattern, 'search'), f"Rule {name} is not a compiled regex"
