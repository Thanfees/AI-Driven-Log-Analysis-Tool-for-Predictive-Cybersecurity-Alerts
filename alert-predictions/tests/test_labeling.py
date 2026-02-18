"""
Unit tests for labeling functions.

Tests that the labeling rules correctly identify anomaly types.
"""

import pytest
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from constants since label_window uses shared rules
from src.linux.common.constants import LABELING_RULES


def label_window(text: str):
    """Apply labeling rules to window text to detect anomalies."""
    if not isinstance(text, str):
        return 0, "NORMAL"
    
    for name, rx in LABELING_RULES:
        if rx.search(text):
            return 1, name
    
    return 0, "NORMAL"


class TestLabelWindow:
    """Tests for the label_window function."""
    
    def test_normal_text(self):
        """Test that normal text returns NORMAL label."""
        text = "Started Session c1 of user root. Finished Daily apt download."
        is_anomaly, label = label_window(text)
        
        assert is_anomaly == 0
        assert label == "NORMAL"
    
    def test_auth_fail(self):
        """Test AUTH_FAIL detection."""
        text = "sshd[1234]: Failed password for invalid user admin from 192.168.1.100"
        is_anomaly, label = label_window(text)
        
        assert is_anomaly == 1
        assert label == "AUTH_FAIL"
    
    def test_segfault(self):
        """Test SEGFAULT detection."""
        text = "app[5678]: segfault at 0 ip 00007f stackpointer error 4 in lib.so"
        is_anomaly, label = label_window(text)
        
        assert is_anomaly == 1
        assert label == "SEGFAULT"
    
    def test_core_dumped(self):
        """Test core dumped detection as SEGFAULT."""
        text = "myapp crashed (core dumped)"
        is_anomaly, label = label_window(text)
        
        assert is_anomaly == 1
        assert label == "SEGFAULT"
    
    def test_oom_killer(self):
        """Test OOM_KILL detection."""
        text = "kernel: Out of memory: Kill process 1234 (python3) score 987"
        is_anomaly, label = label_window(text)
        
        assert is_anomaly == 1
        assert label == "OOM_KILL"
    
    def test_ufw_block(self):
        """Test UFW_BLOCK detection."""
        text = "[UFW BLOCK] IN=eth0 OUT= MAC= SRC=1.2.3.4 DST=10.0.0.1"
        is_anomaly, label = label_window(text)
        
        assert is_anomaly == 1
        assert label == "UFW_BLOCK"
    
    def test_disk_error(self):
        """Test DISK_ERR detection."""
        text = "EXT4-fs error (device sda1): ext4_find_entry:1455: inode #2"
        is_anomaly, label = label_window(text)
        
        assert is_anomaly == 1
        assert label == "DISK_ERR"
    
    def test_kernel_panic(self):
        """Test KERNEL_PANIC detection."""
        text = "kernel: Kernel panic - not syncing: Fatal exception"
        is_anomaly, label = label_window(text)
        
        assert is_anomaly == 1
        assert label == "KERNEL_PANIC"
    
    def test_none_input(self):
        """Test that None input returns NORMAL."""
        is_anomaly, label = label_window(None)
        
        assert is_anomaly == 0
        assert label == "NORMAL"
    
    def test_empty_string(self):
        """Test that empty string returns NORMAL."""
        is_anomaly, label = label_window("")
        
        assert is_anomaly == 0
        assert label == "NORMAL"
    
    def test_first_match_wins(self):
        """Test that first matching rule is used (order matters)."""
        # Text matches both SEGFAULT and AUTH_FAIL - SEGFAULT should win
        text = "segfault: Failed password for admin"
        is_anomaly, label = label_window(text)
        
        assert is_anomaly == 1
        assert label == "SEGFAULT"  # SEGFAULT comes before AUTH_FAIL in rules
