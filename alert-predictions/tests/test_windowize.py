"""
Unit tests for windowize functions.

Tests keyword extraction and window creation.
"""

import pytest
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.linux.common.constants import KW_PATTERNS


class TestKeywordExtraction:
    """Tests for keyword pattern matching in windowize."""
    
    def test_error_keyword_detection(self):
        """Test that error keywords are detected correctly."""
        import re
        pattern = re.compile(KW_PATTERNS["kw_error"], re.IGNORECASE)
        
        test_cases = [
            ("Connection error occurred", True),
            ("Login failed for user", True),
            ("Permission denied for root", True),
            ("Critical system alert", True),
            ("kernel panic detected", True),
            ("Task completed successfully", False),
            ("User logged in", False),
        ]
        
        for text, expected in test_cases:
            result = bool(pattern.search(text))
            assert result == expected, f"Failed for: {text}"
    
    def test_timeout_keyword_detection(self):
        """Test that timeout keywords are detected correctly."""
        import re
        pattern = re.compile(KW_PATTERNS["kw_timeout"], re.IGNORECASE)
        
        test_cases = [
            ("Request timeout after 30s", True),
            ("Connection timed out", True),
            ("No route to host", True),
            ("Connection refused", True),
            ("Connection established", False),
            ("Request completed", False),
        ]
        
        for text, expected in test_cases:
            result = bool(pattern.search(text))
            assert result == expected, f"Failed for: {text}"
    
    def test_auth_keyword_detection(self):
        """Test authentication failure keyword detection."""
        import re
        pattern = re.compile(KW_PATTERNS["kw_auth"], re.IGNORECASE)
        
        test_cases = [
            ("Failed password for admin", True),
            ("invalid user test", True),
            ("authentication failure", True),
            ("possible break-in attempt", True),
            ("Accepted password for user", False),
            ("Session opened for user", False),
        ]
        
        for text, expected in test_cases:
            result = bool(pattern.search(text))
            assert result == expected, f"Failed for: {text}"


class TestWindowCreation:
    """Tests for window creation logic."""
    
    def test_dataframe_creation(self):
        """Test basic DataFrame operations used in windowize."""
        df = pd.DataFrame({
            "timestamp": ["Jan 15 10:00:00", "Jan 15 10:00:30", "Jan 15 10:01:00"],
            "process": ["sshd", "sshd", "kernel"],
            "message": ["msg1", "msg2", "msg3"],
        })
        
        assert len(df) == 3
        assert "timestamp" in df.columns
        assert "process" in df.columns
    
    def test_rolling_features(self):
        """Test rolling feature calculations like those in windowize."""
        values = pd.Series([1, 2, 3, 4, 5])
        
        # Rolling mean (window=3)
        roll_mean = values.rolling(3, min_periods=1).mean()
        assert roll_mean.iloc[0] == 1.0  # Only 1 value
        assert roll_mean.iloc[1] == 1.5  # (1+2)/2
        assert roll_mean.iloc[2] == 2.0  # (1+2+3)/3
        assert roll_mean.iloc[3] == 3.0  # (2+3+4)/3
        
        # Diff
        diff = values.diff().fillna(0)
        assert diff.iloc[0] == 0
        assert diff.iloc[1] == 1
        assert diff.iloc[2] == 1
    
    def test_group_aggregation(self):
        """Test window aggregation logic."""
        df = pd.DataFrame({
            "bucket": ["A", "A", "B", "B", "B"],
            "msg": ["hello", "world", "foo", "bar", "baz"],
        })
        
        agg = df.groupby("bucket").agg(
            count=("msg", "size"),
            text=("msg", lambda s: " || ".join(s)),
        )
        
        assert agg.loc["A", "count"] == 2
        assert agg.loc["B", "count"] == 3
        assert agg.loc["A", "text"] == "hello || world"


class TestTimestampParsing:
    """Tests for timestamp parsing logic."""
    
    def test_syslog_timestamp_with_year(self):
        """Test parsing syslog timestamp with added year."""
        from datetime import datetime
        
        ts = "Jan 15 10:23:45"
        year = 2026
        dt = datetime.strptime(f"{ts} {year}", "%b %d %H:%M:%S %Y")
        
        assert dt.year == 2026
        assert dt.month == 1
        assert dt.day == 15
        assert dt.hour == 10
        assert dt.minute == 23
        assert dt.second == 45
    
    def test_pandas_datetime_floor(self):
        """Test pandas datetime floor operation."""
        import pandas as pd
        
        dt = pd.Timestamp("2026-01-15 10:23:45")
        floored = dt.floor("60s")
        
        assert floored.second == 0
        assert floored.minute == 23
        
        floored_5min = dt.floor("5min")
        assert floored_5min.minute == 20
