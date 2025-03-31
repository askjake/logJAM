#!/usr/bin/env python3
"""
var.py

Parser for the new log file format used for VAR logs.

Example log lines:
    2025-03-27 19:50:32.666  4807 31770 I VAR - c : aspect ratio scale factor: 1
    2025-03-27 19:50:32.666  4807 31770 I VAR - c : frame rate: 29
    2025-03-27 19:50:32.752  4807 31770 I VAR - c : video resolution: 1440x1080
    2025-03-27 19:50:33.178  4807 31699 I VAR - OMEGAk_0.z2_0: Parsed Header:

Each line is expected to contain:
  - A timestamp (with milliseconds)
  - A process ID and a thread ID
  - The literal "I VAR -" token
  - A channel identifier (e.g. "c" or "OMEGAk_0.z2_0")
  - A colon followed by a key (e.g. "aspect ratio scale factor", "frame rate", etc.)
  - Optionally, another colon and a value

This parser extracts these fields and returns a list of dictionaries.
Each dictionary contains the following keys:
    "timestamp", "pid", "tid", "channel", "key", "value"
"""

import re
from datetime import datetime

# Regular expression to match the new log line format.
LOG_LINE_RE = re.compile(
    r'^(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)\s+'
    r'(?P<pid>\d+)\s+(?P<tid>\d+)\s+I\s+VAR\s+-\s+'
    r'(?P<channel>[^:]+):\s+(?P<rest>.*)$'
)

def parse_var(content: str) -> list:
    """
    Parse the content of a VAR log file.

    Args:
        content (str): The raw text content of the log file.

    Returns:
        list: A list of dictionaries, each containing the parsed fields:
              "timestamp", "pid", "tid", "channel", "key", "value".
    """
    parsed_logs = []

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue

        match = LOG_LINE_RE.match(line)
        if not match:
            # Optionally log or print that this line didn't match the expected format.
            continue

        log_dict = match.groupdict()

        # Convert the timestamp into ISO format.
        try:
            dt = datetime.strptime(log_dict["timestamp"], "%Y-%m-%d %H:%M:%S.%f")
            log_dict["timestamp"] = dt.isoformat()
        except Exception:
            # Leave the timestamp as-is if conversion fails.
            pass

        # Process the "rest" field to extract key and value.
        rest = log_dict.pop("rest").strip()
        if ':' in rest:
            key_part, value_part = rest.split(':', 1)
            log_dict["key"] = key_part.strip()
            log_dict["value"] = value_part.strip()
        else:
            log_dict["key"] = rest
            log_dict["value"] = ""

        # Try converting the value to a number if possible.
        try:
            if log_dict["value"]:
                if '.' in log_dict["value"]:
                    log_dict["value"] = float(log_dict["value"])
                else:
                    log_dict["value"] = int(log_dict["value"])
        except ValueError:
            # If conversion fails, keep the value as a string.
            pass

        parsed_logs.append(log_dict)

    return parsed_logs

# For testing the parser independently:
if __name__ == "__main__":
    sample_log = """
2025-03-27 19:50:32.666  4807 31770 I VAR - c : aspect ratio scale factor: 1
2025-03-27 19:50:32.666  4807 31770 I VAR - c : frame rate: 29
2025-03-27 19:50:32.752  4807 31770 I VAR - c : video resolution: 1440x1080
2025-03-27 19:50:33.178  4807 31699 I VAR - OMEGAk_0.z2_0: Parsed Header:
    """
    results = parse_var(sample_log)
    import pprint
    pprint.pprint(results)
