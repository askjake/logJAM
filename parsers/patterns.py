import re
from datetime import datetime, timezone
from dateutil import parser as dateparser





def convert_to_utc(timestamp_str):
        timestamp_str = timestamp_str.strip()
        current_year = datetime.now().year
        try:
            dt = parser.parse(timestamp_str, fuzzy=True)
        except:
            if re.match(r'^\d{1,2}/\d{1,2}\b', timestamp_str):
                timestamp_str = f"{current_year}/{timestamp_str}"
            elif re.match(r'^[A-Za-z]{3,}\s+\d{1,2}\b', timestamp_str):
                parts = timestamp_str.split()
                if len(parts) >= 2:
                    timestamp_str = f"{parts[0]} {parts[1]} {current_year} {' '.join(parts[2:])}"
            try:
                dt = parser.parse(timestamp_str, fuzzy=True)
            except:
                dt = None

        if not dt:
            return fallback_to_previous_timestamp(timestamp_str)

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt_utc = dt.astimezone(timezone.utc)
        return dt_utc.isoformat()

def fallback_to_previous_timestamp(timestamp_str):
        if last_valid_timestamp:
            time_match = re.search(r'(\d{1,2}):(\d{1,2}):(\d{1,2})', timestamp_str)
            if time_match:
                hours, minutes, seconds = map(int, time_match.groups())
                #print(f'hours: {hours}')
                fallback = last_valid_timestamp.replace(
                    hour=hours, minute=minutes, second=seconds, microsecond=0
                )
                return fallback.astimezone(timezone.utc).isoformat()
            else:
                return last_valid_timestamp.astimezone(timezone.utc).isoformat()
        return None

def parse_log_line(line, last_valid_timestamp=None, rx_id=None):
    
    log_patterns = [
        {
            "name": "standard_log",
            "pattern": r"^\[(.*?)\]<([\d/:\.\s\-]+)><(.*?)>\s(.*?):\s(.*)$",
            "format": lambda m: {
                "category": m.group(1),
                "timestamp": convert_to_utc(m.group(2)),
                "file_line": m.group(3),
                "function": m.group(4),
                "data": m.group(5)
            }
        },
        {
            "name": "qt_ui",
            "pattern": r"^\[(.*?)\]<([\d/:\.\s\-]+)>\s(.*?),\s(.*)$",
            "format": lambda m: {
                "category": m.group(1),
                "timestamp": convert_to_utc(m.group(2)),
                "info": m.group(3),
                "data": m.group(4)
            }
        },
        {
            "name": "qt_ui_chunks",
            "pattern": r"^\[(.*?)\]<([\d/:\.\s\-]+)>\s(.*)$",
            "format": lambda m: {
                "category": m.group(1),
                "timestamp": convert_to_utc(m.group(2)),
                "data": m.group(3)
            }
        },
        {
            "name": "invidi_debug",
            "pattern": r"^([\d/:\s]+):\s(D.*?)\s:\s(.*)$",
            "format": lambda m: {
                "timestamp": convert_to_utc(m.group(1)),
                "component": m.group(2),
                "data": m.group(3)
            }
        },
        {
            "name": "wjap_logs",
            "pattern": r"^([A-Za-z]+\s+\d+\s+[\d:]+)\s(.*?)\[(\d+)\]:\s(.*)$",
            "format": lambda m: {
                "timestamp": convert_to_utc(m.group(1)),
                "service": m.group(2),
                "pid": m.group(3),
                "data": m.group(4)
            }
        },
        {
            "name": "sbsdk_logs",
            "pattern": r"^\[.*?\]<([\d/:\.\s\-]+)>\s(.*?)-(.*?)\s\|\s(.*?)\s\|\s(.*?)\s\|\s(.*?)$",
            "format": lambda m: {
                "timestamp": convert_to_utc(m.group(1)),
                "error_level": m.group(2),
                "code": m.group(3),
                "module": m.group(4),
                "function": m.group(5),
                "data": m.group(6)
            }
        },
        {
            "name": "timestamp_colon_ms",
            "pattern": r"^([\d/]+\s[\d:]+):(\d+)",
            "format": lambda m: {
                "timestamp": convert_to_utc(f"{m.group(1)}.{m.group(2)}"),
                "data": m.string.strip()
            }
        },
        {
            "name": "pipeline_control",
            "pattern": r"^\\[(\\w+)\\]<([\\d/:.\\s\\-]+)\\s([\\-\\d]+)>\\s(\\d+):\\[(\\w+)]<([\\w.]+):(\\d+)>\\s([\\w()\\[\\]:]+)\\((.*?)\\):\\s(.+)$",
            "format": lambda m: {
                "category": m.group(1),
                "timestamp": convert_to_utc(m.group(2)),
                "timezone_offset": m.group(3).strip(),
                "pid": m.group(4),
                "sub_category": m.group(5),
                "file_line": f"<{m.group(6)}:{m.group(7)}>",
                "function": m.group(8).strip(),
                "details": {
                    key.strip(): value.strip()
                    for key, value in (
                        item.split(":", 1) for item in m.group(10).split(",")
                    )
                }
            }
        },
        {
            "name": "generic",
            "pattern": r"^([\d/:\.\s\-]+)\s(.*)$",
            "format": lambda m: {
                "timestamp": convert_to_utc(m.group(1)),
                "data": m.group(2)
            }
        }
    ]

    for pattern in log_patterns:
        match = re.match(pattern["pattern"], line)
        if match:
            parsed_data = pattern["format"](match)
            if parsed_data:
                parsed_data["RxID"] = RxID
            if not parsed_data.get("timestamp"):
                parsed_data["timestamp"] = fallback_to_previous_timestamp(line)
            if "group_by" in pattern:
                group_pipeline_control(parsed_data[pattern["group_by"]], parsed_data)
                return None  # Do not return grouped lines immediately
            return parsed_data

    return None