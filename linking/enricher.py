import re

def enrich_log_data(log_data):
    """Extracts structured information from log lines."""

    data_str = log_data.get("data", "")

    # Extract Type
    type_match = re.search(r'Type:<(\d+):([\w_]+)>', data_str)
    if type_match:
        log_data["msg_type_code"] = int(type_match.group(1))
        log_data["msg_type_name"] = type_match.group(2)

    # Extract Class
    class_match = re.search(r'Class:<(\d+):([\w_]+)>', data_str)
    if class_match:
        log_data["msg_class_code"] = int(class_match.group(1))
        log_data["msg_class_name"] = class_match.group(2)

    # Extract key parameters (stbh, cid, sid, sessMode, etc.)
    field_patterns = {
        "stbh": r"stbh:<([^>]+)>",
        "cid": r"cid:<(\d+)>",
        "sid": r"sid:<(\d+)>",
        "sessMode": r"sessMode:<(\d+)>"
    }

    for field, pattern in field_patterns.items():
        match = re.search(pattern, data_str)
        if match:
            log_data[field] = match.group(1)

    # Label event types for easy classification
    known_events = {
        "ES_STB_MSG_TYPE_CHAN_LIST_EVENT_TRANSITION": "CHAN_LIST_EVENT_TRANSITION",
        "ES_STB_MSG_TYPE_EPG_CUR_PROGRAM_ALL": "EPG_CUR_PROGRAM_ALL",
        "ES_STB_MSG_TYPE_RCA_GET_GET_TUNE_TO_SERVICE": "RCA_GET_TUNE_TO_SERVICE"
    }

    if log_data.get("msg_type_name") in known_events:
        log_data["event_type"] = known_events[log_data["msg_type_name"]]

    return log_data
