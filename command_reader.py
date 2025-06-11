import pandas as pd

def process_command_string(cmd_string: pd.DataFrame) -> [list[str], str, int]:
    """Converts a command into separate pieces."""

    at_time = [int(r) for r in cmd_string.iloc[0].split(":")]

    arduino_command = cmd_string.iloc[3]

    if cmd_string.iloc[1] == "PM" and at_time[0] != 12:
        at_time[0] += 12

    video_type = cmd_string.iloc[2]

    return at_time, arduino_command, video_type
