import pandas as pd

def process_command_string(cmd_string: pd.DataFrame) -> [list[str], str, int]:
    """Converts a command into separate pieces."""

    time_arr = [int(r) for r in cmd_string.iloc[0].split(":")]

    arduino_command = cmd_string.iloc[3]

    if cmd_string.iloc[1] == "PM" and time_arr[0] != 12:
        time_arr[0] += 12

    video_type = cmd_string.iloc[2]

    j = [3600, 60, 1]
    at_time = sum([time_arr[i] * j[i] for i in range(len(time_arr))])
    
    return at_time, arduino_command, video_type
