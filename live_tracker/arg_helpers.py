from .config import DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT, DEFAULT_EVENT_SCHEDULES
import os
"""
Shared arguments between no_gui and gui modes.
"""

def check_event_schedules(exp_folder):
    for default in DEFAULT_EVENT_SCHEDULES:
        path = f'{exp_folder}\{default}'
        if os.path.isfile(path):
            return default
        
def setup_args(parser):
    parser.add_argument(
        "-e",
        "--exp_folder",
        required=True
    )

    parser.add_argument(
        "-s",
        "--event_schedule",
    )

    parser.add_argument(
        "-d",
        "--debug",
        action='store_true'
    )

    parser.add_argument(
        "-frame_width",
        "--frame_width",
        default=DEFAULT_FRAME_WIDTH
    )
    
    parser.add_argument(
        "-frame_height",
        "--frame_height",
        default=DEFAULT_FRAME_HEIGHT
    )


def get_args(args):
    if args.event_schedule:
        event_schedule = args.event_schedule
    else:
        event_schedule = check_event_schedules(args.exp_folder)
    
    return args.exp_folder, event_schedule, args.debug, args.frame_width, args.frame_height
