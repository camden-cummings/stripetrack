from .config import DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT
"""
Shared arguments between no_gui and gui modes.
"""

def setup_args(parser):
    parser.add_argument(
        "-exp_folder",
        "--exp_folder",
        required=True
    )

    parser.add_argument(
        "-event_schedule",
        "--event_schedule",
        required=True
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
    return args.exp_folder, args.event_schedule, args.debug, args.frame_width, args.frame_height
