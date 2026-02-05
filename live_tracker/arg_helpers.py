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



def get_args(args):
    return args.exp_folder, args.rois_fname, args.event_schedule, args.debug, args.mode
