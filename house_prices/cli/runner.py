from cli.params import get_parser


def runner():
    parser = get_parser()
    args = parser.parse_args()
    user_args = vars(args)

    args.func(user_args)


if __name__ == '__main__':
    runner()
