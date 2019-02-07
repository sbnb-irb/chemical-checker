import os
import sys
import argparse


from chemicalchecker.util import logged

# Variables


# Parse arguments


def get_parser():
    description = 'Run preprocess script.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-o', '--output_file', type=str,
                        required=False, default='.', help='Output file')
    return parser


@logged
def main():

    args = get_parser().parse_args(sys.argv[1:])

    dataset = os.path.dirname(os.path.abspath(__file__))[-6:]

    main._log.debug(
        "Running preprocess for dataset " + dataset + ". Saving output in " + args.output_file)


if __name__ == '__main__':
    main()
