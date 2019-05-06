import argparse

from generate_data import add_base_args, make_get_path, save_pose
from utils import get_pose_normed_estimate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="path to target video", default=None)
    parser.add_argument("target", help="path to target video", default=None)
    parser.add_argument(
        "--regen_source", action='store_true', dest='regen_source')
    parser.add_argument(
        "--regen_target", action='store_true', dest='regen_target')
    parser.add_argument("--regen_norm", action='store_true', dest='regen_norm')
    parser.add_argument("--no-regen_source",
                        action='store_false', dest='regen_source')
    parser.add_argument("--no-regen_target",
                        action='store_false', dest='regen_target')
    parser.add_argument("--no-regen_norm",
                        action='store_false', dest='regen_norm')
    parser.add_argument("--no_norm", action='store_false', dest='no_norm')

    add_base_args(parser)

    args = parser.parse_args()

    data = get_pose_normed_estimate(args.source, args.target,
                                    regen_source=args.regen_source,
                                    regen_target=args.regen_target,
                                    regen_norm=args.regen_norm,
                                    rotated=args.rotated,
                                    height=args.height, width=args.width,
                                    max_frames=args.max_frames)

    source_subsets = data["source_subsets"]
    source_indexes = data["source_indexes"]

    if args.no_norm:
        transformed_all = data['source_poses']
    else:
        transformed_all = data['transformed_all']

    assert len(source_indexes) == len(source_subsets)
    assert len(transformed_all) == len(source_subsets)

    test_path_label = make_get_path(False, True)

    for pose, subsets, index in zip(transformed_all, source_subsets,
                                    source_indexes):
        save_pose(pose, subsets, test_path_label(
            index), args.height, args.width)

        print('test written', index, flush=True)


if __name__ == "__main__":
    main()
