import argparse

from generate_data import add_base_args, make_get_path, save_pose
from utils import get_pose_normed_estimate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="path to target video", default=None)
    parser.add_argument("target", help="path to target video", default=None)
    parser.add_argument("--regen_source", choices=["true", "false"],
                        default='true')
    parser.add_argument("--regen_target", choices=["true", "false"],
                        default='true')
    parser.add_argument("--regen_norm", choices=["true", "false"],
                        default='true')
    add_base_args(parser)

    args = parser.parse_args()

    data = get_pose_normed_estimate(args.source, args.target,
                                    regen_source=args.regen_source,
                                    regen_target=args.regen_target,
                                    regen_norm=args.regen_norm,
                                    rotate=args.rotated,
                                    max_frames=args.max_frames,
                                    height=args.height, width=args.width)

    source_subsets = data["source_subsets"]
    source_indexes = data["source_indexes"]

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
