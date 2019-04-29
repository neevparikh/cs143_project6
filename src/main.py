import argparse

def main():
    # Creates command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument('source', help='path to source video')
    parser.add_argument('target', help='path to target video')
    args = parser.parse_args()
    # TODO check if video file exists or not

if __name__ == '__main__':
    main()