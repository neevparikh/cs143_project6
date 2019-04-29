import sys
sys.path.insert(0, 'pytorch-openpose')
import argparse
from pose import get_normalized_target_pose

def main():
	# Creates command line parser
	parser = argparse.ArgumentParser()
	parser.add_argument('source', help='path to source video')
	parser.add_argument('target', help='path to target video')

	args = parser.parse_args()    
	source = args.source
	target = args.target

	try:
		source_file = open(source)
		target_file = open(target)
	except FileNotFoundError:
		print('Video files could not be found')
		sys.exit()
	
	norm_poses = get_normalized_target_pose(source, target)
    

if __name__ == '__main__':
	main()
