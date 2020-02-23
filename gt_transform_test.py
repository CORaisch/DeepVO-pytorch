# builtins
import sys, math, argparse
# external dependencies
import numpy as np

def mat_to_euler_chiwei(poses):
    '''poses: list of numpy matrices of shape (3,4)'''
    # TODO reimplement from 'data_helper.py' line 203 onward
    return None

def euler_to_mat_chiwei(poses):
    '''poses: list of numpy arrays of shape (1,6)'''
    # TODO reimplement from 'test.py' line 81 onward
    return None

def mat_to_string(mat):
    # NOTE expecting 3x4 mat
    ret  = str(mat[0,0]) + " " + str(mat[0,1]) + " " + str(mat[0,2]) + " " + str(mat[0,3]) + " "
    ret += str(mat[1,0]) + " " + str(mat[1,1]) + " " + str(mat[1,2]) + " " + str(mat[1,3]) + " "
    ret += str(mat[2,0]) + " " + str(mat[2,1]) + " " + str(mat[2,2]) + " " + str(mat[2,3]) + "\n"
    return ret

def main():
    # setup argparser
    argparser = argparse.ArgumentParser(description="Script converts poses from matrix into Euler poses and back. Afterward compare the scripts outpu with its input, to check if transformation pipeline works properly. The script reimplements the Chi Weis pipeline.")
    argparser.add_argument('--poses', '-p', type=str, help="path to GT pose matrices (3x4 format)")
    args = argparser.parse_args()

    # read in poses as numpy matrices
    print("[INFO] parse poses...", end='', flush=True)
    filename = args.poses
    f = open(filename, 'r'); lines = f.readlines(); f.close();
    poses_og = []
    for line in lines:
        l = [float(x.replace('\n', '')) for x in line.split(' ')]
        poses_og.append(np.matrix([[l[0], l[1], l[2], l[3]], [l[4], l[5], l[6], l[7]], [l[8], l[9], l[10], l[11]], [0.0, 0.0, 0.0, 1.0]]))
    print(" done")

    print(poses_og)
    print(poses_og.shape)

    # give status information
    print("[INFO] parsed {} poses".format(len(poses_og)))

    # apply Chi Weis transformation pipeline back and forth
    eulers    = mat_to_euler_chiwei(poses_og)
    poses_new = euler_to_mat_chiwei(eulers)

    # iterate poses and serialize to string
    print("[INFO] serialize new poses...", end='', flush=True)
    poses_str = mat_to_string(poses_new.pop(0).copy())
    for pose in poses_new:
        poses_str += mat_to_string(pose)
    print(" done")

    # write new poses to file
    filename_out = filename.replace('.txt', '_new.txt')
    print("[INFO] writing matrix poses to file at '{}'...".format(filename_out), end='', flush=True)
    with open(filename_out, 'w') as poses_file:
        poses_file.write(poses_str)
    print(" done")

if __name__ == "__main__":
    main()
