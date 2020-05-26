import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2


def process_image(imagename, resultname, params="--edge-thresh 10 --peak-thresh 5"):
    "Process and image and save results in a file"
    print(imagename)

    if imagename[-3:] != 'pgm':
        # create a pgm file
        im = Image.open(imagename)
        im.save('tmp.pgm')
        imagename = 'tmp.pgm'

    cmmd = str("sift "+imagename+" --output="+resultname+" "+params)
    os.system(cmmd)
    print('processed', imagename, 'to', resultname)


def read_features_from_file(filename):
    """read feature properties and return in matrix form"""
    f = np.loadtxt(filename)
    return f[:, :4], f[:,4:]  # feature locations, descriptors


def match(desc1, desc2):
    """For each descriptor in the first image, select its match in the second image.
    Input: desc1 (descriptors for the first image),
    desc2 (same for second image)."""

    desc1 = np.array([d/np.linalg.norm(d) for d in desc1])
    desc2 = np.array([d/np.linalg.norm(d) for d in desc2])

    dist_ratio = 0.6
    desc1_size = desc1.shape

    matchscores = np.zeros((desc1_size[0], 1), 'int')
    desc2t = desc2.T  # precompute matrix transpose
    for i in range(desc1_size[0]):
        dotprods = np.dot(desc1[i, :], desc2t)
        dotprods = 0.999*dotprods
        # inverse cosine and sort, return index for features in second image
        indx = np.argsort(np.arccos(dotprods))

        # check if nearest neighbour has angle less than dist_ratio times 2nd
        if np.arccos(dotprods)[indx[0]] < dist_ratio * np.arccos(dotprods)[indx[1]]:
            matchscores[i] = int(indx[0])

    return matchscores


def match_twosided(desc1, desc2):
    """Two sided symmetric version of match"""

    matches_12 = match(desc1, desc2)
    matches_21 = match(desc2, desc1)

    ndx_12 = matches_12.nonzero()[0]

    # remove matches that are not symmetric
    for n in ndx_12:
        if matches_21[int(matches_12[n])] != n:
            matches_12[n] = 0

    return matches_12


def appendimages(im1, im2):
    """return a new image that appends the two images side-by-side"""

    # select the image with the fewest rows and fill in enough empty rows
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]

    if rows1 < rows2:
        im1 = np.concatenate((im1, np.zeros((rows2-rows1, im1.shape[1]))), axis=0)
    elif rows1 > rows2:
        im2 = np.concatenate((im2, np.zeros((rows1 - rows2, im2.shape[1]))), axis=0)
    # if none of these cases they are equal, no filling needed

    return np.concatenate((im1, im2), axis=1)


def plot_matches(im1, im2, locs1, locs2, matchscores, mask, show_below=True):
    """Show a figure with lines joining the accepted matches
    input: im1, im2 (images as arrays), locs1, locs2 (feature locations),
    matchscores (output from match()), show_below (if images should be shown below matches)"""

    im3 = appendimages(im1, im2)
    if show_below:
        im3 = np.vstack((im3, im3))

    plt.imshow(im3, cmap='bone')

    cols1 = im1.shape[1]
    for i, m in enumerate(matchscores):
        if m > 0:
            mark1 = locs1[i]
            mark1a = mark1.pt
            mark2 = locs2[i]
            mark2a = mark2.pt
            if mask[int(mark1a[0]), int(mark1a[1])] == 1 and mask[int(mark2a[0]), int(mark2a[1])] == 1:
                plt.plot([mark1a[1], mark2a[1]+cols1], [mark1a[0], mark2a[0]], 'r--')  # plot([x1, x2], [y1, y2])
                #plt.plot(mark1a[1], mark1a[0], 'ro')
                #plt.plot(mark2a[1]+cols1, mark2a[0], 'ro')

    plt.axis('off')
    plt.show()





