import numpy as np
import matplotlib.pyplot as plt
import cv2


# interpolation func for attach bins to image again
def interpolate(subBin, LU, RU, LB, RB, subX, subY):
    subImage = np.zeros(subBin.shape)
    num = subX * subY
    for i in range(subX):
        inverseI = subX - i
        for j in range(subY):
            inverseJ = subY - j
            val = subBin[i, j].astype(int)
            subImage[i, j] = np.floor(
                (inverseI * (inverseJ * LU[val] + j * RU[val]) + i * (inverseJ * LB[val] + j * RB[val])) / num)
    return subImage


def clahe(img, clipLimit, nrBins=128, nrX=0, nrY=0):
    # img - Input image
    # clipLimit - Normalized clipLimit. Higher value gives more contrast
    # nrBins - Number of gray level bins for histogram("dynamic range") that means we can change the numbers
    # nrX - Number of contextial regions in x direction
    # nrY - Number of contextial regions in y direction

    h, w = img.shape
    if clipLimit == 1:
        return
    nrBins = max(nrBins, 128)
    if nrX == 0:
        # Taking dimensions of each contextial region to be a square of 32X32
        xsz = 32
        ysz = 32
        nrX = int(np.ceil(h / xsz))
        # Excess number of pixels to get an integer value of nrX and nrY
        excX = int(xsz * (nrX - h / xsz))
        nrY = int(np.ceil(w / ysz))
        excY = int(ysz * (nrY - w / ysz))
        # Pad that number of pixels to the image
        if excX != 0:
            img = np.append(img, np.zeros((excX, img.shape[1])).astype(int), axis=0)
        if excY != 0:
            img = np.append(img, np.zeros((img.shape[0], excY)).astype(int), axis=1)
    else:
        xsz = round(h / nrX)
        ysz = round(w / nrY)
    # number of region pixels
    nrPixels = xsz * ysz
    xsz2 = round(xsz / 2)
    ysz2 = round(ysz / 2)
    claheimg = np.zeros(img.shape)

    if clipLimit > 0:
        clipLimit = max(1, clipLimit * xsz * ysz / nrBins)
    else:
        clipLimit = 50

    # make Look-Up Table
    minVal = np.min(img)
    maxVal = np.max(img)

    # maxVal1 = maxVal + np.maximum(np.array([0]),minVal) - minVal
    # minVal1 = np.maximum(np.array([0]),minVal)

    binSz = np.floor(1 + (maxVal - minVal) / nrBins)
    LUT = np.floor((np.arange(minVal, maxVal + 1) - minVal) / binSz)

    # continue of clahe
    bins = LUT[img]
    # print(bins.shape)
    # makeHistogram
    hist = np.zeros((nrX, nrY, nrBins))
    print(nrX, nrY, hist.shape)
    for i in range(nrX):
        for j in range(nrY):
            bin_ = bins[i * xsz:(i + 1) * xsz, j * ysz:(j + 1) * ysz].astype(int)
            for i1 in range(xsz):
                for j1 in range(ysz):
                    hist[i, j, bin_[i1, j1]] += 1

    # clipHistogram
    if clipLimit > 0:
        for i in range(nrX):
            for j in range(nrY):
                nrExcess = 0
                for nr in range(nrBins):
                    excess = hist[i, j, nr] - clipLimit
                    if excess > 0:
                        nrExcess += excess

                binIncr = nrExcess / nrBins
                upper = clipLimit - binIncr
                for nr in range(nrBins):
                    if hist[i, j, nr] > clipLimit:
                        hist[i, j, nr] = clipLimit
                    else:
                        if hist[i, j, nr] > upper:
                            nrExcess += upper - hist[i, j, nr]
                            hist[i, j, nr] = clipLimit
                        else:
                            nrExcess -= binIncr
                            hist[i, j, nr] += binIncr

                if nrExcess > 0:
                    stepSz = max(1, np.floor(1 + nrExcess / nrBins))
                    for nr in range(nrBins):
                        nrExcess -= stepSz
                        hist[i, j, nr] += stepSz
                        if nrExcess < 1:
                            break

    # mapping Histogram here
    map_ = np.zeros((nrX, nrY, nrBins))
    # print(map_.shape)
    scale = (maxVal - minVal) / nrPixels
    for i in range(nrX):
        for j in range(nrY):
            sum_ = 0
            for nr in range(nrBins):
                sum_ += hist[i, j, nr]
                map_[i, j, nr] = np.floor(min(minVal + sum_ * scale, maxVal))

    # interpolation func and pre processes here for mapping and using
    xI = 0
    for i in range(nrX + 1):
        if i == 0:
            subX = int(xsz / 2)
            xU = 0
            xB = 0
        elif i == nrX:
            subX = int(xsz / 2)
            xU = nrX - 1
            xB = nrX - 1
        else:
            subX = xsz
            xU = i - 1
            xB = i

        yI = 0
        for j in range(nrY + 1):
            if j == 0:
                subY = int(ysz / 2)
                yL = 0
                yR = 0
            elif j == nrY:
                subY = int(ysz / 2)
                yL = nrY - 1
                yR = nrY - 1
            else:
                subY = ysz
                yL = j - 1
                yR = j
            UL = map_[xU, yL, :]
            UR = map_[xU, yR, :]
            BL = map_[xB, yL, :]
            BR = map_[xB, yR, :]
            subBin = bins[xI:xI + subX, yI:yI + subY]
            # print(subBin.shape)
            subImage = interpolate(subBin, UL, UR, BL, BR, subX, subY)
            claheimg[xI:xI + subX, yI:yI + subY] = subImage
            yI += subY
        xI += subX
    # check more excess pixels
    if excX == 0 and excY != 0:
        return claheimg[:, :-excY]
    elif excX != 0 and excY == 0:
        return claheimg[:-excX, :]
    elif excX != 0 and excY != 0:
        return claheimg[:-excX, :-excY]
    else:
        return claheimg


def main(fileName, resultName):
    # Load the input image
    original_img = cv2.imread(fileName)

    # Convert the input image into grayscale
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    clahe_img = clahe(gray_img, 8, 0, 0)
    # turn clahe imame to float 32
    asd = np.array(clahe_img, dtype=np.float32)
    # plot hist for org image
    print(clahe_img.shape)
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    plt.hist(gray_img.ravel(), 256, [0, 256])
    plt.title(fileName)
    plt.show()
    # plot hist for output image
    hist = cv2.calcHist([asd], [0], None, [256], [0, 256])
    plt.hist(asd.ravel(), 256, [0, 256])
    plt.title(resultName)
    plt.show()
    # plot org image
    plt.imshow(gray_img, cmap='gray')
    plt.title(fileName)
    plt.show()
    # plot output image
    plt.imshow(clahe_img, cmap='gray')
    plt.title(resultName)
    plt.imsave(resultName, clahe_img, cmap='gray')
    plt.show()


# mainnnnnnnnnnnnnnnnnnnnnnn
# runnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn

main('statue.jpg', 'output Statue.png')
main('Tem.jpg', 'output Tem.png')
