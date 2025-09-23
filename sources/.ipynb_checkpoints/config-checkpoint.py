
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import numpy as np

FORMAT_TIMESTAMP = "%Y%m%d-%H%M%S"
DIR_OUTPUT = "output/"
DIR_MEERKAT_CATS = "meerkat-catalogues/"
DIR_MULTI_CATS = "multi-wavelength-catalogues/"
FILENAME_CAT_DESCRIPTION = "description-catalogues.txt"

SEPERATOR_1 = "="*79
SEPERATOR_2 = "-"*79
SEPERATOR_3 = str("- "*40)[:79]
SEPERATOR_space = " "*79
SEPERATOR_12 = SEPERATOR_1 + "\n" + SEPERATOR_2
SEPERATOR_21 = SEPERATOR_2 + "\n" + SEPERATOR_1
SEPERATOR_13 = SEPERATOR_1 + "\n" + SEPERATOR_3
SEPERATOR_31 = SEPERATOR_3 + "\n" + SEPERATOR_1
SEPERATOR_23 = SEPERATOR_2 + "\n" + SEPERATOR_3
SEPERATOR_32 = SEPERATOR_3 + "\n" + SEPERATOR_2


def rgb(r, g, b):
    return [r/255, g/255, b/255]

colorA1 = rgb(255,48,48)
# colorB1 = rgb(173, 216, 230)
# colorA1 = rgb(200,10, 120)
colorA2 = rgb(10, 90, 200)
colorA3 = rgb(10, 90, 120)

# colorB1 = rgb(195, 20, 130)
colorB1 = rgb(30,144,255)
colorB2 = (0.49, 0.09, 0.42) #7d176a
colorB3 = (0.92, 0.33, 0.68) #d63791
y = rgb(255,211,67)
colorC1 = rgb(0,0,0)
# colorC1 = rgb(20, 20, 20)
colorC2 = rgb(50, 50, 50)
colorC3 = rgb(90, 90, 90)
colorC4 = rgb(150, 150, 150)
colorC5 = rgb(200, 200, 200)

colorD1 = rgb(245, 180, 50)
colorD1 = rgb(110, 210, 255)
colorD2 = rgb(50, 150, 200)
colorD3 = rgb(120, 90, 10)

silver = rgb(192,192,192)
N = 256

mapA1 = np.ones((N, 4))
mapA1[:, 0] = np.linspace(10/N, 50/N, N) # R = 255
mapA1[:, 1] = np.linspace(200/N, 50/N, N) # G = 232
mapA1[:, 2] = np.linspace(120/N, 50/N, N)  # B = 11
colorA1_cmp = ListedColormap(mapA1)

mapB1 = np.ones((N, 4))
mapB1[:, 0] = np.linspace(195/N, 50/N, N) # R = 255
mapB1[:, 1] = np.linspace(20/N, 50/N, N) # G = 232
mapB1[:, 2] = np.linspace(130/N, 50/N, N)  # B = 11
colorB1_cmp = ListedColormap(mapB1)

survey_colors = [rgb(100, 100, 255)]
survey_colors += [rgb(100, 255, 100)]
survey_colors += [rgb(255, 100, 100)]
survey_colors += [rgb(255, 100, 255)]
survey_colors += [rgb(100, 255, 255)]
survey_colors += [rgb(255, 255, 100)]
