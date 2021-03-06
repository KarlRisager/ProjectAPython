"""
@project: LinalgDat2022 Projekt A
@file: ProjektA.py

@description: data and routines to test BasicExtension module.
Do not modify, this file is automatically generated.

@author: François Lauze, University of Copenhagen
@date: Monday April 04. 2022

random state = 9cddaad2-2d1d-44ab-8ab0-593fb34e2b31
"""
from sys import path
path.append("../Core")
from Matrix import Matrix
from Vector import Vector



# Matrix size = (6, 12)
array2D_0001 = [[ -3.18000,   2.80000,  -2.78000,   3.51000,   9.57000,   0.96000,  -3.47000,   1.57000,   7.21000,   6.46000,  -9.69000,  -0.81000],
                [ -0.73000,  -9.67000,   8.87000,   8.27000,   4.91000,   0.21000,   4.20000,  -0.89000,   9.76000,   0.04000,  -5.42000,   2.25000],
                [ -9.97000,   8.01000,   2.53000,  -8.21000,  -6.09000,  -8.44000,  -4.21000,  -4.62000,   9.08000,   8.97000,   1.73000,  -2.07000],
                [  7.59000,  -0.23000,  -4.28000,  -8.54000,  -5.93000,   6.06000,   0.37000,  -3.02000,  -4.49000,  -5.51000,  -8.09000,   2.52000],
                [  7.40000,   8.90000,  -8.66000,  -8.22000,   4.50000,  -3.31000,  -7.16000,  -2.84000,   4.82000,  -7.08000,   9.47000,   6.66000],
                [ -0.47000,  -5.62000,   1.95000,   5.61000,  -7.42000,  -9.37000,  -9.60000,  -8.85000,  -0.78000,  -2.09000,   3.16000,  -4.44000]]
Matrix_0000 = Matrix.fromArray(array2D_0001)
# Vector size = 12
array1D_0003 = [ 2.73000,  5.13000, -2.07000, -4.09000,  4.66000,  9.73000]
Vector_0002 = Vector.fromArray(array1D_0003)
array2D_0005 = [[ -3.18000,   2.80000,  -2.78000,   3.51000,   9.57000,   0.96000,  -3.47000,   1.57000,   7.21000,   6.46000,  -9.69000,  -0.81000,   2.73000],
                [ -0.73000,  -9.67000,   8.87000,   8.27000,   4.91000,   0.21000,   4.20000,  -0.89000,   9.76000,   0.04000,  -5.42000,   2.25000,   5.13000],
                [ -9.97000,   8.01000,   2.53000,  -8.21000,  -6.09000,  -8.44000,  -4.21000,  -4.62000,   9.08000,   8.97000,   1.73000,  -2.07000,  -2.07000],
                [  7.59000,  -0.23000,  -4.28000,  -8.54000,  -5.93000,   6.06000,   0.37000,  -3.02000,  -4.49000,  -5.51000,  -8.09000,   2.52000,  -4.09000],
                [  7.40000,   8.90000,  -8.66000,  -8.22000,   4.50000,  -3.31000,  -7.16000,  -2.84000,   4.82000,  -7.08000,   9.47000,   6.66000,   4.66000],
                [ -0.47000,  -5.62000,   1.95000,   5.61000,  -7.42000,  -9.37000,  -9.60000,  -8.85000,  -0.78000,  -2.09000,   3.16000,  -4.44000,   9.73000]]
Matrix_0004 = Matrix.fromArray(array2D_0005)
# Matrix size = (8, 5)
array2D_0007 = [[  2.86000,  -5.97000,   0.18000,   0.00000,   9.73000],
                [  6.99000,   0.97000,  -0.30000,  -8.92000,   4.43000],
                [ -9.13000,   0.93000,   4.79000,  -2.70000,   3.24000],
                [  6.36000,  -6.96000,  -5.39000,   5.65000,   2.43000],
                [  2.22000,  -0.96000,   5.55000,   3.96000,   5.20000],
                [  2.48000,   2.47000,  -3.94000,   0.05000,   9.13000],
                [  9.87000,  -5.65000,   2.21000,   8.20000,  -7.39000],
                [ -6.60000,  -7.94000,  -8.11000,   4.89000,   1.80000]]
Matrix_0006 = Matrix.fromArray(array2D_0007)
# Vector size = 5
array1D_0009 = [ 7.37000,  6.87000,  8.54000,  4.60000, -4.35000, -1.37000,  1.54000,  1.48000]
Vector_0008 = Vector.fromArray(array1D_0009)
array2D_0011 = [[  2.86000,  -5.97000,   0.18000,   0.00000,   9.73000,   7.37000],
                [  6.99000,   0.97000,  -0.30000,  -8.92000,   4.43000,   6.87000],
                [ -9.13000,   0.93000,   4.79000,  -2.70000,   3.24000,   8.54000],
                [  6.36000,  -6.96000,  -5.39000,   5.65000,   2.43000,   4.60000],
                [  2.22000,  -0.96000,   5.55000,   3.96000,   5.20000,  -4.35000],
                [  2.48000,   2.47000,  -3.94000,   0.05000,   9.13000,  -1.37000],
                [  9.87000,  -5.65000,   2.21000,   8.20000,  -7.39000,   1.54000],
                [ -6.60000,  -7.94000,  -8.11000,   4.89000,   1.80000,   1.48000]]
Matrix_0010 = Matrix.fromArray(array2D_0011)
# Matrix size = (2, 7)
array2D_0013 = [[-0.79000,  2.12000,  5.28000,  0.39000,  6.97000, -7.19000, -1.17000],
                [ 3.99000,  2.08000,  3.72000, -8.94000,  2.36000,  2.83000, -4.52000]]
Matrix_0012 = Matrix.fromArray(array2D_0013)
# Vector size = 7
array1D_0015 = [-2.09000, -0.18000]
Vector_0014 = Vector.fromArray(array1D_0015)
array2D_0017 = [[-0.79000,  2.12000,  5.28000,  0.39000,  6.97000, -7.19000, -1.17000, -2.09000],
                [ 3.99000,  2.08000,  3.72000, -8.94000,  2.36000,  2.83000, -4.52000, -0.18000]]
Matrix_0016 = Matrix.fromArray(array2D_0017)

AugmentRightMatrixList = [Matrix_0000, Matrix_0006, Matrix_0012]
AugmentRightVectorList = [Vector_0002, Vector_0008, Vector_0014]
AugmentRightExpectedList = [Matrix_0004, Matrix_0010, Matrix_0016]
AugmentRightArgs = [AugmentRightMatrixList, AugmentRightVectorList, AugmentRightExpectedList]



# Matrix size = (1, 9)
array2D_0019 = [[  6.29000,   7.64000,  -2.83000,  -0.80000,  -4.03000,   8.12000,  -9.43000,  -5.30000,   5.23000]]
Matrix_0018 = Matrix.fromArray(array2D_0019)
# Vector size = 9
array1D_0021 = [-1.20000,  1.01000, -8.29000, -0.79000,  7.27000, -2.34000, -2.78000, -0.39000, -2.33000]
Vector_0020 = Vector.fromArray(array1D_0021)
array1D_0023 = [-7.94130]
Vector_0022 = Vector.fromArray(array1D_0023)
# Matrix size = (4, 11)
array2D_0025 = [[  3.14000,  -1.71000,  -0.09000,  -5.85000,  -9.02000,  -4.30000,  -1.20000,  -4.90000,   6.30000,   3.48000,   3.60000],
                [ -0.02000,  -3.89000,   1.17000,  -6.46000,  -6.85000,  -4.81000,   8.79000,   8.04000,   4.58000,  -2.83000,  -1.59000],
                [  9.36000,  -4.15000,  -7.41000,  -9.59000,  -9.27000,   6.55000,  -7.10000,   5.80000,  -1.23000,   5.66000,  -3.56000],
                [ -5.97000,  -8.55000,  -3.18000,   3.48000,   5.34000,  -6.76000,   3.91000,   8.41000,   4.59000,   3.32000,  -9.07000]]
Matrix_0024 = Matrix.fromArray(array2D_0025)
# Vector size = 11
array1D_0027 = [  7.22000,   2.17000,   1.41000,  -9.26000,   6.00000,   8.30000,   2.07000,   9.44000,   0.11000,  -2.01000,   6.92000]
Vector_0026 = Vector.fromArray(array1D_0027)
array1D_0029 = [ -46.93560,   61.14280,  139.58190, -103.88210]
Vector_0028 = Vector.fromArray(array1D_0029)
# Matrix size = (4, 14)
array2D_0031 = [[  8.31000,  -2.31000,  -2.43000,   3.19000,  -8.80000,  -6.28000,   4.16000,  -9.86000,  -1.75000,   7.02000,  -2.02000,  -1.12000,  -1.95000,  -3.96000],
                [  7.12000,   7.03000,   9.30000,  -8.05000,   9.35000,   8.11000,  -9.82000,   8.85000,   0.37000,   6.59000,   7.03000,   6.57000,  -3.43000,   5.83000],
                [ -8.28000,  -8.94000,  -0.96000,   1.34000,  -1.40000,  -0.59000,   5.44000,   4.13000,  -5.41000,   9.55000,   1.79000,  -0.32000,  -0.16000,  -3.37000],
                [ -3.56000,  -9.99000,   7.70000,   8.45000,   8.33000,  -1.47000,   6.48000,   3.74000,  -0.29000,   6.47000,   1.32000,   5.00000,   2.81000,  -1.57000]]
Matrix_0030 = Matrix.fromArray(array2D_0031)
# Vector size = 14
array1D_0033 = [ -4.25000,  -8.00000,   8.65000,  -9.88000,   2.95000,   3.73000,  -4.75000,  -2.04000,   7.47000,  -9.78000,  -3.12000,  -4.80000,  -7.30000,  -9.43000]
Vector_0032 = Vector.fromArray(array1D_0033)
array1D_0035 = [-136.87610,   14.80900,  -60.34250,  -40.41940]
Vector_0034 = Vector.fromArray(array1D_0035)

MatVecProductMatrixList = [Matrix_0018, Matrix_0024, Matrix_0030]
MatVecProductVectorList = [Vector_0020, Vector_0026, Vector_0032]
MatVecProductExpectedList = [Vector_0022, Vector_0028, Vector_0034]
MatVecProductArgs = [MatVecProductMatrixList, MatVecProductVectorList, MatVecProductExpectedList]



# Matrix size = (6, 10)
array2D_0037 = [[  1.94000,   4.43000,   4.92000,   9.67000,   6.07000,  -6.60000,   7.35000,  -5.10000,  -6.35000,  -2.72000],
                [ -3.67000,  -4.93000,  -8.21000,  -4.64000,   7.27000,  -8.62000,   0.95000,   7.28000,  -0.85000,  -2.58000],
                [  7.02000,   0.30000,   3.64000,   3.71000,   0.81000,   7.78000,   4.29000,  -5.21000,  -7.79000,   3.68000],
                [  4.45000,   4.93000,   1.38000,   7.16000,  -8.50000,   0.41000,   1.14000,  -1.71000,  -5.80000,  -6.20000],
                [ -1.90000,   1.42000,   2.99000,   2.90000,  -5.86000,  -3.41000,  -6.77000,   8.89000,  -3.01000,   4.26000],
                [  1.70000,   4.93000,  -2.34000,  -5.02000,  -9.72000,  -8.24000,  -8.55000,  -1.52000,  -9.40000,   4.19000]]
Matrix_0036 = Matrix.fromArray(array2D_0037)
# Matrix size = (10, 15)
array2D_0039 = [[  7.05000,   5.87000,  -8.28000,  -3.07000,   5.29000,   5.45000,   5.49000,  -8.44000,  -3.01000,  -9.84000,  -4.18000,  -0.07000,   0.35000,   7.21000,   4.39000],
                [ -8.38000,   4.24000,   1.30000,   9.84000,  -7.38000,   0.26000,   2.04000,  -5.06000,   5.92000,   3.48000,  -0.04000,   6.46000,  -6.09000,   7.14000,  -9.47000],
                [  9.48000,  -7.07000,   6.33000,   6.32000,  -7.76000,  -7.47000,   9.23000,  -2.49000,   0.64000,   2.29000,  -5.70000,  -3.82000,   4.38000,   2.81000,   7.66000],
                [ -3.49000,  -6.15000,  -4.21000,  -1.98000,  -1.55000,  -0.56000,  -6.48000,  -1.58000,  -7.55000,   5.26000,  -7.92000,   1.68000,  -3.24000,   0.07000,   8.82000],
                [  2.28000,   1.56000,  -5.08000,   7.67000,  -3.12000,  -8.56000,  -8.85000,   8.73000,  -9.73000,  -1.17000,   4.52000,   2.68000,  -9.14000,  -2.80000,   7.90000],
                [ -8.63000,  -1.09000,  -8.17000,  -8.12000,  -7.63000,  -4.63000,   0.63000,   9.19000,   0.55000,  -0.46000,  -2.22000,   3.28000,  -9.02000,  -7.79000,   6.90000],
                [  9.15000,  -3.35000,  -6.77000,   3.03000,  -0.86000,  -0.16000,   8.80000,   8.27000,  -9.40000,   8.52000,   5.31000,   6.15000,  -9.79000,  -6.62000,  -9.33000],
                [ -1.33000,   5.66000,  -8.39000,   4.83000,   7.73000,   7.61000,  -3.07000,  -2.70000,   3.74000,  -2.01000,   5.32000,   8.22000,  -6.15000,  -4.86000,   4.72000],
                [  3.14000,   4.99000,   4.04000,   8.52000,  -9.62000,   3.10000,   7.49000,   9.64000,   0.53000,   0.76000,   6.89000,   5.33000,   9.72000,  -3.41000,   1.10000],
                [  5.08000,  -0.63000,   8.21000,  -4.91000,  -0.82000,   3.78000,   9.18000,  -3.75000,   5.36000,  -3.29000,  -3.85000,   6.45000,   3.25000,  -4.79000,  -4.12000]]
Matrix_0038 = Matrix.fromArray(array2D_0039)
array2D_0041 = [[ 100.52340, -130.88210,  -51.74060,  106.62280,  -26.60550, -121.79760,  -47.63380,  -50.44120, -218.27330,  131.38770,  -92.21140,  -27.55620, -143.18220,  105.34900,    3.52800],
                [  28.00360,  100.27780,  -67.08900,   89.27770,  196.70890,   63.18460, -189.72720,   57.62550,  -59.82150,  -27.48820,  198.93100,   27.69870,  -51.60190,  -64.67510,  -40.08990],
                [  43.65830,  -98.33910, -104.58110, -156.50650,  -33.00420,  -84.45950,   75.62310,  -36.47300,  -92.85270,  -15.71380, -165.87560,  -12.76360, -146.74260,    6.22760,   60.31490],
                [ -81.76790,  -59.00640,  -79.71980,  -62.90880,   35.42330,   27.46630,   -8.76470, -166.30710,   -7.87060,   53.08650, -141.83740,  -61.09090,  -48.79850,  142.50960,  -17.29400],
                [ -52.58270,    5.76200,   75.98420,   -8.43260,   96.07840,  107.51270,  -19.32220, -210.65430,  167.41120,  -37.68880,  -76.83660,   18.74080,   74.89860,   39.48730,   41.13270],
                [ -69.48390,   42.61100,  182.41360, -103.10960,  174.34220,  128.69120,    8.63700, -359.04980,  242.67720, -106.89250, -114.23280, -108.99620,  155.05450,  207.90810, -190.07460]]
Matrix_0040 = Matrix.fromArray(array2D_0041)
# Matrix size = (3, 6)
array2D_0043 = [[-6.82000, -7.40000,  6.57000,  8.62000, -6.97000,  2.72000],
                [ 6.58000,  7.87000, -4.36000,  1.29000, -6.09000,  9.69000],
                [-1.92000, -3.52000, -7.91000, -0.03000,  9.09000,  5.00000]]
Matrix_0042 = Matrix.fromArray(array2D_0043)
# Matrix size = (6, 6)
array2D_0045 = [[  4.71000,  -5.64000,   0.16000,   8.50000,   5.53000,  -7.36000],
                [ -2.41000,   7.42000,   1.49000,   6.34000,  -1.27000,  -3.49000],
                [ -8.19000,   2.98000,   1.29000,  -8.68000,  -2.73000,   2.39000],
                [ -0.40000,   7.91000,  -3.70000,  -1.88000,   2.60000,   3.31000],
                [ -5.91000,  -9.61000,   5.84000,   6.13000,   7.75000,  -1.32000],
                [  7.36000,  -9.31000,   9.36000,   7.32000,   9.48000,  -8.79000]]
Matrix_0044 = Matrix.fromArray(array2D_0045)
array2D_0047 = [[ -10.33260,  112.97810,  -50.78150, -200.93490,  -52.07260,  105.54730],
                [ 154.52780,  -13.19370,   57.51450,  174.84450,   86.31300, -159.18190],
                [  47.31300, -173.00360,   84.24070,  122.40010,  133.21660,  -48.53700]]
Matrix_0046 = Matrix.fromArray(array2D_0047)
# Matrix size = (11, 8)
array2D_0049 = [[ -8.17000,   4.08000,  -9.99000,   8.92000,  -1.64000,  -2.49000,   4.66000,   5.29000],
                [  0.83000,   8.64000,   8.40000,  -1.47000,   6.67000,   0.99000,   8.47000,   1.01000],
                [ -8.58000,   7.19000,   6.68000,   9.72000,  -0.51000,  -1.52000,  -9.42000,   0.06000],
                [ -9.60000,  -3.05000,   9.63000,  -6.36000,   3.10000,   4.76000,   7.83000,  -6.70000],
                [  9.84000,  -9.56000,   2.14000,  -5.62000,  -4.48000,  -8.31000,  -6.71000,  -0.93000],
                [  4.57000,  -9.68000,   1.78000,  -9.35000,  -5.25000,   1.97000,   3.82000,  -0.80000],
                [  6.68000,   4.47000,  -7.16000,   7.41000,  -5.46000,  -6.17000,   5.36000,   2.35000],
                [ -8.72000,  -9.40000,  -0.75000,  -5.59000,   2.20000,   6.51000,   4.28000,   2.60000],
                [ -1.17000,   6.71000,  -7.79000,  -7.54000,   0.54000,   2.99000,  -3.13000,  -2.25000],
                [  3.84000,  -6.82000,   8.18000,  -1.56000,  -9.77000,  -6.91000,  -3.37000,   8.97000],
                [  0.80000,  -6.37000,   5.74000,  -1.39000,  -0.74000,  -6.16000,  -3.61000,   6.40000]]
Matrix_0048 = Matrix.fromArray(array2D_0049)
# Matrix size = (8, 13)
array2D_0051 = [[ -3.63000,   6.69000,  -6.35000,  -4.06000,   9.60000,   4.32000,   4.73000,  -6.26000,   6.15000,  -9.64000,   1.80000,  -5.74000,   3.67000],
                [  1.73000,   9.02000,  -0.65000,  -3.40000,   0.57000,   6.35000,   3.25000,   2.55000,  -4.72000,   1.42000,  -0.71000,   7.27000,  -2.53000],
                [  1.31000,  -1.94000,   3.60000,   9.50000,   1.70000,   6.71000,   3.95000,   8.71000,  -0.97000,   2.45000,   1.66000,   4.32000,   9.30000],
                [ -8.85000,   5.52000,   1.79000,   9.64000,   6.61000,   7.24000,  -5.63000,   7.47000,  -1.18000,   3.45000,  -9.42000,  -5.47000,  -3.24000],
                [  5.32000,   0.06000,  -3.07000,  -1.42000,   5.62000,  -6.46000,   7.11000,  10.00000,  -8.37000,  -7.75000,   9.84000,  -2.67000,   0.25000],
                [  7.97000,  -2.63000,  -7.14000,   5.50000,   9.68000,  -0.17000,   6.25000,  -0.32000,   7.91000,   7.22000,  -1.67000,  -3.79000,   2.16000],
                [ -4.46000,   9.10000,   2.81000,   9.52000,   0.77000,  -8.23000,  -2.17000,   8.05000,   4.40000,  -3.91000,  -5.35000,   1.61000,   8.80000],
                [  0.62000,   3.61000,   3.22000,   8.28000,   5.27000,  -4.45000,   8.81000,  -0.12000,  -5.31000,   1.32000,  -3.61000,   0.90000,   8.68000]]
Matrix_0050 = Matrix.fromArray(array2D_0051)
array2D_0053 = [[-101.38730,  118.71650,   82.17210,   87.18020,  -35.98170,  -62.71310, -105.79440,   62.44270,  -83.89340,   74.34530, -174.21980,   10.68770,  -80.97730],
                [  42.17250,  137.59470,   16.22960,  117.85420,   76.36930,  -13.28830,  117.59140,  213.46480,  -58.18180,  -56.55310,   38.16990,   95.36220,  151.17840],
                [  -6.46430,  -33.38960,   77.39780,   70.73220,  -27.18150,  204.59910,  -37.70960,  122.38470, -154.17410,  172.71050,  -53.32210,   69.22000, -104.83400],
                [ 113.82620, -110.79120,   43.15100,  120.36420,  -85.34820,  -97.72970,   -5.70230,  181.99890,   35.25330,   60.74740,   65.63380,   89.58020,  104.45450],
                [ -60.43190,  -98.40740,   -7.38850, -152.21620,  -60.18160,   45.18870,  -21.85610, -205.36310,   57.37480, -122.84850,   90.04270,  -54.18020,   12.22040],
                [  21.98150,  -85.42760,  -22.84610,  -10.83380,  -32.13310,  -91.77430,    9.47520, -129.91630,  163.68270,  -46.77820,   33.63170,  -25.78990,  117.72500],
                [-192.14410,  212.96120,   25.60900,    5.39700,   29.58490,   44.59710,  -92.18720,  -47.18880,   26.18600,  -70.11330, -153.41950,  -28.60310,  -24.51610],
                [ 109.99250, -141.18190,   15.93930,  101.30520,  -34.91650, -204.97760,   26.66150,   36.38620,   36.16980,   66.23920,   20.88310,  -12.26400,   77.75980],
                [ 111.64740,  -18.24810,  -77.51930, -197.50380,  -52.77930,  -37.53070,   37.44960, -120.22330,   -5.10660,    2.37960,   76.41360,   43.25030, -109.76820],
                [ -87.67350,  -41.00540,  105.44960,   88.32680,  -40.54820,   68.98270,   10.77840, -105.52850,   14.37070,   18.80140,  -58.92110,   27.16990,  143.31130],
                [ -27.06660,  -64.50440,   73.95450,   45.33600,  -28.22010,   -1.48380,   34.03650,  -16.89670,  -61.34100,  -23.66700,   27.80000,    6.76830,   87.23110]]
Matrix_0052 = Matrix.fromArray(array2D_0053)

MatrixProductMatrixAList = [Matrix_0036, Matrix_0042, Matrix_0048]
MatrixProductMatrixBList = [Matrix_0038, Matrix_0044, Matrix_0050]
MatrixProductExpectedList = [Matrix_0040, Matrix_0046, Matrix_0052]
MatrixProductArgs = [MatrixProductMatrixAList, MatrixProductMatrixBList, MatrixProductExpectedList]



# Matrix size = (4, 6)
array2D_0055 = [[  7.68000,   9.94000,   5.26000,   7.05000,  -7.40000,   6.11000],
                [  6.83000,   1.81000,   9.33000,  -9.33000,  -7.73000,   1.05000],
                [ -9.66000,  -2.24000,   3.02000,   7.86000,  -5.68000,  -9.45000],
                [ -9.26000,  -8.03000,   2.71000,   1.50000,  -7.66000,  -6.64000]]
Matrix_0054 = Matrix.fromArray(array2D_0055)
array2D_0057 = [[  7.68000,   6.83000,  -9.66000,  -9.26000],
                [  9.94000,   1.81000,  -2.24000,  -8.03000],
                [  5.26000,   9.33000,   3.02000,   2.71000],
                [  7.05000,  -9.33000,   7.86000,   1.50000],
                [ -7.40000,  -7.73000,  -5.68000,  -7.66000],
                [  6.11000,   1.05000,  -9.45000,  -6.64000]]
Matrix_0056 = Matrix.fromArray(array2D_0057)
# Matrix size = (3, 12)
array2D_0059 = [[ -6.26000,   0.93000,  -8.74000,   4.75000,   5.88000,   9.50000,  -9.90000,   5.35000,   3.68000,   8.89000,  -9.73000,  -1.18000],
                [ -4.69000,  -4.70000,  -8.45000,  -2.06000,   9.57000,   3.99000,   7.33000,   1.16000,   9.09000,  -6.35000,   2.49000,  -0.06000],
                [ -0.54000,  -8.84000,  -2.24000,  -1.37000,   8.85000,   5.23000,   6.28000,  -2.38000,   0.52000,  -6.58000,   6.87000,   2.52000]]
Matrix_0058 = Matrix.fromArray(array2D_0059)
array2D_0061 = [[ -6.26000,  -4.69000,  -0.54000],
                [  0.93000,  -4.70000,  -8.84000],
                [ -8.74000,  -8.45000,  -2.24000],
                [  4.75000,  -2.06000,  -1.37000],
                [  5.88000,   9.57000,   8.85000],
                [  9.50000,   3.99000,   5.23000],
                [ -9.90000,   7.33000,   6.28000],
                [  5.35000,   1.16000,  -2.38000],
                [  3.68000,   9.09000,   0.52000],
                [  8.89000,  -6.35000,  -6.58000],
                [ -9.73000,   2.49000,   6.87000],
                [ -1.18000,  -0.06000,   2.52000]]
Matrix_0060 = Matrix.fromArray(array2D_0061)
# Matrix size = (2, 13)
array2D_0063 = [[  6.06000,   4.98000,   4.27000,  -3.79000,   4.53000,  -4.97000,  -9.91000,   5.98000,   9.68000,  -1.14000,   3.18000,   7.06000,  -3.49000],
                [ -5.65000,  -0.97000,  -8.34000,  -0.68000,  -4.24000,   2.37000,  -4.42000,   8.48000,  -3.34000,  -7.07000,  -2.48000,   7.25000,  -3.91000]]
Matrix_0062 = Matrix.fromArray(array2D_0063)
array2D_0065 = [[  6.06000,  -5.65000],
                [  4.98000,  -0.97000],
                [  4.27000,  -8.34000],
                [ -3.79000,  -0.68000],
                [  4.53000,  -4.24000],
                [ -4.97000,   2.37000],
                [ -9.91000,  -4.42000],
                [  5.98000,   8.48000],
                [  9.68000,  -3.34000],
                [ -1.14000,  -7.07000],
                [  3.18000,  -2.48000],
                [  7.06000,   7.25000],
                [ -3.49000,  -3.91000]]
Matrix_0064 = Matrix.fromArray(array2D_0065)

TransposeMatrixList = [Matrix_0054, Matrix_0058, Matrix_0062]
TransposeExpectedList = [Matrix_0056, Matrix_0060, Matrix_0064]
MatrixTransposeArgs = [TransposeMatrixList, TransposeExpectedList]



# Vector size = [13]
array1D_0067 = [  9.10000,  -7.18000,   0.65000,  -5.84000,   5.89000,  -1.84000,   6.39000,   5.23000,  -3.25000,  -1.57000,  -7.07000,  -9.11000,   2.32000]
Vector_0066 = Vector.fromArray(array1D_0067)
float_0068 = 20.65284
# Vector size = [7]
array1D_0070 = [ 7.55000, -7.56000, -2.24000,  8.16000,  1.37000, -8.72000,  1.69000]
Vector_0069 = Vector.fromArray(array1D_0070)
float_0071 = 16.32577
# Vector size = [1]
array1D_0073 = [5.15000]
Vector_0072 = Vector.fromArray(array1D_0073)
float_0074 = 5.15000

VectorNormVectorList = [Vector_0066, Vector_0069, Vector_0072]
VectorNormExpected = [float_0068, float_0071, float_0074]
VectorNormArgs = [VectorNormVectorList, VectorNormExpected]



