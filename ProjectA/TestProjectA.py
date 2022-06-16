"""
@Project: LinalgDat 2022
@File: TestProjectA.py

@Description: Routines for testing implementations for the first LinalgDat
programming project.

A few routines are there to check validity of some vectors/matrix objects.
The tests have the same structure:
1) Check that teh implemented function runs
2) When applicable, test that the result has the expected size/dimension
3) Check that the result has the expected value
Only when all these checks are successful does the test method return True.

This an adaptation of my previous F# and C# code
"""

from sys import path
path.append('../Core')
from Vector import Vector
from Matrix import Matrix
import BasicExtensions
from data_projectA import (
    AugmentRightArgs,
    MatVecProductArgs,
    MatrixProductArgs,
    MatrixTransposeArgs,
    VectorNormArgs
)
# for test and debugging purposes
import numpy as np



__author__ = "François Lauze, University of Copenhagen"
__date__ = "3/29/22"
__version__ = "0.0.1"

Tolerance = 1e-3


class ProjectAException(Exception):
    """Dummy exception class for the project."""
    pass


def toNumpy(x) -> np.ndarray:
    if type(x) is Matrix:
        m, n = x.M_Rows, x.N_Cols
        A = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                A[i, j] = x[i, j]
    elif type(x) is Vector:
        m = len(x)
        A = np.zeros(m)
        for i in range(m):
            A[i] = x[i]
    else:
        raise ProjectAException("Convert only from Vector of Matrix type.")
    return A


def l1_distance(v: Vector, w: Vector) -> float:
    """
    l1-distance between two vectors of the same length.

    Internal use, as the lengths are actually not compared.
    """
    l1 = 0.0
    for i in range(len(v)):
        l1 += abs(v[i] - w[i])
    return l1


def compareVectorDimensions(v: Vector, w: Vector) -> bool:
    """Return True if v and w have same length, False otherwise."""
    return len(v) == len(w)


def compareVectors(v: Vector, w: Vector) -> bool:
    """
    Return True if they have the same length and L1-distance is less than Tolerance.

    Somewhat obfuscated version...
    """
    if compareVectorDimensions(v, w):
        return l1_distance(v, w) < Tolerance
    else:
        return False


def compareMatrixDimensions(A: Matrix, B: Matrix) -> bool:
    """Compare matrix dimensions."""
    return (A.N_Cols == B.N_Cols) and (A.M_Rows == B.M_Rows)


def compareMatrices(A: Matrix, B: Matrix) -> bool:
    """Compare matrices up to tolerance via l1-distance."""
    if compareMatrixDimensions(A, B):
        l1_norm = 0.0
        for i in range(A.N_Cols):
            l1_norm += l1_distance(A.Row(i), B.Row(i))
        return l1_norm < Tolerance
    else:
        raise ProjectAException("Matrix dimensions mismatch.")


def outMessage(taskName: str, subTaskName: str, status: bool) -> str:
    """Display a message followed by [PASSED] or [FAILED]."""
    passed = '[PASSED]'
    failed = '[FAILED]'
    s = f'{taskName} {subTaskName}'
    return f'{s:50} {passed if status else failed}'


# noinspection PyBroadException
def TestMatrixAugmentation(A: Matrix, v: Vector, expected: Matrix) -> tuple:
    """Test that the AugmentRight routine returns a correct result."""
    taskName = 'AugmentRight(Matrix, Vector)'
    status = True
    resultStr = f'\nTests for the {taskName} function'
    resultStr += '\n==========================================================='

    try:
        Av = BasicExtensions.AugmentRight(A, v)
        try:
            if not compareMatrices(Av, expected):
                resultStr += f'\n{outMessage(taskName, "Values", False)}'
                status = False
            else:
                resultStr += f'\n{outMessage(taskName, "Dims", True)}'
                resultStr += f'\n{outMessage(taskName, "Values", True)}'
        except:
            resultStr += f'\n{outMessage(taskName, "Dims", False)}'
            status = False
    except Exception as e:
        resultStr += f'\n{outMessage(taskName, "Run", False)}'
        status = False

    if status:
        resultStr += f'\n{outMessage(taskName, "All", True)}'
    resultStr += f'\nEnd of test for the {taskName} function.'
    resultStr += '\n-----------------------------------------------------------\n'
    return taskName, status, resultStr


# noinspection PyBroadException
def TestMatrixVectorProduct(A: Matrix, v: Vector, expected: Vector) -> tuple:
    """Test that MatVecProduct returns a correct result."""
    taskName = 'MatVecProduct(Matrix, Vector)'
    status = True
    resultStr = f'\nTests for the {taskName} function'
    resultStr += '\n==========================================================='
    npA = toNumpy(A)
    npv = toNumpy(v)
    try:
        Av = BasicExtensions.MatVecProduct(A, v)
        npAv = toNumpy(Av)
        npe = toNumpy(expected)
        try:
            if not compareVectors(Av, expected):
                resultStr += f'\n{outMessage(taskName, "Values", False)}'
                status = False
                # TODO: dump the expected result and some context.
            else:
                resultStr += f'\n{outMessage(taskName, "Dims", True)}'
                resultStr += f'\n{outMessage(taskName, "Values", True)}'
        except:
            resultStr += f'\n{outMessage(taskName, "Dims", False)}'
            status = False
    except Exception as e:
        resultStr += f'\n{outMessage(taskName, "Run", False)}'
        status = False

    if status:
        resultStr += f'\n{outMessage(taskName, "All", True)}'
    resultStr += f'\nEnd of test for the {taskName} function.'
    resultStr += '\n-----------------------------------------------------------\n'
    return taskName, status, resultStr


# noinspection PyBroadException
def TestMatrixMatrixProduct(A: Matrix, B: Matrix, expected: Matrix) -> tuple:
    """Test that the MatrixProduct routine returns a correct result."""
    taskName = 'MatrixProduct(Matrix, Matrix)'
    status = True
    resultStr = f'\nTests for the {taskName} function'
    resultStr += '\n==========================================================='

    try:
        Av = BasicExtensions.MatrixProduct(A, B)
        try:
            if not compareMatrices(Av, expected):
                resultStr += f'\n{outMessage(taskName, "Values", False)}'
                status = False
                # TODO: dump the expected result, just in case?
            else:
                resultStr += f'\n{outMessage(taskName, "Dims", True)}'
                resultStr += f'\n{outMessage(taskName, "Values", True)}'
        except:
            resultStr += f'\n{outMessage(taskName, "Dims", False)}'
            status = False
    except:
        resultStr += f'\n{outMessage(taskName, "Run", False)}'
        status = False

    if status:
        resultStr += f'\n{outMessage(taskName, "All", True)}'
    resultStr += f'\nEnd of test for the {taskName} function.'
    resultStr += '\n-----------------------------------------------------------\n'
    return taskName, status, resultStr


# noinspection PyBroadException
def TestTranspose(A: Matrix, expected: Matrix) -> tuple:
    """Test that the Transpose routine returns a correct result."""
    taskName = 'Transpose(Matrix)'
    status = True
    resultStr = f'\nTests for the {taskName} function'
    resultStr += '\n==========================================================='

    try:
        Av = BasicExtensions.Transpose(A)
        try:
            if not compareMatrices(Av, expected):
                resultStr += f'\n{outMessage(taskName, "Values", False)}'
                status = False
                # TODO: dumps the expected result just in case?
            else:
                resultStr += f'\n{outMessage(taskName, "Dims", True)}'
                resultStr += f'\n{outMessage(taskName, "Values", True)}'
        except:
            resultStr += f'\n{outMessage(taskName, "Dims", False)}'
            status = False
    except:
        resultStr += f'\n{outMessage(taskName, "Run", False)}'
        status = False

    if status:
        resultStr += f'\n{outMessage(taskName, "All", True)}'
    resultStr += f'\nEnd of test for the {taskName} function.'
    resultStr += '\n-----------------------------------------------------------\n'
    return taskName, status, resultStr


# noinspection PyBroadException
def TestVectorNorm(v: Vector, expected: float) -> tuple:
    """Test that the VectorNorm routine returns a correct result."""
    taskName = 'VectorNorm(Vector)'
    status = True
    resultStr = f'\nTests for the {taskName} function'
    resultStr += '\n==========================================================='

    try:
        nv = BasicExtensions.VectorNorm(v)
        if abs(nv - expected) > Tolerance:
            resultStr += f'\n{outMessage(taskName, "Value", False)}'
            status = False
            # TODO: dumps the expected result just in case?
        else:
            resultStr += f'\n{outMessage(taskName, "Value", True)}'
    except:
        resultStr += f'\n{outMessage(taskName, "Run", False)}'
        status = False

    if status:
        resultStr += f'\n{outMessage(taskName, "All", True)}'
    resultStr += f'\nEnd of test for the {taskName} function.'
    resultStr += '\n-----------------------------------------------------------\n'
    return taskName, status, resultStr


def runTestMatrixAugmentation(matrices: list, vectors: list, expected_matrices: list) -> tuple:
    """Run matrix augmentation tests and collect output."""
    n = len(matrices)
    passed = 0
    taskName = ''
    for M, v, Mv in zip(matrices, vectors, expected_matrices):
        taskName, status, resultString = TestMatrixAugmentation(M, v, Mv)
        passed += int(status)
        print(resultString)
    return taskName, passed, n


def runTestMatrixVectorProduct(matrices: list, vectors: list, expected_vectors: list) -> tuple:
    """Run matrix vector product tests and collect output."""
    n = len(matrices)
    passed = 0
    taskName = ''
    for M, v, Mv in zip(matrices, vectors, expected_vectors):
        taskName, status, resultString = TestMatrixVectorProduct(M, v, Mv)
        passed += int(status)
        print(resultString)
    return taskName, passed, n


def runTestMatrixMatrixProduct(matricesA: list, matricesB: list, expected_matrices: list) -> tuple:
    """Run matrix matrix product tests and collect output."""
    n = len(matricesA)
    passed = 0
    taskName = ''
    for A, B, AB in zip(matricesA, matricesB, expected_matrices):
        taskName, status, resultString = TestMatrixMatrixProduct(A, B, AB)
        passed += int(status)
        print(resultString)
    return taskName, passed, n


def runTestTranspose(matrices: list, expected_matrices: list) -> tuple:
    """Run matrix transpose tests and collect output."""
    n = len(matrices)
    passed = 0
    taskName = ''
    for A, AT in zip(matrices, expected_matrices):
        taskName, status, resultString = TestTranspose(A, AT)
        passed += int(status)
        print(resultString)
    return taskName, passed, n


def runTestVectorNorm(vectors: list, expected_norms: list) -> tuple:
    """Run vector norm tests and collect output."""
    n = len(vectors)
    passed = 0
    taskName = ''
    for v, nv in zip(vectors, expected_norms):
        taskName, status, resultString = TestVectorNorm(v, nv)
        passed += int(status)
        print(resultString)
    return taskName, passed, n


def printSummaryInfo(taskName: str, passed: int, total: int):
    """Print a line with execution summary for a given task."""
    str1 = f'Test of {taskName} passed/total'
    print(f'{str1:<52} [{passed}/{total}]')


def runALL():
    results = [
        runTestMatrixAugmentation(*AugmentRightArgs),
        runTestMatrixVectorProduct(*MatVecProductArgs),
        runTestMatrixMatrixProduct(*MatrixProductArgs),
        runTestTranspose(*MatrixTransposeArgs),
        runTestVectorNorm(*VectorNormArgs)
    ]

    print('=' * 80)
    for result in results:
        printSummaryInfo(*result)
    print('-' * 80)


if __name__ == '__main__':
    runALL()