#include <gtest/gtest.h>

#include <stdlib.h>

#include "common.h"

#include <tu/graphic.h>

void testBinaryGraphicMatrix(
  TU* tu,           /**< \ref TU environment. */
  TU_CHRMAT* matrix /**< Matrix to be used for testing. */
)
{
  bool isGraphic;
  TU_GRAPH* graph = NULL;
  TU_GRAPH_EDGE* basis = NULL;
  TU_GRAPH_EDGE* cobasis = NULL;
  TU_CHRMAT* transpose = NULL;
  ASSERT_TU_CALL( TUchrmatTranspose(tu, matrix, &transpose) );

  ASSERT_TU_CALL( TUtestBinaryGraphic(tu, transpose, &isGraphic, &graph, &basis, &cobasis, NULL) );

  ASSERT_TRUE( isGraphic );
  ASSERT_TRUE( basis );
  ASSERT_TRUE( cobasis );

  ASSERT_TU_CALL( TUchrmatFree(tu, &transpose) );

  TU_CHRMAT* result = NULL;
  bool isCorrectBasis;
  ASSERT_TU_CALL( TUcomputeGraphBinaryRepresentationMatrix(tu, graph, &result, NULL, matrix->numRows, basis,
    matrix->numColumns, cobasis, &isCorrectBasis) );
  ASSERT_TRUE( isCorrectBasis );
  ASSERT_TRUE( result );

  if (!TUchrmatCheckEqual(matrix, result))
  {
    printf("Input matrix:\n");
    ASSERT_TU_CALL( TUchrmatPrintDense(stdout, matrix, ' ', true) );
  
    printf("Graph:\n");
    ASSERT_TU_CALL( TUgraphPrint(stdout, graph) );

    printf("Representation matrix:\n");
    ASSERT_TU_CALL( TUchrmatPrintDense(stdout, result, ' ', true) );

    printf("Basis:");
    for (int r = 0; r < matrix->numRows; ++r)
      printf(" %d", basis[r]);
    printf("\n");

    printf("Cobasis:");
    for (int c = 0; c < matrix->numColumns; ++c)
      printf(" %d", cobasis[c]);
    printf("\n");
  }

  ASSERT_TRUE( TUchrmatCheckEqual(matrix, result) );
  ASSERT_TRUE( isGraphic );

  ASSERT_TU_CALL( TUgraphFree(tu, &graph) );
  ASSERT_TU_CALL( TUfreeBlockArray(tu, &basis) );
  ASSERT_TU_CALL( TUfreeBlockArray(tu, &cobasis) );
  ASSERT_TU_CALL( TUchrmatFree(tu, &result) );
}

void testBinaryNongraphicMatrix(
  TU* tu,           /**< \ref TU environment. */
  TU_CHRMAT* matrix /**< Matrix to be used for testing. */
)
{
  bool isGraphic;
  TU_CHRMAT* transpose = NULL;
  ASSERT_TU_CALL( TUchrmatTranspose(tu, matrix, &transpose) );
  ASSERT_TU_CALL( TUtestBinaryGraphic(tu, transpose, &isGraphic, NULL, NULL, NULL, NULL) );
  ASSERT_FALSE( isGraphic );
  ASSERT_TU_CALL( TUchrmatFree(tu, &transpose) );
}

void testBinaryMatrix(
  TU* tu,           /**< \ref TU environment. */
  TU_CHRMAT* matrix /**< Matrix to be used for testing. */
)
{
  TU_GRAPH* graph = NULL;
  bool isGraphic;
  TU_CHRMAT* transpose = NULL;
  ASSERT_TU_CALL( TUchrmatTranspose(tu, matrix, &transpose) );
  ASSERT_TU_CALL( TUtestBinaryGraphic(tu, transpose, &isGraphic, &graph, NULL, NULL, NULL) );
  ASSERT_TU_CALL( TUchrmatFree(tu, &transpose) );
  if (graph)
    ASSERT_TU_CALL( TUgraphFree(tu, &graph) );
}

TEST(Graphic, TypingManyChildrenTerminals)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "5 10 "
    "1 0 1 1 1 0 1 0 0 0 "
    "0 0 1 0 0 1 1 0 0 0 "
    "0 1 0 1 0 0 1 1 0 0 "
    "0 0 0 0 1 0 1 1 0 0 "
    "0 0 1 0 0 0 1 0 1 1 "
  ) );
  testBinaryNongraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, TypingInnerParallelWithEdge)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "4 6 "
    "1 0 0 0 0 1 "
    "0 0 0 1 0 1 "
    "0 0 1 0 1 1 "
    "1 0 1 1 0 1 "
  ) );
  testBinaryNongraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, TypingInnerSeriesDoubleChild)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "4 3 "
    "1 0 1 "
    "0 1 1 "
    "1 1 1 "
    "1 1 0 "
  ) );
  testBinaryNongraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, TypingRootSeriesDoubleChild)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "4 3 "
    "1 1 0 "
    "1 1 1 "
    "1 0 1 "
    "0 1 1 "
  ) );
  testBinaryNongraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, TypingAnyRigidDegree3)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "4 4 "
    "1 1 0 1 "
    "0 0 1 0 "
    "1 0 1 1 "
    "0 1 1 1 "
  ) );
  testBinaryNongraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, TypingAnyRigidManyPaths)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "5 7 "
    "0 0 0 1 1 1 1 "
    "0 1 1 1 1 0 0 "
    "0 0 0 1 1 0 1 "
    "0 1 1 1 0 0 1 "
    "0 1 0 0 0 1 1 "
  ) );
  testBinaryNongraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, TypingRootRigidNoPathsDisjointSingleChildren)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "5 4 "
    "0 1 1 0 "
    "0 0 1 1 "
    "1 0 1 0 "
    "1 1 0 0 "
    "1 1 0 1 "
  ) );
  testBinaryNongraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}


TEST(Graphic, TypingRootRigidOnePathSingleChild)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "4 4 "
    "1 1 0 0 "
    "0 1 1 0 "
    "0 1 0 1 "
    "1 0 1 1 "
  ) );
  testBinaryNongraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, TypingRootRigidOnePathTwoSingleChildren)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "5 4 "
    "0 1 1 1 "
    "0 0 1 1 "
    "0 1 1 0 "
    "1 0 1 0 "
    "1 1 0 1 "
  ) );
  testBinaryNongraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}


TEST(Graphic, TypingRootRigidOnePathDoubleChild)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "5 6 "
    "0 1 1 1 0 1 "
    "1 1 1 1 1 1 "
    "0 1 0 0 1 0 "
    "0 0 1 0 1 1 "
    "1 1 1 0 1 0 "
  ) );
  testBinaryNongraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, TypingRootRigidTwoPaths)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "3 4 "
    "1 1 1 0 "
    "1 1 0 1 "
    "0 1 1 1 "
  ) );
  testBinaryNongraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, TypingInnerRigidNoPathNoSingleChild)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "5 7 "
    "1 1 0 0 0 1 0 "
    "1 1 0 1 1 0 0 "
    "1 0 0 1 0 0 0 "
    "0 0 0 0 1 1 0 "
    "1 1 0 1 1 1 0 "
  ) );
  testBinaryNongraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, TypingInnerRigidNoPathOneSingleChild)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "4 4 "
    "1 0 1 1 "
    "0 1 0 1 "
    "0 1 1 0 "
    "1 1 0 0 "
  ) );
  testBinaryNongraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, TypingInnerRigidNoPathTwoSingleChildren)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "5 4 "
    "0 1 1 0 "
    "1 0 1 1 "
    "1 0 0 1 "
    "1 1 0 0 "
    "0 1 0 1 "
  ) );
  testBinaryNongraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}


TEST(Graphic, TypingInnerRigidOnePathOneSingleChild)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "4 4 "
    "1 0 1 1 "
    "0 1 0 1 "
    "0 1 1 1 "
    "1 1 0 1 "
  ) );
  testBinaryNongraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );

  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "5 5 "
    "0 0 0 1 1 "
    "1 0 0 0 1 "
    "1 1 0 1 1 "
    "1 1 0 0 1 "
    "0 1 0 1 0 "
  ) );
  testBinaryNongraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );

  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "4 4 "
    "0 1 1 1 "
    "1 1 0 1 "
    "1 1 0 0 "
    "1 0 1 1 "
  ) );
  testBinaryNongraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );

  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, TypingInnerRigidOnePathTwoSingleChildren)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "5 4 "
    "0 1 1 0 "
    "1 0 1 1 "
    "1 0 0 1 "
    "1 1 0 0 "
    "0 1 0 1 "
  ) );
  testBinaryNongraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );

  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "6 6 "
    "0 1 1 0 1 0 "
    "0 1 1 1 1 1 "
    "1 0 0 1 0 0 "
    "0 0 1 1 0 0 "
    "1 0 1 1 0 0 "
    "1 1 0 1 0 0 "
  ) );
  testBinaryNongraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );

  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "5 4 "
    "0 1 0 1 "
    "1 0 1 1 "
    "1 0 0 1 "
    "0 1 1 0 "
    "1 1 1 1 "
  ) );
  testBinaryNongraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );

  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, TypingInnerRigidOnePathNoChildren)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "3 4 "
    "1 1 0 1 "
    "1 1 1 0 "
    "1 0 1 1 "
  ) );
  testBinaryNongraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, TypingInnerRigidOnePathDoubleChild)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "5 8 "
    "1 1 1 0 0 0 1 1 "
    "0 0 1 1 1 1 0 0 "
    "0 0 0 0 0 1 1 0 "
    "0 1 0 1 0 0 1 0 "
    "0 0 0 0 1 0 1 0 "
  ) );
  testBinaryNongraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, TypingInnerRigidTwoPathsNonadjacentParent)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "5 6 "
    "0 0 0 0 0 1 "
    "1 1 0 1 1 0 "
    "1 1 1 0 0 0 "
    "0 1 0 1 1 0 "
    "0 1 1 0 1 0 "
  ) );
  testBinaryNongraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );

  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "5 6 "
    "0 0 1 0 1 1 "
    "0 1 1 0 0 1 "
    "0 0 0 1 0 0 "
    "0 1 1 1 0 0 "
    "1 0 0 1 1 1 "
  ) );
  testBinaryNongraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );

  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, TypingInnerRigidTwoPathOneSingleChild)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "5 6 "
    "0 0 0 1 0 1 "
    "1 1 1 0 1 1 "
    "1 0 0 1 1 1 "
    "1 1 0 0 0 1 "
    "0 0 0 1 1 0 "
  ) );
  testBinaryNongraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, TypingInnerRigidTwoPathTwoSingleChildren)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "5 6 "
    "0 1 1 1 1 1 "
    "0 0 0 0 1 1 "
    "0 1 1 0 1 1 "
    "0 1 0 0 0 1 "
    "1 0 1 1 0 1 "
  ) );
  testBinaryNongraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, TypingInnerRigidTwoPathsDoubleChild)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "7 6 "
    "1 1 1 1 1 1 "
    "0 0 1 0 0 0 "
    "1 1 1 1 1 0 "
    "0 1 0 1 1 1 "
    "0 0 0 1 0 1 "
    "1 0 0 0 1 1 "
    "0 1 0 0 0 1 "
  ) );
  testBinaryNongraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, RandomMatrix)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  
  srand(0);
  const int numMatrices = 1000;
  const int numRows = 10;
  const int numColumns = 20;
  const double probability = 0.3;

  for (int i = 0; i < numMatrices; ++i)
  {
    TU_CHRMAT* A = NULL;
    TUchrmatCreate(tu, &A, numRows, numColumns, numRows * numColumns);

    A->numNonzeros = 0;
    for (int row = 0; row < numRows; ++row)
    {
      A->rowStarts[row] = A->numNonzeros;
      for (int column = 0; column < numColumns; ++column)
      {
        if ((rand() * 1.0 / RAND_MAX) < probability)
        {
          A->entryColumns[A->numNonzeros] = column;
          A->entryValues[A->numNonzeros] = 1;
          A->numNonzeros++;
        }
      }
    }
    A->rowStarts[numRows] = A->numNonzeros;
    
    /* TUchrmatPrintDense(stdout, A, '0', false); */

    testBinaryMatrix(tu, A);

    ASSERT_TU_CALL( TUchrmatFree(tu, &A) );
  }

  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, UpdateRootParallelNoChildren)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "1 1 "
    "1 "
  ) );
  testBinaryGraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, UpdateRootParallelTwoSingleChildren)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "3 3 "
    "1 1 0 "
    "0 1 1 "
    "1 0 1 "
  ) );
  testBinaryGraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, UpdateRootParallelTwoSingleChildrenSplit)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "3 4 "
    "1 1 1 0 "
    "1 0 0 1 "
    "0 0 1 1 "
  ) );
  testBinaryGraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, UpdateInnerParallelOneSingleChild)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "3 3 "
    "1 0 1 "
    "1 1 0 "
    "0 1 1 "
  ) );
  testBinaryGraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, UpdateRootSeriesNoChildrenParent)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "3 3 "
    "1 1 1 "
    "0 1 1 "
    "0 1 0 "
  ) );
  testBinaryGraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, UpdateRootSeriesNoChildrenHamiltonianPath)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "3 3 "
    "1 0 0 "
    "1 1 1 "
    "1 1 1 "
  ) );
  testBinaryGraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, UpdateRootSeriesNoChildren)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "3 2 "
    "1 0 "
    "1 1 "
    "1 1 "
  ) );
  testBinaryGraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}


TEST(Graphic, UpdateRootSeriesOneSingleChildParent)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "3 3 "
    "1 0 1 "
    "1 1 0 "
    "0 1 1 "
  ) );
  testBinaryGraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, UpdateRootSeriesOneSingleChild)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "4 3 "
    "1 1 0 "
    "1 1 1 "
    "1 0 1 "
    "0 0 1 "
  ) );
  testBinaryGraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, UpdateRootSeriesTwoSingleChildren)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "4 4 "
    "1 1 0 0 "
    "1 1 1 1 "
    "0 1 0 1 "
    "0 0 1 1 "
  ) );
  testBinaryGraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, UpdateRootSeriesTwoSingleChildrenParent)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "4 4 "
    "1 0 1 1 "
    "0 1 0 1 "
    "1 1 1 0 "
    "0 0 1 1 "
  ) );
  testBinaryGraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, UpdateLeafSeries)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "3 3 "
    "1 0 1 "
    "1 1 0 "
    "0 1 1 "
  ) );
  testBinaryGraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, UpdateInnerSeriesOneSingleChild)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "4 4 "
    "1 0 0 1 "
    "1 1 0 1 "
    "1 1 1 0 "
    "0 0 1 1 "
  ) );
  testBinaryGraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, UpdateRootRigidParentOnly)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "4 4 "
    "1 0 1 1 "
    "0 1 1 0 "
    "1 1 1 0 "
    "0 1 0 1 "
  ) );
  testBinaryGraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, UpdateRootRigidParentJoinsPaths)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "4 4 "
    "1 1 1 1 "
    "1 1 0 1 "
    "0 0 1 0 "
    "0 1 1 1 "
  ) );
  testBinaryGraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, UpdateRootRigidNoChildren)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "3 4 "
    "1 0 1 1 "
    "1 1 0 0 "
    "0 1 1 1 "
  ) );
  testBinaryGraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, UpdateRootRigidOneSingleChild)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "4 4 "
    "1 1 0 0 "
    "0 1 0 1 "
    "1 1 1 0 "
    "1 0 1 1 "
  ) );
  testBinaryGraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, UpdateRootRigidTwoSingleChildren)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "5 5 "
    "1 0 0 1 1 "
    "1 1 0 0 1 "
    "0 1 1 0 1 "
    "0 1 0 1 0 "
    "0 0 0 1 1 "
  ) );
  testBinaryGraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, UpdateRootRigidTwoParallelSingleChildren)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "5 5 "
    "1 1 0 1 0 "
    "1 0 0 0 1 "
    "1 0 1 1 0 "
    "0 0 0 1 1 "
    "0 1 1 0 0 "
  ) );
  testBinaryGraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}


TEST(Graphic, UpdateLeafRigid)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "4 4 "
    "1 1 0 1 "
    "1 1 0 0 "
    "1 0 1 0 "
    "0 1 1 1 "
  ) );
  testBinaryGraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, UpdateInnerRigidOnePath)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "5 4 "
    "0 1 1 1 "
    "1 0 1 0 "
    "1 1 0 0 "
    "0 0 1 1 "
    "1 0 1 1 "
  ) );
  testBinaryGraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, UpdateInnerRigidNoPath)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "5 5 "
    "1 1 0 0 1 "
    "0 0 1 1 0 "
    "1 0 1 0 0 "
    "0 1 1 0 1 "
    "0 1 0 1 1 "
  ) );
  testBinaryGraphicMatrix(tu, matrix);
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, UpdateRandomGraph)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );

  srand(1);
  const int numGraphs = 100;
  const int numNodes = 10;
  const int numEdges = 30;

  TU_GRAPH_NODE* nodes = NULL;
  ASSERT_TU_CALL( TUallocBlockArray(tu, &nodes, numNodes) );
  for (int i = 0; i < numGraphs; ++i)
  {
    TU_GRAPH* graph = NULL;
    ASSERT_TU_CALL( TUgraphCreateEmpty(tu, &graph, numNodes, numEdges) );
    for (int v = 0; v < numNodes; ++v)
      ASSERT_TU_CALL( TUgraphAddNode(tu, graph, &nodes[v]) );

    for (int e = 0; e < numEdges; ++e)
    {
      int u = (rand() * 1.0 / RAND_MAX) * numNodes;
      int v = (rand() * 1.0 / RAND_MAX) * numNodes;
      ASSERT_TU_CALL( TUgraphAddEdge(tu, graph, nodes[u], nodes[v], NULL) );
    }

    TU_CHRMAT* A = NULL;
    ASSERT_TU_CALL( TUcomputeGraphBinaryRepresentationMatrix(tu, graph, &A, NULL, 0, NULL, 0, NULL, NULL) );

    ASSERT_TU_CALL( TUgraphFree(tu, &graph) );

    testBinaryGraphicMatrix(tu, A);

    ASSERT_TU_CALL( TUchrmatFree(tu, &A) );
  }
  ASSERT_TU_CALL( TUfreeBlockArray(tu, &nodes) );

  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, RepresentationMatrix)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );

  TU_GRAPH* graph = NULL;
  ASSERT_TU_CALL( TUgraphCreateEmpty(tu, &graph, 0, 0) );

  /**
   * Spanning forest: 
   *
   *        c2                h7 --  
   *        ^                 ^   \
   *        |                 \   /
   * a0 --> b1 <-- d3 --> e4   ---
   *
   *     -->
   *  f5 --> g6
   *     -->
   */
  
  TU_GRAPH_NODE a, b, c, d, e, f, g, h;
  ASSERT_TU_CALL( TUgraphAddNode(tu, graph, &a) );
  ASSERT_TU_CALL( TUgraphAddNode(tu, graph, &b) );
  ASSERT_TU_CALL( TUgraphAddNode(tu, graph, &c) );
  ASSERT_TU_CALL( TUgraphAddNode(tu, graph, &d) );
  ASSERT_TU_CALL( TUgraphAddNode(tu, graph, &e) );
  ASSERT_TU_CALL( TUgraphAddNode(tu, graph, &f) );
  ASSERT_TU_CALL( TUgraphAddNode(tu, graph, &g) );
  ASSERT_TU_CALL( TUgraphAddNode(tu, graph, &h) );
  
  TU_GRAPH_EDGE basis[5];
  ASSERT_TU_CALL( TUgraphAddEdge(tu, graph, a, b, &basis[0]) );
  ASSERT_TU_CALL( TUgraphAddEdge(tu, graph, b, c, &basis[1]) );
  ASSERT_TU_CALL( TUgraphAddEdge(tu, graph, d, b, &basis[2]) );
  ASSERT_TU_CALL( TUgraphAddEdge(tu, graph, d, e, &basis[3]) );
  ASSERT_TU_CALL( TUgraphAddEdge(tu, graph, f, g, &basis[4]) );

  TU_GRAPH_EDGE cobasis[6];
  ASSERT_TU_CALL( TUgraphAddEdge(tu, graph, c, a, &cobasis[0]) );
  ASSERT_TU_CALL( TUgraphAddEdge(tu, graph, a, e, &cobasis[1]) );
  ASSERT_TU_CALL( TUgraphAddEdge(tu, graph, e, c, &cobasis[2]) );
  ASSERT_TU_CALL( TUgraphAddEdge(tu, graph, f, g, &cobasis[3]) );
  ASSERT_TU_CALL( TUgraphAddEdge(tu, graph, g, f, &cobasis[4]) );
  ASSERT_TU_CALL( TUgraphAddEdge(tu, graph, h, h, &cobasis[5]) );

  bool isCorrectBasis = false;
  TU_CHRMAT* matrix = NULL;
  TU_CHRMAT* check = NULL;

  /* Check binary representation matrix. */
  ASSERT_TU_CALL( TUcomputeGraphBinaryRepresentationMatrix(tu, graph, &matrix, NULL, 5, basis, 6, cobasis, &isCorrectBasis) );
  ASSERT_TRUE( isCorrectBasis );
  stringToCharMatrix(tu, &check, "5 6 "
    "1 1 0 0 0 0 "
    "1 0 1 0 0 0 "
    "0 1 1 0 0 0 "
    "0 1 1 0 0 0 "
    "0 0 0 1 1 0 "
  );
  ASSERT_TRUE( TUchrmatCheckEqual(matrix, check) );
  ASSERT_TU_CALL( TUchrmatFree(tu, &check) );
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );

  /* Check ternary representation matrix. */
  bool edgesReversed[] = { false, false, false, false, false, false, false, false, false, false, false };
  ASSERT_TU_CALL( TUcomputeGraphTernaryRepresentationMatrix(tu, graph, &matrix, NULL, edgesReversed, 5, basis, 6, cobasis,
    &isCorrectBasis) );
  ASSERT_TRUE( isCorrectBasis );
  stringToCharMatrix(tu, &check, "5 6 "
    "-1  1  0  0  0  0 "
    "-1  0  1  0  0  0 "
    "0  -1  1  0  0  0 "
    "0   1 -1  0  0  0 "
    "0   0  0  1 -1  0 "
  );
  ASSERT_TRUE( TUchrmatCheckEqual(matrix, check) );
  ASSERT_TU_CALL( TUchrmatFree(tu, &check) );
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );

  ASSERT_TU_CALL( TUgraphFree(tu, &graph) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

void testTernaryGraphicMatrix(
  TU* tu,           /**< \ref TU environment. */
  TU_CHRMAT* matrix /**< Matrix to be used for testing. */
)
{
  bool isGraphic;
  TU_GRAPH* graph = NULL;
  TU_GRAPH_EDGE* basis = NULL;
  TU_GRAPH_EDGE* cobasis = NULL;
  TU_CHRMAT* transpose = NULL;
  bool* edgesReversed = NULL;
  ASSERT_TU_CALL( TUchrmatTranspose(tu, matrix, &transpose) );

  ASSERT_TU_CALL( TUtestTernaryGraphic(tu, transpose, &isGraphic, &graph, &basis, &cobasis, &edgesReversed, NULL) );

  ASSERT_TRUE( isGraphic );
  ASSERT_TRUE( basis );
  ASSERT_TRUE( cobasis );

  ASSERT_TU_CALL( TUchrmatFree(tu, &transpose) );

  TU_CHRMAT* result = NULL;
  bool isCorrectBasis;
  ASSERT_TU_CALL( TUcomputeGraphTernaryRepresentationMatrix(tu, graph, &result, NULL, edgesReversed, matrix->numRows,
    basis, matrix->numColumns, cobasis, &isCorrectBasis) );
  ASSERT_TRUE( isCorrectBasis );
  ASSERT_TRUE( result );

  if (!TUchrmatCheckEqual(matrix, result))
  {
    printf("Input matrix:\n");
    ASSERT_TU_CALL( TUchrmatPrintDense(stdout, matrix, '0', true) );
  
    printf("Graph:\n");
    ASSERT_TU_CALL( TUgraphPrint(stdout, graph) );

    printf("Representation matrix:\n");
    ASSERT_TU_CALL( TUchrmatPrintDense(stdout, result, '0', true) );

    printf("Basis:");
    for (int r = 0; r < matrix->numRows; ++r)
      printf(" %d", basis[r]);
    printf("\n");

    printf("Cobasis:");
    for (int c = 0; c < matrix->numColumns; ++c)
      printf(" %d", cobasis[c]);
    printf("\n");
  }

  ASSERT_TRUE( TUchrmatCheckEqual(matrix, result) );
  ASSERT_TRUE( isGraphic );

  ASSERT_TU_CALL( TUgraphFree(tu, &graph) );
  ASSERT_TU_CALL( TUfreeBlockArray(tu, &basis) );
  ASSERT_TU_CALL( TUfreeBlockArray(tu, &cobasis) );
  ASSERT_TU_CALL( TUfreeBlockArray(tu, &edgesReversed) );
  ASSERT_TU_CALL( TUchrmatFree(tu, &result) );
}

TEST(Graphic, Ternary)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  TU_CHRMAT* matrix = NULL;
  ASSERT_TU_CALL( stringToCharMatrix(tu, &matrix, "6 7 "
    "-1  0  0  0  1 -1  0 "
    " 1  0  0  1 -1  1  0 "
    " 0 -1  0 -1  1 -1  0 "
    " 0  1  0  0  0  0  1 "
    " 0  0  1 -1  1  0  1 "
    " 0  0 -1  1 -1  0  0 "
  ) );
  
  testTernaryGraphicMatrix(tu, matrix);
  
  ASSERT_TU_CALL( TUchrmatFree(tu, &matrix) );
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}
