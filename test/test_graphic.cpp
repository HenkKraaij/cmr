#include <gtest/gtest.h>

#include "common.h"
#include <tu/graphic.h>
#include <tu/tdec.h>

void testGraphicMatrix(
  TU* tu,             /**< \ref TU environment. */
  TU_CHRMAT* matrix,  /**< Matrix to be used for testing. */
  int mergeLeafBonds  /**< Leaf bonds of the t-decomposition are merged (1: at the end; 2: after each column). */
)
{
  TU_GRAPH* graph = NULL;
  ASSERT_TU_CALL( TUgraphCreateEmpty(tu, &graph, 0, 0) );
  TU_GRAPH_EDGE* basis = NULL;
  ASSERT_TU_CALL( TUallocBlockArray(tu, &basis, matrix->numRows) );
  TU_GRAPH_EDGE* cobasis = NULL;
  ASSERT_TU_CALL( TUallocBlockArray(tu, &cobasis, matrix->numColumns) );
  bool isGraphic;
  TU_CHRMAT* transpose = NULL;
  ASSERT_TU_CALL( TUchrmatTranspose(tu, matrix, &transpose) );

  ASSERT_TU_CALL( testGraphicnessTDecomposition(tu, matrix, transpose, &isGraphic, graph, basis,
    cobasis, NULL, mergeLeafBonds) );

  ASSERT_TRUE( isGraphic );
  ASSERT_TRUE( basis );
  ASSERT_TRUE( cobasis );

  ASSERT_TU_CALL( TUchrmatFree(tu, &transpose) );

  TU_CHRMAT* result = NULL;
  ASSERT_TU_CALL( TUconvertGraphToBinaryMatrix(tu, graph, &result, matrix->numRows, basis,
    matrix->numColumns, cobasis) );

  ASSERT_TRUE( result );

  if (TUchrmatCheckEqual(matrix, result))
  {
    printf("The representation matrix of represented graph is equal to input matrix.\n");
  }
  else
  {
    printf("Input matrix:\n");
    ASSERT_TU_CALL( TUchrmatPrintDense(stdout, matrix, ' ', true) );

    printf("Graph:\n");
    ASSERT_TU_CALL( TUgraphPrint(stdout, graph) );

    printf("Representation matrix:\n");
    ASSERT_TU_CALL( TUchrmatPrintDense(stdout, result, ' ', true) );
  }

  ASSERT_TU_CALL( TUgraphFree(tu, &graph) );
  ASSERT_TU_CALL( TUfreeBlockArray(tu, &basis) );
  ASSERT_TU_CALL( TUfreeBlockArray(tu, &cobasis) );
  ASSERT_TU_CALL( TUchrmatFree(tu, &result) );
}

TEST(Graphic, RootBondTwoOneEnds)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );

  /* A root bond (attached to a polygon) with two child markers, each containing a (single-edge-) path. */
  {
    TU_CHRMAT* A = NULL;
    ASSERT_TU_CALL( stringToCharMatrix(tu, &A, "3 3 "
      "1 1  1 "
      "1 0  1 "
      "0 1  1 "
    ) );
    testGraphicMatrix(tu, A, 0);
    ASSERT_TU_CALL( TUchrmatFree(tu, &A) );
  }

  /* A bond with an additional edge and two child markers, each containing a (single-edge-) path. */
  {
    TU_CHRMAT* A = NULL;
    ASSERT_TU_CALL( stringToCharMatrix(tu, &A, "3 4 "
      "1 1 1  1 "
      "0 1 0  1 "
      "0 0 1  1 "
    ) );
    testGraphicMatrix(tu, A, 0);
    ASSERT_TU_CALL( TUchrmatFree(tu, &A) );
  }

  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, RootBondOneTwoEnd)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );

  /* A root bond with a K_4 prime as a child, whose tree nodes are parallel edges. */
  {
    TU_CHRMAT* A = NULL;
    ASSERT_TU_CALL( stringToCharMatrix(tu, &A, "3 5 "
      "1 1 1 1  1 "
      "0 0 1 1  1 "
      "0 1 0 1  1 "
    ) );
    testGraphicMatrix(tu, A, 0);
    ASSERT_TU_CALL( TUchrmatFree(tu, &A) );
  }

  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, InternalBondOneOneEnd)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );

  /* A triangle linked to a small bond that is linked to a triangle.
   * Path contains one edge of each member. */
  {
    TU_CHRMAT* A = NULL;
    ASSERT_TU_CALL( stringToCharMatrix(tu, &A, "3 3 "
      "1 0  1 " /* edge of first triangle */
      "1 1  1 " /* bond edge */
      "0 1  1 " /* edge of second triangle */
    ) );
    testGraphicMatrix(tu, A, 0);
    ASSERT_TU_CALL( TUchrmatFree(tu, &A) );
  }

  /* A triangle linked to a large bond that is linked to a triangle.
   * Path contains one edge of each member. */
  {
    TU_CHRMAT* A = NULL;
    ASSERT_TU_CALL( stringToCharMatrix(tu, &A, "3 4 "
      "1 0 0  1 " /* edge of first triangle */
      "1 1 1  1 " /* bond edge */
      "0 0 1  1 " /* edge of second triangle */
    ) );
    testGraphicMatrix(tu, A, 0);
    ASSERT_TU_CALL( TUchrmatFree(tu, &A) );
  }

  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, Bond)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  {
    TU_CHRMAT* A = NULL;
    ASSERT_TU_CALL( stringToCharMatrix(tu, &A, "1 1 "
      "1 "
    ) );
    testGraphicMatrix(tu, A, 0);
    ASSERT_TU_CALL( TUchrmatFree(tu, &A) );
  }
  {
    TU_CHRMAT* A = NULL;
    ASSERT_TU_CALL( stringToCharMatrix(tu, &A, "1 3 "
      "1 1 1 "
    ) );
    testGraphicMatrix(tu, A, 0);
    ASSERT_TU_CALL( TUchrmatFree(tu, &A) );
  }
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, Polygon)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  {
    TU_CHRMAT* A = NULL;
    ASSERT_TU_CALL( stringToCharMatrix(tu, &A, "4 1 "
      "1 "
      "1 "
      "1 "
      "1 "
    ) );
    testGraphicMatrix(tu, A, 0);
    ASSERT_TU_CALL( TUchrmatFree(tu, &A) );
  }
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, PolygonPlusEdge)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );

  {
    TU_CHRMAT* A = NULL;
    ASSERT_TU_CALL( stringToCharMatrix(tu, &A, "7 2 "
      "1 0 " // 0
      "1 0 " // 1
      "1 0 " // 2
      "1 0 " // 3
      "1 0 " // 4
      "1 0 " // 5
      "1 1 " // 6
    ) );
    testGraphicMatrix(tu, A, 0);
    ASSERT_TU_CALL( TUchrmatFree(tu, &A) );
  }

  {
    TU_CHRMAT* A = NULL;
    ASSERT_TU_CALL( stringToCharMatrix(tu, &A, "7 2 "
      "1 0 " // 0
      "1 1 " // 1
      "1 0 " // 2
      "1 1 " // 3
      "1 1 " // 4
      "1 0 " // 5
      "1 1 " // 6
    ) );
    testGraphicMatrix(tu, A, 0);
    ASSERT_TU_CALL( TUchrmatFree(tu, &A) );
  }

  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}

TEST(Graphic, BixbyWagnerAppendix)
{
  TU* tu = NULL;
  ASSERT_TU_CALL( TUcreateEnvironment(&tu) );
  {
    TU_CHRMAT* A = NULL;
    ASSERT_TU_CALL( stringToCharMatrix(tu, &A, "9 7 "
      "1 0 0 0 0 0 1 " // 0
      "0 1 1 1 0 0 1 " // 1
      "0 1 1 1 0 0 1 " // 2
      "0 1 1 1 0 1 1 " // 3
      "0 0 0 0 1 1 1 " // 4
      "0 1 1 1 1 0 0 " // 5
      "1 1 1 1 0 0 0 " // 6
      "1 1 0 1 0 0 0 " // 7
      "1 0 0 0 0 0 0 " // 8
    ) );
    testGraphicMatrix(tu, A, 0);
    ASSERT_TU_CALL( TUchrmatFree(tu, &A) );
  }
  ASSERT_TU_CALL( TUfreeEnvironment(&tu) );
}
