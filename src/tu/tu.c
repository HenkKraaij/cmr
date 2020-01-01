#include <tu/tu.h>

#include "matrix_internal.h"
#include "one_sum.h"
#include "sign_internal.h"

#include <stdlib.h>
#include <assert.h>


/**
 * \brief Tests the 1-sum of sparse char matrices for total unimodularity.
 *
 * Returns \c true if and only if \p matrix is TU.
 *
 * If \p decomposition is not \c NULL and the algorithm has to test regularity of the support
 * matrix, then \c *decomposition will point to a decomposition tree for which the caller must use
 * \ref TUfreeDec to free memory. It is set to \c NULL otherwise.
 *
 * If \p submatrix is not \c NULL and the matrix is not TU, then a submatrix with an absolute
 * determinant larger than 1 will be searched, which may cause extra computational effort. In this
 * case, \c *submatrix will point to this submatrix for which the caller must use 
 * \ref TUfreeSubmatrix to free memory. It is set to \c NULL otherwise.
 */

static bool testTotalUnimodularityOneSum(TU* tu, int numComponents,
  TU_ONESUM_COMPONENT_CHAR* components, TU_DEC** decomposition, TU_SUBMATRIX** submatrix)
{
  assert(numComponents >= 0);
  assert(components);


  assert(false);

  for (int c = 0; c < numComponents; ++c)
  {
    TUclearSparseChar(&components[c].matrix);
    TUclearSparseChar(&components[c].transpose);
    free(components[c].rowsToOriginal);
    free(components[c].columnsToOriginal);
  }

  return true;
}

bool TUtestTotalUnimodularityDouble(TU* tu, TU_SPARSE_DOUBLE* matrix, double epsilon,
  TU_DEC** decomposition, TU_SUBMATRIX** submatrix)
{
  int numComponents;
  TU_ONESUM_COMPONENT_CHAR* components;

  assert(tu);
  assert(matrix);

  /* Check entries. */

  if (!TUisTernaryDouble(matrix, epsilon, submatrix))
    return false;

  /* Perform 1-sum decomposition. */

  decomposeOneSumDoubleToChar(tu, matrix, &numComponents, &components, NULL, NULL, NULL, NULL);

  /* Check correct signing for each component. */

  for (int comp = 0; comp < numComponents; ++comp)
  {
    TU_SUBMATRIX* compSubmatrix;
    char signFailed = signSequentiallyConnected(tu, &components[comp].matrix,
      &components[comp].transpose, false, submatrix ? &compSubmatrix : NULL);

    if (signFailed)
    {
      if (submatrix)
      {
        /* Translate component indices to indices of whole matrix and sort them again. */
        for (int r = 0; r < compSubmatrix->numRows; ++r)
          compSubmatrix->rows[r] = components[comp].rowsToOriginal[compSubmatrix->rows[r]];
        for (int c = 0; c < compSubmatrix->numColumns; ++c)
          compSubmatrix->columns[c] = components[comp].columnsToOriginal[compSubmatrix->columns[c]];
        TUsortSubmatrix(compSubmatrix);
        *submatrix = compSubmatrix;
      }

      for (int c = 0; c < numComponents; ++c)
      {
        TUclearSparseChar(&components[c].matrix);
        TUclearSparseChar(&components[c].transpose);
        free(components[c].rowsToOriginal);
        free(components[c].columnsToOriginal);
      }

      return false;
    }
  }

  return testTotalUnimodularityOneSum(tu, numComponents, components, decomposition, submatrix);
}

bool TUtestTotalUnimodularityInt(TU* tu, TU_SPARSE_INT* matrix, TU_DEC** decomposition,
  TU_SUBMATRIX** submatrix)
{
  int numComponents;
  TU_ONESUM_COMPONENT_CHAR* components;

  assert(tu);
  assert(matrix);

  /* Check entries. */

  if (!TUisTernaryInt(matrix, submatrix))
    return false;

  /* Perform 1-sum decomposition. */

  decomposeOneSumIntToChar(tu, matrix, &numComponents, &components, NULL, NULL, NULL, NULL);

  /* Check correct signing for each component. */

  for (int comp = 0; comp < numComponents; ++comp)
  {
    TU_SUBMATRIX* compSubmatrix;
    char signFailed = signSequentiallyConnected(tu, &components[comp].matrix,
      &components[comp].transpose, false, submatrix ? &compSubmatrix : NULL);

    if (signFailed)
    {
      if (submatrix)
      {
        /* Translate component indices to indices of whole matrix and sort them again. */
        for (int r = 0; r < compSubmatrix->numRows; ++r)
          compSubmatrix->rows[r] = components[comp].rowsToOriginal[compSubmatrix->rows[r]];
        for (int c = 0; c < compSubmatrix->numColumns; ++c)
          compSubmatrix->columns[c] = components[comp].columnsToOriginal[compSubmatrix->columns[c]];
        TUsortSubmatrix(compSubmatrix);
        *submatrix = compSubmatrix;
      }

      for (int c = 0; c < numComponents; ++c)
      {
        TUclearSparseChar(&components[c].matrix);
        TUclearSparseChar(&components[c].transpose);
        free(components[c].rowsToOriginal);
        free(components[c].columnsToOriginal);
      }

      return false;
    }
  }

  return testTotalUnimodularityOneSum(tu, numComponents, components, decomposition, submatrix);
}

bool TUtestTotalUnimodularityChar(TU* tu, TU_SPARSE_CHAR* matrix, TU_DEC** decomposition,
  TU_SUBMATRIX** submatrix)
{
  int numComponents;
  TU_ONESUM_COMPONENT_CHAR* components;

  assert(tu);
  assert(matrix);

  /* Check entries. */

  if (!TUisTernaryChar(matrix, submatrix))
    return false;

  /* Perform 1-sum decomposition. */

  decomposeOneSumCharToChar(tu, matrix, &numComponents, &components, NULL, NULL, NULL, NULL);

  /* Check correct signing for each component. */

  for (int comp = 0; comp < numComponents; ++comp)
  {
    TU_SUBMATRIX* compSubmatrix;
    char signFailed = signSequentiallyConnected(tu, &components[comp].matrix,
      &components[comp].transpose, false, submatrix ? &compSubmatrix : NULL);

    if (signFailed)
    {
      if (submatrix)
      {
        /* Translate component indices to indices of whole matrix and sort them again. */
        for (int r = 0; r < compSubmatrix->numRows; ++r)
          compSubmatrix->rows[r] = components[comp].rowsToOriginal[compSubmatrix->rows[r]];
        for (int c = 0; c < compSubmatrix->numColumns; ++c)
          compSubmatrix->columns[c] = components[comp].columnsToOriginal[compSubmatrix->columns[c]];
        TUsortSubmatrix(compSubmatrix);
        *submatrix = compSubmatrix;
      }

      for (int c = 0; c < numComponents; ++c)
      {
        TUclearSparseChar(&components[c].matrix);
        TUclearSparseChar(&components[c].transpose);
        free(components[c].rowsToOriginal);
        free(components[c].columnsToOriginal);
      }

      return false;
    }
  }

  return testTotalUnimodularityOneSum(tu, numComponents, components, decomposition, submatrix);
}
