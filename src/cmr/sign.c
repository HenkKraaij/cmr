// #define CMR_DEBUG /* Uncomment to debug graphic. */

#include <cmr/sign.h>
#include "sign_internal.h"
#include "matrix_internal.h"
#include "one_sum.h"
#include "env_internal.h"

#include <assert.h>
#include <stdlib.h>

/**
 * \brief Graph node for BFS in signing algorithm.
 */

typedef struct
{
  int status;             /**< \brief 0: not visited, 1: in queue, 2: processed */
  int predecessorNode;    /**< \brief Node number of predecessor. */
  char predecessorValue;  /**< \brief Value of matrix entry of predecessor. */
  char targetValue;       /**< \brief Entry in current row if a target node, and 0 otherwise. */
} GRAPH_NODE;

CMR_ERROR signSequentiallyConnected(
  CMR* cmr,                 /**< \ref CMR environment. */
  CMR_CHRMAT* matrix,      /**< The matrix to be signed. */
  CMR_CHRMAT* transpose,   /**< The transpose of \p matrix. */
  bool change,            /**< Whether to modify the matrix. */
  char* pmodification,    /**< Pointer for storing which matrix was modified.*/
  CMR_SUBMAT** psubmatrix  /**< Pointer for storing a submatrix with bad determinant (may be \c NULL). */
)
{
  assert(cmr);
  assert(matrix);
  assert(transpose);
  assert(pmodification);

  assert(CMRchrmatCheckTranspose(matrix, transpose));
  assert(CMRisTernaryChr(cmr, matrix, NULL));

  /* If we have more rows than columns, we work with the transpose. */
  if (matrix->numRows > matrix->numColumns)
  {
    CMR_CALL( signSequentiallyConnected(cmr, transpose, matrix, change, pmodification, psubmatrix) );
    assert(*pmodification == 0 || *pmodification == 'm');
    if (psubmatrix && *psubmatrix)
    {
      assert((*psubmatrix)->numRows == (*psubmatrix)->numColumns);
      size_t* tmp = (*psubmatrix)->rows;
      (*psubmatrix)->rows = (*psubmatrix)->columns;
      (*psubmatrix)->columns = tmp;
    }
    if (*pmodification == 'm')
      *pmodification = 't';
    return CMR_OKAY;
  }

  CMRdbgMsg(2, "signSequentiallyConnected.\n");

  *pmodification = 0;
  const int firstRowNode = matrix->numColumns;
  GRAPH_NODE* graphNodes = NULL;
  int* bfsQueue = NULL;
  int bfsQueueBegin = 0;
  int bfsQueueEnd = 0;

  CMRallocStackArray(cmr, &graphNodes, matrix->numColumns + matrix->numRows);
  CMRallocStackArray(cmr, &bfsQueue, matrix->numColumns + matrix->numRows);

  /* Main loop iterates over the rows. */
  for (int row = 1; row < matrix->numRows; ++row)
  {
    CMRdbgMsg(2, "Before processing row %d:\n", row);
#if defined(CMR_DEBUG)
    CMRchrmatPrintDense(stdout, matrix, ' ', true);
#endif

    for (int v = 0; v < matrix->numColumns + matrix->numRows; ++v)
    {
      graphNodes[v].targetValue = 0;
      graphNodes[v].status = 0;
      graphNodes[v].predecessorNode = -1;
    }

    bool rowChanged = false;
    int begin = matrix->rowStarts[row];
    int end = matrix->rowStarts[row+1];
    if (begin == end)
    {
      CMRdbgMsg(2, "Empty row.\n");
      continue;
    }

    /* First nonzero in row determines start column node. */
    int startNode = matrix->entryColumns[begin];
    /* All columns of the row's nonzeros are target column nodes. */
    for (int e = begin; e < end; ++e)
      graphNodes[matrix->entryColumns[e]].targetValue = matrix->entryValues[e];
    bfsQueue[0] = startNode;
    graphNodes[startNode].status = 1;
    bfsQueueBegin = 0;
    bfsQueueEnd = 1;

    while (bfsQueueBegin < bfsQueueEnd)
    {
      int currentNode = bfsQueue[bfsQueueBegin];
      assert(graphNodes[currentNode].status == 1);
      graphNodes[currentNode].status = 2;
      ++bfsQueueBegin;

      if (currentNode >= firstRowNode)
      {
        int r = currentNode - firstRowNode;
        CMRdbgMsg(4, "Current node is %d (row %d), queue length is %d\n", currentNode, r, bfsQueueEnd - bfsQueueBegin);

        /* Iterate over outgoing edges. */
        begin = matrix->rowStarts[r];
        end = matrix->rowStarts[r+1];
        for (int e = begin; e < end; ++e)
        {
          int c = matrix->entryColumns[e];
          if (graphNodes[c].status == 0)
          {
            graphNodes[c].status = 1;
            graphNodes[c].predecessorNode = currentNode;
            graphNodes[c].predecessorValue = matrix->entryValues[e];
            bfsQueue[bfsQueueEnd++] = c;
            /* If we reach a target node for the first time, we trace back to the previous target
               node (which might be the starting node). */
            if (graphNodes[c].targetValue != 0)
            {
              int length = 2;
              int sum = graphNodes[c].targetValue;
              int pathNode = c;
              do
              {
                sum += graphNodes[pathNode].predecessorValue;
                pathNode = graphNodes[pathNode].predecessorNode;
                ++length;
              }
              while (graphNodes[pathNode].targetValue == 0);
              sum += graphNodes[pathNode].targetValue;
              CMRdbgMsg(6, "Found a chordless cycle between %d and %d with sum %d of length %d\n", c, pathNode, sum,
                length);

              if (sum % 4 != 0)
              {
                assert(sum % 4 == -2 || sum % 4 == 2);

                /* If we didn't find a submatrix yet: */
                if (psubmatrix && *psubmatrix == NULL)
                {
                  int i = 1;
                  int j = 1;
                  CMR_CALL( CMRsubmatCreate(cmr, length/2, length/2, psubmatrix) );
                  CMR_SUBMAT* submatrix = *psubmatrix;
                  pathNode = c;
                  submatrix->columns[0] = c;
                  submatrix->rows[0] = row;
                  do
                  {
                    pathNode = graphNodes[pathNode].predecessorNode;
                    if (pathNode >= firstRowNode)
                      submatrix->rows[i++] = pathNode - firstRowNode;
                    else
                      submatrix->columns[j++] = pathNode;
                  }
                  while (graphNodes[pathNode].targetValue == 0);
                  CMR_CALL( CMRsortSubmatrix(cmr, submatrix) );

                  CMRdbgMsg(6, "Submatrix filled with %d rows and %d columns.\n", i, j);
                }
                CMRdbgMsg(6, "Sign change required.\n");
                graphNodes[c].targetValue *= -1;
                *pmodification = 'm';
                if (change)
                  rowChanged = true;
                else
                {
                  CMRfreeStackArray(cmr, &bfsQueue);
                  CMRfreeStackArray(cmr, &graphNodes);
                  return CMR_OKAY;
                }
              }
            }
          }
        }
      }
      else
      {
        int c = currentNode;
        CMRdbgMsg(4, "Current node is %d (column %d), queue length is %d\n", currentNode, c,
          bfsQueueEnd - bfsQueueBegin);

        /* Iterate over outgoing edges. */
        begin = transpose->rowStarts[c];
        end = transpose->rowStarts[c+1];
        for (int e = begin; e < end; ++e)
        {
          int r = transpose->entryColumns[e];
          /* Only rows before current iteration row participate. */
          if (r >= row)
            break;
          if (graphNodes[firstRowNode + r].status == 0)
          {
            graphNodes[firstRowNode + r].status = 1;
            graphNodes[firstRowNode + r].predecessorNode = currentNode;
            graphNodes[firstRowNode + r].predecessorValue = transpose->entryValues[e];
            bfsQueue[bfsQueueEnd++] = firstRowNode + r;
          }
        }
      }
    }

#if defined(CMR_DEBUG)
    for (int v = 0; v < matrix->numColumns + row; ++v)
    {
      if (v == startNode)
        CMRdbgMsg(4, "Source node ");
      else if (graphNodes[v].targetValue != 0)
        CMRdbgMsg(4, "Target node ");
      else
        CMRdbgMsg(4, "Node ");
      CMRdbgMsg(0, "%d is %s%d and has predecessor %d.\n", v, v >= firstRowNode ? "row ": "column ",
        v >= firstRowNode ? v-firstRowNode : v, graphNodes[v].predecessorNode);
    }
#endif

    if (rowChanged)
    {
      begin = matrix->rowStarts[row];
      end = matrix->rowStarts[row+1];
      for (int e = begin; e < end; ++e)
      {
        int column = matrix->entryColumns[e];
        if (matrix->entryValues[e] != graphNodes[column].targetValue)
          CMRdbgMsg(2, "Sign change at %d,%d.\n", row, column);
        matrix->entryValues[e] = graphNodes[column].targetValue;
      }
    }
  }

#if defined(CMR_DEBUG)
  if (change)
  {
    CMRdbgMsg(2, "After signing:\n");
    CMR_CALL( CMRchrmatPrintDense(stdout, matrix, ' ', true) );
  }
#endif /* CMR_DEBUG */

  CMRfreeStackArray(cmr, &bfsQueue);
  CMRfreeStackArray(cmr, &graphNodes);

  return CMR_OKAY;
}

/**
 * \brief Signs a given ternary double matrix.
 */

static
CMR_ERROR signDbl(
  CMR* cmr,                 /**< \ref CMR environment. */
  CMR_DBLMAT* matrix,      /**< Sparse double matrix. */
  bool change,            /**< Whether the signs of \p matrix shall be modified. */
  bool* palreadySigned,   /**< Pointer for storing whether \p matrix was already signed correctly. */
  CMR_SUBMAT** psubmatrix  /**< Pointer for storing a submatrix with bad determinant (may be \c NULL). */
)
{
  assert(cmr);
  assert(matrix);
  assert(palreadySigned);

  size_t numComponents;
  CMR_ONESUM_COMPONENT* components = NULL;

  assert(CMRisTernaryDbl(cmr, matrix, 1.0e-3, NULL));

#if defined(CMR_DEBUG)
  CMRdbgMsg(0, "sign:\n");
  CMRdblmatPrintDense(stdout, matrix, ' ', true);
#endif /* CMR_DEBUG */

  /* Decompose into 1-connected components. */

  CMR_CALL( decomposeOneSum(cmr, (CMR_MATRIX*) matrix, sizeof(double), sizeof(double), &numComponents, &components, NULL,
    NULL, NULL, NULL) );

  *palreadySigned = true;
  for (int comp = 0; comp < numComponents; ++comp)
  {
    CMR_SUBMAT* compSubmatrix = NULL;

    CMRdbgMsg(2, "-> Component %d of size %dx%d\n", comp, components[comp].matrix->numRows,
      components[comp].matrix->numColumns);

    char modified;
    CMR_CALL( signSequentiallyConnected(cmr, (CMR_CHRMAT*) components[comp].matrix,
      (CMR_CHRMAT*) components[comp].transpose, change, &modified,
      (psubmatrix && !*psubmatrix) ? &compSubmatrix : NULL) );

    CMRdbgMsg(2, "-> Component %d yields: %c\n", comp, modified ? modified : '0');

    if (modified == 0)
    {
      assert(compSubmatrix == NULL);
      continue;
    }

    *palreadySigned = false;

    /* If we found a submatrix for the first time: */
    if (compSubmatrix)
    {
      assert(psubmatrix && !*psubmatrix);
      /* Translate component indices to indices of whole matrix and sort them again. */
      for (int r = 0; r < compSubmatrix->numRows; ++r)
        compSubmatrix->rows[r] = components[comp].rowsToOriginal[compSubmatrix->rows[r]];
      for (int c = 0; c < compSubmatrix->numColumns; ++c)
        compSubmatrix->columns[c] = components[comp].columnsToOriginal[compSubmatrix->columns[c]];
      CMR_CALL( CMRsortSubmatrix(cmr, compSubmatrix) );
      *psubmatrix = compSubmatrix;
    }

    /* As we don't modify, we can abort early. */
    if (!change)
      break;

    assert(modified == 'm' || modified == 't');
    bool copyTranspose = modified == 't';

    /* Either the matrix or its transposed was modified. */
    CMR_DBLMAT* sourceMatrix = copyTranspose ?
      (CMR_DBLMAT*) components[comp].transpose :
      (CMR_DBLMAT*) components[comp].matrix;

    /* We have to copy the changes back to the original matrix. */
    for (int sourceRow = 0; sourceRow < sourceMatrix->numRows; ++sourceRow)
    {
      int sourceBegin = sourceMatrix->rowStarts[sourceRow];
      int sourceEnd = sourceMatrix->rowStarts[sourceRow + 1];
      for (int sourceEntry = sourceBegin; sourceEntry < sourceEnd; ++sourceEntry)
      {
        int sourceColumn = sourceMatrix->entryColumns[sourceEntry];
        int compRow = copyTranspose ? sourceColumn : sourceRow;
        int compColumn = copyTranspose ? sourceRow : sourceColumn;
        int row = components[comp].rowsToOriginal[compRow];
        int column = components[comp].columnsToOriginal[compColumn];

        /* Perform binary search in row of original matrix to find the column. */

        int lower = matrix->rowStarts[row];
        int upper = row + 1 < matrix->numRows ? matrix->rowStarts[row + 1] : matrix->numNonzeros;
        while (lower < upper)
        {
          int entry = (lower + upper) / 2;
          int searchColumn = matrix->entryColumns[entry];
          if (column < searchColumn)
            upper = entry;
          else if (column > searchColumn)
            lower = entry + 1;
          else
          {
            matrix->entryValues[entry] = sourceMatrix->entryValues[sourceEntry];
            break;
          }
        }
        assert(lower < upper);
      }
    }
  }

#if defined(CMR_DEBUG)
  if (!*palreadySigned && change)
  {
    CMRdbgMsg(0, "Modified original matrix:\n");
    CMRdblmatPrintDense(stdout, matrix, ' ', true);
  }
#endif /* CMR_DEBUG */

  /* Clean-up */

  for (int c = 0; c < numComponents; ++c)
  {
    CMRchrmatFree(cmr, (CMR_CHRMAT**) &components[c].matrix);
    CMRchrmatFree(cmr, (CMR_CHRMAT**) &components[c].transpose);
    CMRfreeBlockArray(cmr, &components[c].rowsToOriginal);
    CMRfreeBlockArray(cmr, &components[c].columnsToOriginal);
  }
  CMRfreeBlockArray(cmr, &components);

  return CMR_OKAY;
}

CMR_ERROR CMRtestSignDbl(CMR* cmr, CMR_DBLMAT* matrix, bool* pcorrectSign, CMR_SUBMAT** psubmatrix)
{
  return signDbl(cmr, matrix, false, pcorrectSign, psubmatrix);
}

CMR_ERROR CMRcorrectSignDbl(CMR* cmr, CMR_DBLMAT* matrix, bool* palreadySigned, CMR_SUBMAT** psubmatrix)
{
  return signDbl(cmr, matrix, true, palreadySigned, psubmatrix);
}

/**
 * \brief Signs a given ternary int matrix.
 */

static
CMR_ERROR signInt(
  CMR* cmr,                 /**< \ref CMR environment. */
  CMR_INTMAT* matrix,      /**< Sparse int matrix. */
  bool change,            /**< Whether the signs of \p matrix shall be modified. */
  bool* palreadySigned,   /**< Pointer for storing whether \p matrix was already signed correctly. */
  CMR_SUBMAT** psubmatrix  /**< Pointer for storing a submatrix with bad determinant (may be \c NULL). */
)
{
  assert(cmr);
  assert(matrix);
  assert(palreadySigned);

  size_t numComponents;
  CMR_ONESUM_COMPONENT* components = NULL;

  assert(CMRisTernaryInt(cmr, matrix, NULL));

#if defined(CMR_DEBUG)
  CMRdbgMsg(0, "sign:\n");
  CMRintmatPrintDense(stdout, matrix, ' ', true);
#endif /* CMR_DEBUG */

  /* Decompose into 1-connected components. */

  CMR_CALL( decomposeOneSum(cmr, (CMR_MATRIX*) matrix, sizeof(int), sizeof(int), &numComponents, &components, NULL, NULL,
    NULL, NULL) );

  *palreadySigned = true;
  for (int comp = 0; comp < numComponents; ++comp)
  {
    CMR_SUBMAT* compSubmatrix = NULL;

    CMRdbgMsg(2, "-> Component %d of size %dx%d\n", comp, components[comp].matrix->numRows,
      components[comp].matrix->numColumns);

    char modified;
    CMR_CALL( signSequentiallyConnected(cmr, (CMR_CHRMAT*) components[comp].matrix,
      (CMR_CHRMAT*) components[comp].transpose, change, &modified,
      (psubmatrix && !*psubmatrix) ? &compSubmatrix : NULL) );

    CMRdbgMsg(2, "-> Component %d yields: %c\n", comp, modified ? modified : '0');

    if (modified == 0)
    {
      assert(compSubmatrix == NULL);
      continue;
    }

    *palreadySigned = false;

    /* If we found a submatrix for the first time: */
    if (compSubmatrix)
    {
      assert(psubmatrix && !*psubmatrix);
      /* Translate component indices to indices of whole matrix and sort them again. */
      for (int r = 0; r < compSubmatrix->numRows; ++r)
        compSubmatrix->rows[r] = components[comp].rowsToOriginal[compSubmatrix->rows[r]];
      for (int c = 0; c < compSubmatrix->numColumns; ++c)
        compSubmatrix->columns[c] = components[comp].columnsToOriginal[compSubmatrix->columns[c]];
      CMRsortSubmatrix(cmr, compSubmatrix);
      *psubmatrix = compSubmatrix;
    }

    /* As we don't modify, we can abort early. */
    if (!change)
      break;

    assert(modified == 'm' || modified == 't');
    bool copyTranspose = modified == 't';

    /* Either the matrix or its transposed was modified. */
    CMR_CHRMAT* sourceMatrix = copyTranspose ?
      (CMR_CHRMAT*) components[comp].transpose :
      (CMR_CHRMAT*) components[comp].matrix;

    /* We have to copy the changes back to the original matrix. */
    for (int sourceRow = 0; sourceRow < sourceMatrix->numRows; ++sourceRow)
    {
      int sourceBegin = sourceMatrix->rowStarts[sourceRow];
      int sourceEnd = sourceMatrix->rowStarts[sourceRow + 1];
      for (int sourceEntry = sourceBegin; sourceEntry < sourceEnd; ++sourceEntry)
      {
        int sourceColumn = sourceMatrix->entryColumns[sourceEntry];
        int compRow = copyTranspose ? sourceColumn : sourceRow;
        int compColumn = copyTranspose ? sourceRow : sourceColumn;
        int row = components[comp].rowsToOriginal[compRow];
        int column = components[comp].columnsToOriginal[compColumn];

        /* Perform binary search in row of original matrix to find the column. */

        int lower = matrix->rowStarts[row];
        int upper = row + 1 < matrix->numRows ? matrix->rowStarts[row + 1] : matrix->numNonzeros;
        while (lower < upper)
        {
          int entry = (lower + upper) / 2;
          int searchColumn = matrix->entryColumns[entry];
          if (column < searchColumn)
            upper = entry;
          else if (column > searchColumn)
            lower = entry + 1;
          else
          {
            matrix->entryValues[entry] = sourceMatrix->entryValues[sourceEntry];
            break;
          }
        }
        assert(lower < upper);
      }
    }
  }

#if defined(CMR_DEBUG)
  if (!*palreadySigned && change)
  {
    CMRdbgMsg(0, "Modified original matrix:\n");
    CMRintmatPrintDense(stdout, matrix, ' ', true);
  }
#endif /* CMR_DEBUG */

  /* Clean-up */

  for (int c = 0; c < numComponents; ++c)
  {
    CMRchrmatFree(cmr, (CMR_CHRMAT**) &components[c].matrix);
    CMRchrmatFree(cmr, (CMR_CHRMAT**) &components[c].transpose);
    CMRfreeBlockArray(cmr, &components[c].rowsToOriginal);
    CMRfreeBlockArray(cmr, &components[c].columnsToOriginal);
  }
  CMRfreeBlockArray(cmr, &components);

  return CMR_OKAY;
}

CMR_ERROR CMRtestSignInt(CMR* cmr, CMR_INTMAT* matrix, bool* pcorrectSign, CMR_SUBMAT** psubmatrix)
{
  return signInt(cmr, matrix, false, pcorrectSign, psubmatrix);
}

CMR_ERROR CMRcorrectSignInt(CMR* cmr, CMR_INTMAT* matrix, bool* palreadySigned, CMR_SUBMAT** psubmatrix)
{
  return signInt(cmr, matrix, true, palreadySigned, psubmatrix);
}


/**
 * \brief Signs a given ternary char matrix.
 */

static
CMR_ERROR signChr(
  CMR* cmr,                 /**< \ref CMR environment. */
  CMR_CHRMAT* matrix,      /**< Sparse char matrix. */
  bool change,            /**< Whether the signs of \p matrix shall be modified. */
  bool* palreadySigned,   /**< Pointer for storing whether \p matrix was already signed correctly. */
  CMR_SUBMAT** psubmatrix  /**< Pointer for storing a submatrix with bad determinant (may be \c NULL). */
)
{
  assert(cmr);
  assert(matrix);
  assert(!psubmatrix || !*psubmatrix);

  size_t numComponents;
  CMR_ONESUM_COMPONENT* components = NULL;

  assert(CMRisTernaryChr(cmr, matrix, NULL));

#if defined(CMR_DEBUG)
  CMRdbgMsg(0, "sign:\n");
  CMRchrmatPrintDense(stdout, matrix, ' ', true);
#endif /* CMR_DEBUG */

  /* Decompose into 1-connected components. */

  CMR_CALL( decomposeOneSum(cmr, (CMR_MATRIX*) matrix, sizeof(char), sizeof(char), &numComponents, &components, NULL, NULL,
    NULL, NULL) );

  if (palreadySigned)
    *palreadySigned = true;
  for (int comp = 0; comp < numComponents; ++comp)
  {
    CMR_SUBMAT* compSubmatrix = NULL;

    CMRdbgMsg(2, "-> Component %d of size %dx%d\n", comp, components[comp].matrix->numRows,
      components[comp].matrix->numColumns);

    char modified;
    CMR_CALL( signSequentiallyConnected(cmr, (CMR_CHRMAT*) components[comp].matrix,
      (CMR_CHRMAT*) components[comp].transpose, change, &modified,
      (psubmatrix && !*psubmatrix) ? &compSubmatrix : NULL) );

    CMRdbgMsg(2, "-> Component %d yields: %c\n", comp, modified ? modified : '0');

    if (modified == 0)
    {
      assert(compSubmatrix == NULL);
      continue;
    }

    if (palreadySigned)
      *palreadySigned = false;

    /* If we found a submatrix for the first time: */
    if (compSubmatrix)
    {
      assert(psubmatrix && !*psubmatrix);
      /* Translate component indices to indices of whole matrix and sort them again. */
      for (int r = 0; r < compSubmatrix->numRows; ++r)
        compSubmatrix->rows[r] = components[comp].rowsToOriginal[compSubmatrix->rows[r]];
      for (int c = 0; c < compSubmatrix->numColumns; ++c)
        compSubmatrix->columns[c] = components[comp].columnsToOriginal[compSubmatrix->columns[c]];
      CMRsortSubmatrix(cmr, compSubmatrix);
      *psubmatrix = compSubmatrix;
    }

    /* As we don't modify, we can abort early. */
    if (!change)
      break;

    assert(modified == 'm' || modified == 't');
    bool copyTranspose = modified == 't';

    /* Either the matrix or its transposed was modified. */
    CMR_CHRMAT* sourceMatrix = copyTranspose ?
      (CMR_CHRMAT*) components[comp].transpose :
      (CMR_CHRMAT*) components[comp].matrix;

    /* We have to copy the changes back to the original matrix. */
    for (int sourceRow = 0; sourceRow < sourceMatrix->numRows; ++sourceRow)
    {
      int sourceBegin = sourceMatrix->rowStarts[sourceRow];
      int sourceEnd = sourceMatrix->rowStarts[sourceRow + 1];
      for (int sourceEntry = sourceBegin; sourceEntry < sourceEnd; ++sourceEntry)
      {
        int sourceColumn = sourceMatrix->entryColumns[sourceEntry];
        int compRow = copyTranspose ? sourceColumn : sourceRow;
        int compColumn = copyTranspose ? sourceRow : sourceColumn;
        int row = components[comp].rowsToOriginal[compRow];
        int column = components[comp].columnsToOriginal[compColumn];

        /* Perform binary search in row of original matrix to find the column. */

        int lower = matrix->rowStarts[row];
        int upper = row + 1 < matrix->numRows ? matrix->rowStarts[row + 1] : matrix->numNonzeros;
        while (lower < upper)
        {
          int entry = (lower + upper) / 2;
          int searchColumn = matrix->entryColumns[entry];
          if (column < searchColumn)
            upper = entry;
          else if (column > searchColumn)
            lower = entry + 1;
          else
          {
            matrix->entryValues[entry] = sourceMatrix->entryValues[sourceEntry];
            break;
          }
        }
        assert(lower < upper);
      }
    }
  }

#if defined(CMR_DEBUG)
  if (palreadySigned && !*palreadySigned && change)
  {
    CMRdbgMsg(0, "Modified original matrix:\n");
    CMRchrmatPrintDense(stdout, matrix, ' ', true);
  }
#endif /* CMR_DEBUG */

  /* Clean-up */

  for (int c = 0; c < numComponents; ++c)
  {
    CMRchrmatFree(cmr, (CMR_CHRMAT**) &components[c].matrix);
    CMRchrmatFree(cmr, (CMR_CHRMAT**) &components[c].transpose);
    CMRfreeBlockArray(cmr, &components[c].rowsToOriginal);
    CMRfreeBlockArray(cmr, &components[c].columnsToOriginal);
  }
  CMRfreeBlockArray(cmr, &components);

  return CMR_OKAY;
}

CMR_ERROR CMRtestSignChr(CMR* cmr, CMR_CHRMAT* matrix, bool* pcorrectSign, CMR_SUBMAT** psubmatrix)
{
  return signChr(cmr, matrix, false, pcorrectSign, psubmatrix);
}

CMR_ERROR CMRcorrectSignChr(CMR* cmr, CMR_CHRMAT* matrix, bool* palreadySigned, CMR_SUBMAT** psubmatrix)
{
  return signChr(cmr, matrix, true, palreadySigned, psubmatrix);
}
