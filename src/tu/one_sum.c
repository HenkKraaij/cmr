#include "one_sum.h"

#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>

#include "env_internal.h"

struct GraphNode
{
  int adjacencyStart; /**< Index of first outgoing arc. */
  int component; /**< Index of component of matrix. */
  int degree; /**< Used to count the degree. */
  int order; /**< Corresponding row/column in component. */
};
typedef struct GraphNode GRAPH_NODE;

void decomposeOneSum(TU* tu, TU_MATRIX* matrix, size_t matrixType, size_t targetType,
  int* numComponents, TU_ONESUM_COMPONENT** components, int* rowsToComponents,
  int* columnsToComponents, int* rowsToComponentRows, int* columnsToComponentColumns)
{
  GRAPH_NODE* graphNodes = NULL;
  int* graphAdjacencies = NULL;
  int* queue = NULL;
  int queueLength = 0;
  int numNodes = matrix->numRows + matrix->numColumns;
  int countComponents = 0;
  const int firstColumnNode = matrix->numRows;
  int i;

  assert(tu != NULL);
  assert(matrix != NULL);
  assert(numComponents != NULL);
  assert(components != NULL);
  assert(*components == NULL);

#ifdef DEBUG_ONE_SUM
  printf("decomposeOneSum:\n");
  if (matrixType == sizeof(double))
    TUprintDoubleMatrixDense(stdout, (TU_DOUBLE_MATRIX*) matrix, '0', true);
  else if (matrixType == sizeof(int))
    TUprintIntMatrixDense(stdout, (TU_INT_MATRIX*) matrix, '0', true);
  else if (matrixType == sizeof(char))
    TUprintCharMatrixDense(stdout, (TU_CHAR_MATRIX*) matrix, '0', true);
#endif

  TUallocStackArray(tu, &graphNodes, numNodes + 1);
  TUallocStackArray(tu, &graphAdjacencies, 2 * matrix->numNonzeros);
  TUallocStackArray(tu, &queue, numNodes);

  for (int node = 0; node < numNodes; ++node)
  {
    graphNodes[node].component = -1;
    graphNodes[node].degree = 0;
  }

  /* Count degrees */
  for (int row = 0; row < matrix->numRows; ++row)
  {
    int start = matrix->rowStarts[row];
    int end = row + 1 < matrix->numRows ?  matrix->rowStarts[row+1] : matrix->numNonzeros;
    for (int e = start; e < end; ++e)
    {
      bool nonzero;
      if (matrixType == sizeof(double))
        nonzero = round(((double*)matrix->entryValues)[e]) != 0;
      else if (matrixType == sizeof(int))
        nonzero = ((int*)matrix->entryValues)[e] != 0;
      else
      {
        assert(matrixType == sizeof(char));
        nonzero = ((char*)matrix->entryValues)[e] != 0;
      }
      if (nonzero)
      {
        int column = matrix->entryColumns[e];
        graphNodes[row].degree++;
        graphNodes[firstColumnNode + column].degree++;
      }
    }
  }

  /* Compute ranges for adjacencies */
  i = 0;
  for (int node = 0; node < numNodes; ++node)
  {
    graphNodes[node].adjacencyStart = i;
    i += graphNodes[node].degree;
  }
  graphNodes[numNodes].adjacencyStart = i;

  /* Scan entries and create adjacencies. */
  for (int row = 0; row < matrix->numRows; ++row)
  {
    int start = matrix->rowStarts[row];
    int end = row + 1 < matrix->numRows ? matrix->rowStarts[row+1] : matrix->numNonzeros;
    for (int e = start; e < end; ++e)
    {
      bool nonzero;
      if (matrixType == sizeof(double))
        nonzero = round(((double*)matrix->entryValues)[e]) != 0;
      else if (matrixType == sizeof(int))
        nonzero = ((int*)matrix->entryValues)[e] != 0;
      else if (matrixType == sizeof(char))
        nonzero = ((char*)matrix->entryValues)[e] != 0;
      else
        assert("Invalid matrixType parameter." == 0);      
      if (nonzero)
      {
        int column = matrix->entryColumns[e];
        int columnNode = firstColumnNode + column;
        graphAdjacencies[graphNodes[row + 1].adjacencyStart - graphNodes[row].degree] = columnNode;
        graphNodes[row].degree--;
        graphAdjacencies[graphNodes[columnNode + 1].adjacencyStart - graphNodes[columnNode].degree]
          = row;
        graphNodes[columnNode].degree--;
      }
    }
  }

  /*
   * We decremented the degree entries, so they should be 0. From now on we can query 
   * graphNodes[node+1].adjacencyStart - graphNodes[node].adjacencyStart if necessary.
   * 
   * TODO: make degree a union with something else to save memory and improve cache behavior.
   */

  for (int node = 0; node < numNodes; ++node)
    assert(graphNodes[node].degree == 0);

  /* Run DFS. */
  for (int startNode = 0; startNode < numNodes; ++startNode)
  {
    if (graphNodes[startNode].component < 0)
    {
      /* Start a new component. */
      int currentOrderRow = 0;
      int currentOrderColumn = 0;

      graphNodes[startNode].component = countComponents;
      graphNodes[startNode].order = 0;
      if (startNode < firstColumnNode)
        currentOrderRow++;
      else
        currentOrderColumn++;
      queueLength = 1;
      queue[0] = startNode;
      while (queueLength > 0)
      {
        int currentNode = queue[--queueLength];
        int start = graphNodes[currentNode].adjacencyStart;
        int end = graphNodes[currentNode + 1].adjacencyStart;

        for (int i = start; i < end; ++i)
        {
          int endNode = graphAdjacencies[i];
          if (graphNodes[endNode].component < 0)
          {
            graphNodes[endNode].component = countComponents;
            if (endNode < firstColumnNode)
              graphNodes[endNode].order = currentOrderRow++;
            else
              graphNodes[endNode].order = currentOrderColumn++;
            queue[queueLength] = endNode;
            ++queueLength;
          }
        }
      }

      ++countComponents;
    }
  }

  *numComponents = countComponents;

#ifdef DEBUG_ONE_SUM
  printf("DFS found %d components.\n", countComponents);
  for (int node = 0; node < numNodes; ++node)
  {
    printf("Node %d has component %d.\n", node, graphNodes[node].component);
  }
#endif

  /* Allocate component data. */
  TUallocBlockArray(tu, components, countComponents);

  /* Compute sizes. */
  for (int component = 0; component < countComponents; ++component)
  {
    (*components)[component].matrix = NULL;
    (*components)[component].transpose = NULL;
    (*components)[component].rowsToOriginal = NULL;
    (*components)[component].columnsToOriginal = NULL;
    TUcreateCharMatrix(tu, (TU_CHAR_MATRIX**) &(*components)[component].matrix, 0, 0, 0);
  }

  for (int node = 0; node < numNodes; ++node)
  {
    int component = graphNodes[node].component;
    int start = graphNodes[node].adjacencyStart;
    int end = graphNodes[node + 1].adjacencyStart;
    assert(component >= 0);
    if (node < firstColumnNode)
    {
      (*components)[component].matrix->numRows++;
      (*components)[component].matrix->numNonzeros += end - start;
    }
    else
      (*components)[component].matrix->numColumns++;
  }

  /* Allocate memory */
  for (int component = 0; component < countComponents; ++component)
  {
    TU_MATRIX* compMatrix = (*components)[component].matrix;

#ifdef DEBUG_ONE_SUM
    printf("Component %d has %dx%d matrix with %d nonzeros.\n", component, compMatrix->numRows,
      compMatrix->numColumns, compMatrix->numNonzeros);
#endif

    TUallocBlockArray(tu, &(*components)[component].rowsToOriginal, compMatrix->numRows);
    TUallocBlockArray(tu, &(*components)[component].columnsToOriginal, compMatrix->numColumns);
    TUfreeBlockArray(tu, &compMatrix->rowStarts);
    TUallocBlockArray(tu, &compMatrix->rowStarts, compMatrix->numRows + 1);
    if (compMatrix->numNonzeros > 0)
      TUallocBlockArray(tu, &compMatrix->entryColumns, compMatrix->numNonzeros);

    if (targetType == sizeof(char))
    {
      if (compMatrix->numNonzeros > 0)
        TUallocBlockArray(tu, (char**) &compMatrix->entryValues, compMatrix->numNonzeros);
      TUcreateCharMatrix(tu, (TU_CHAR_MATRIX**) &(*components)[component].transpose,
        compMatrix->numColumns, compMatrix->numRows, compMatrix->numNonzeros);
    }
    else if (targetType == sizeof(int))
    {
      if (compMatrix->numNonzeros > 0)
        TUallocBlockArray(tu, (int**) &compMatrix->entryValues, compMatrix->numNonzeros);
      TUcreateIntMatrix(tu, (TU_INT_MATRIX**) &(*components)[component].transpose,
        compMatrix->numColumns, compMatrix->numRows, compMatrix->numNonzeros);
    }
    else
    {
      assert(targetType == sizeof(double));
      if (compMatrix->numNonzeros > 0)
        TUallocBlockArray(tu, (double**) &compMatrix->entryValues, compMatrix->numNonzeros);
      TUcreateDoubleMatrix(tu, (TU_DOUBLE_MATRIX**) &(*components)[component].transpose,
        compMatrix->numColumns, compMatrix->numRows, compMatrix->numNonzeros);
    }
  }

  /* Fill mapping arrays. */
  for (int node = 0; node < numNodes; ++node)
  {
    int component = graphNodes[node].component;
    int order = graphNodes[node].order;
    if (node < firstColumnNode)
      (*components)[component].rowsToOriginal[order] = node;
    else
      (*components)[component].columnsToOriginal[order] = node - firstColumnNode;
  }

#ifdef DEBUG_ONE_SUM
  for (int comp = 0; comp < countComponents; ++comp)
  {
    printf("Component %d's rows map to original rows:", comp);
    for (int row = 0; row < (*components)[comp].matrix->numRows; ++row)
      printf(" %d", (*components)[comp].rowsToOriginal[row]);
    printf("\n");
    printf("Component %d's columns map to original columns:", comp);
    for (int column = 0; column < (*components)[comp].matrix->numColumns; ++column)
      printf(" %d", (*components)[comp].columnsToOriginal[column]);
    printf("\n");
  }
#endif

  /* We can now fill the matrices of each component. */
  for (int component = 0; component < countComponents; ++component)
  {
    TU_MATRIX* compTranspose = (*components)[component].transpose;

    /* Compute the slices in the transposed component matrix from the graph. */
    int countNonzeros = 0;
    for (int compColumn = 0; compColumn < compTranspose->numRows; ++compColumn)
    {
      int column = (*components)[component].columnsToOriginal[compColumn];
      int node = firstColumnNode + column;
      compTranspose->rowStarts[compColumn] = countNonzeros;
#ifdef DEBUG_ONE_SUM
      printf("Component %d's column %d (row of transposed) starts at component entry %d.\n",
        component, compColumn, countNonzeros);
#endif
      countNonzeros += graphNodes[node+1].adjacencyStart - graphNodes[node].adjacencyStart;
    }

    /* Fill the slices. To ensure that it is sorted, we iterate row-wise. */
    for (int compRow = 0; compRow < compTranspose->numColumns; ++compRow)
    {
      int row = (*components)[component].rowsToOriginal[compRow];
      int start = matrix->rowStarts[row];
      int end = row + 1 < matrix->numRows ? matrix->rowStarts[row + 1] : matrix->numNonzeros;

      /* Iterate over all entries of that row. */
      for (int matrixEntry = start; matrixEntry < end; ++matrixEntry)
      {
        bool nonzero;
        if (matrixType == sizeof(double))
          nonzero = round(((double*)matrix->entryValues)[matrixEntry]) != 0;
        else if (matrixType == sizeof(int))
          nonzero = ((int*)matrix->entryValues)[matrixEntry] != 0;
        else if (matrixType == sizeof(char))
          nonzero = ((char*)matrix->entryValues)[matrixEntry] != 0;
        else
          assert("Invalid matrixType parameter." == 0);
        if (nonzero)
        {
          int column = matrix->entryColumns[matrixEntry];
          int compColumn = graphNodes[firstColumnNode + column].order;
          int compEntry = compTranspose->rowStarts[compColumn];
#ifdef DEBUG_ONE_SUM
          printf("Component %d contains matrix entry %d in at %d,%d.",
            component, matrixEntry, row, column);
          printf(" It will be component entry %d at %d,%d\n", compEntry, compRow, compColumn);
#endif
          compTranspose->entryColumns[compEntry] = compRow;
          compTranspose->rowStarts[compColumn]++;
          if (targetType == sizeof(double) && matrixType == sizeof(double))
          {
            ((double*)compTranspose->entryValues)[compEntry] =
              ((double*)matrix->entryValues)[matrixEntry];
          }
          else if (targetType == sizeof(int) && matrixType == sizeof(double))
          {
            ((int*)compTranspose->entryValues)[compEntry] =
              (int)(round(((double*)matrix->entryValues)[matrixEntry]) + 0.5);
          }
          else if (targetType == sizeof(char) && matrixType == sizeof(double))
          {
            ((char*)compTranspose->entryValues)[compEntry] =
              (char)(round(((double*)matrix->entryValues)[matrixEntry]) + 0.5);
          }
          else if (targetType == sizeof(double) && matrixType == sizeof(int))
          {
            ((double*)compTranspose->entryValues)[compEntry] =
              ((int*)matrix->entryValues)[matrixEntry];
          }
          else if (targetType == sizeof(int) && matrixType == sizeof(int))
          {
            ((int*)compTranspose->entryValues)[compEntry] =
              ((int*)matrix->entryValues)[matrixEntry];
          }
          else if (targetType == sizeof(char) && matrixType == sizeof(int))
          {
            ((char*)compTranspose->entryValues)[compEntry] =
              ((int*)matrix->entryValues)[matrixEntry];
          }
          else if (targetType == sizeof(double) && matrixType == sizeof(char))
          {
            ((double*)compTranspose->entryValues)[compEntry] =
              ((char*)matrix->entryValues)[matrixEntry];
          }
          else if (targetType == sizeof(int) && matrixType == sizeof(char))
          {
            ((int*)compTranspose->entryValues)[compEntry] =
              ((char*)matrix->entryValues)[matrixEntry];
          }
          else if (targetType == sizeof(char) && matrixType == sizeof(char))
          {
            ((char*)compTranspose->entryValues)[compEntry] =
              ((char*)matrix->entryValues)[matrixEntry];
          }
          else
            assert("Invalid targetType / matrixType parameter combination." == 0);
        }
      }
    }

    /* Since we incremented the rowStarts for each nonzero, the array is shifted by one entry.
     * We restore this now. */
    for (int compColumn = compTranspose->numRows; compColumn > 0; --compColumn)
      compTranspose->rowStarts[compColumn] = compTranspose->rowStarts[compColumn-1];
    compTranspose->rowStarts[0] = 0;

#ifdef DEBUG_ONE_SUM
    printf("Component %d's transpose:\n", component);
    if (targetType == sizeof(double))
      TUprintDoubleMatrixDense(stdout, (TU_DOUBLE_MATRIX*) compTranspose, '0', true);
    else if (targetType == sizeof(int))
      TUprintIntMatrixDense(stdout, (TU_INT_MATRIX*) compTranspose, '0', true);
    else if (targetType == sizeof(char))
      TUprintCharMatrixDense(stdout, (TU_CHAR_MATRIX*) compTranspose, '0', true);
#endif
  }

  /* We now create the row-wise representation from the column-wise one. */
  for (int component = 0; component < countComponents; ++component)
  {
    TU_MATRIX* compMatrix = (*components)[component].matrix;
    TU_MATRIX* compTranspose = (*components)[component].transpose;

    /* Compute the slices in the component matrix from the graph. */
    int countNonzeros = 0;
    for (int compRow = 0; compRow < compMatrix->numRows; ++compRow)
    {
      int row = (*components)[component].rowsToOriginal[compRow];
      int node = row;
      compMatrix->rowStarts[compRow] = countNonzeros;
      countNonzeros += graphNodes[node+1].adjacencyStart - graphNodes[node].adjacencyStart;
    }

    /* Fill the slices. To ensure that it is sorted, we iterate column-wise. */
    for (int compColumn = 0; compColumn < compMatrix->numColumns; ++compColumn)
    {
      int start = compTranspose->rowStarts[compColumn];
      int end = compTranspose->rowStarts[compColumn + 1];

      /* Iterate over all entries of that column. */
      for (int compTransposeEntry = start; compTransposeEntry < end; ++compTransposeEntry)
      {
        int compRow = compTranspose->entryColumns[compTransposeEntry];
        int compMatrixEntry = compMatrix->rowStarts[compRow];
        compMatrix->entryColumns[compMatrixEntry] = compColumn;
        compMatrix->rowStarts[compRow]++;
#ifdef DEBUG_ONE_SUM
        printf("Component matrix entry %d (%d,%d in component) copied from component transpose entry %d.\n",
          compMatrixEntry, compRow, compColumn, compTransposeEntry);
#endif
        if (targetType == sizeof(double))
          ((double*)compMatrix->entryValues)[compMatrixEntry] = ((double*)compTranspose->entryValues)[compTransposeEntry];
        else if (targetType == sizeof(int))
          ((int*)compMatrix->entryValues)[compMatrixEntry] = ((int*)compTranspose->entryValues)[compTransposeEntry];
        else if (targetType == sizeof(char))
          ((char*)compMatrix->entryValues)[compMatrixEntry] = ((char*)compTranspose->entryValues)[compTransposeEntry];
        else
          assert("Invalid targetType parameter." == 0);
      }
    }

    /* Since we incremented the rowStarts for each nonzero, the array is shifted by one entry.
     * We restore this now. */
    for (int compRow = compMatrix->numRows; compRow > 0; --compRow)
      compMatrix->rowStarts[compRow] = compMatrix->rowStarts[compRow-1];
    compMatrix->rowStarts[0] = 0;

    if (targetType == sizeof(double))
    {
      assert(TUcheckDoubleMatrixTranspose((TU_DOUBLE_MATRIX*) (*components)[component].matrix,
        (TU_DOUBLE_MATRIX*) (*components)[component].transpose));
    }
    else if (targetType == sizeof(int))
    {
      assert(TUcheckIntMatrixTranspose((TU_INT_MATRIX*) (*components)[component].matrix,
        (TU_INT_MATRIX*) (*components)[component].transpose));
    }
    else if (targetType == sizeof(char))
    {
      assert(TUcheckCharMatrixTranspose((TU_CHAR_MATRIX*) (*components)[component].matrix,
        (TU_CHAR_MATRIX*) (*components)[component].transpose));
    }
  }

  /* Fill arrays for original matrix viewpoint. */
  if (rowsToComponents != NULL)
  {
    for (int row = 0; row < matrix->numRows; ++row)
      rowsToComponents[row] = graphNodes[row].component;
  }
  if (columnsToComponents != NULL)
  {
    for (int column = 0; column < matrix->numColumns; ++column)
      columnsToComponents[column] = graphNodes[firstColumnNode + column].component;
  }
  if (rowsToComponentRows != NULL)
  {
    for (int row = 0; row < matrix->numRows; ++row)
      rowsToComponentRows[row] = graphNodes[row].order;
  }
  if (columnsToComponentColumns != NULL)
  {
    for (int column = 0; column < matrix->numColumns; ++column)
      columnsToComponentColumns[column] = graphNodes[firstColumnNode + column].order;
  }

  TUfreeStackArray(tu, &queue);
  TUfreeStackArray(tu, &graphAdjacencies);
  TUfreeStackArray(tu, &graphNodes);
}
