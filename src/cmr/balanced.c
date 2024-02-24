#define CMR_DEBUG /* Uncomment to debug */

#include <cmr/balanced.h>

#include <cmr/camion.h>
#include <cmr/series_parallel.h>

#include "env_internal.h"
#include "block_decomposition.h"
#include "sort.h"
#include "matrix_internal.h"
#include "camion_internal.h"

#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <time.h>

CMR_ERROR CMRbalancedParamsInit(CMR_BALANCED_PARAMS* params)
{
  assert(params);

  params->algorithm = CMR_BALANCED_ALGORITHM_AUTO;
  params->seriesParallel = true;

  return CMR_OKAY;
}

CMR_ERROR CMRbalancedStatsInit(CMR_BALANCED_STATS* stats)
{
  assert(stats);

  CMR_CALL( CMRspStatsInit(&stats->seriesParallel) );
  stats->totalCount = 0;
  stats->totalTime = 0.0;
  stats->enumeratedColumnSubsets = 0;
  stats->enumeratedRowSubsets = 0;

  return CMR_OKAY;
}

CMR_ERROR CMRbalancedStatsPrint(FILE* stream, CMR_BALANCED_STATS* stats, const char* prefix)
{
  assert(stream);
  assert(stats);

  if (!prefix)
  {
    fprintf(stream, "Balanced matrix recognition:\n");
    prefix = "  ";
  }

  char subPrefix[256];
  snprintf(subPrefix, 256, "%sseries-parallel ", prefix);
  CMR_CALL( CMRspStatsPrint(stream, &stats->seriesParallel, subPrefix) );

  fprintf(stream, "%senumerated row subsets: %lu\n", prefix, (unsigned long)stats->enumeratedRowSubsets);
  fprintf(stream, "%senumerated column subsets: %lu\n", prefix, (unsigned long)stats->enumeratedColumnSubsets);
  fprintf(stream, "%stotal: %lu in %f seconds\n", prefix, (unsigned long)stats->totalCount, stats->totalTime);

  return CMR_OKAY;
}

/**
 * \brief Data for enumeration.
 */

typedef struct
{
  CMR* cmr;                     /**< \ref CMR environment. */
  CMR_CHRMAT* matrix;           /**< Matrix \f$ M \f$. */
  bool* pisBalanced;            /**< Pointer for storing whether \f$ M \f$ is balanced. */
  CMR_SUBMAT** psubmatrix;      /**< Pointer for storing a minimal nonbalanced submatrix (may be \c NULL). */
  CMR_BALANCED_STATS* stats;    /**< Statistics for the computation (may be \c NULL). */
  double timeLimit;             /**< Time limit to impose. */
  bool isTransposed;            /**< Whether we're dealing with the transposed matrix. */
  clock_t startClock;           /**< Clock for when we started. */
  size_t cardinality;           /**< Current cardinality of row/column subsets. */
  size_t* subsetRows;           /**< Array for the enumerated row subset. */
  size_t* usableColumns;        /**< Array of columns usable for enumeration. */
  size_t numUsableColumns;      /**< Length of usableColumns. */
  size_t* subsetUsable;         /**< Array for the enumerated subset from usable columns. */
  size_t* columnsNumNonzeros;   /**< Array with the number of nonzeros per column. */
  size_t* rowsNumNonzeros;      /**< Array with the number of nonzeros per row. */
  int sumEntries;               /**< Sum of the entries of the selected submatrix. */
} CMR_BALANCED_ENUMERATION;


/**
 * \brief Recursive enumeration of column subsets and subsequent testing of balancedness.
 */

static
CMR_ERROR balancedTestEnumerateColumns(
  CMR_BALANCED_ENUMERATION* enumeration,  /**< Enumeration information. */
  size_t numColumns                       /**< Number of already selected columns. */
)
{
  assert(enumeration);

  if (numColumns < enumeration->cardinality)
  {
    /* Recursion step: pick one column and enumerate over the other columns. */

    size_t firstUsable = (numColumns == 0) ? 0 : enumeration->subsetUsable[numColumns - 1] + 1;
    size_t beyondUsable = enumeration->numUsableColumns - enumeration->cardinality + numColumns + 1;
    for (size_t usable = firstUsable; usable < beyondUsable; ++usable)
    {
      size_t column = enumeration->usableColumns[usable];
      assert(enumeration->columnsNumNonzeros[column] == 2);

      CMRdbgMsg(12 + enumeration->cardinality + numColumns,
        "Selecting column %zu as usable %zu of %zu.\n", column, usable, enumeration->numUsableColumns);
      enumeration->subsetUsable[numColumns] = usable;

      /* Increment row nonzero counters. */
      bool tooManyNonzeros = false;
      for (size_t i = 0; i < enumeration->cardinality; ++i)
      {
        size_t row = enumeration->subsetRows[i];
        size_t entry;
        CMRchrmatFindEntry(enumeration->matrix, row, column, &entry);
        if (entry != SIZE_MAX)
        {
          enumeration->sumEntries += enumeration->matrix->entryValues[entry];
          enumeration->rowsNumNonzeros[row]++;
          if (enumeration->rowsNumNonzeros[row] > 2)
            tooManyNonzeros = true;
        }
      }

      CMRdbgMsg(12 + enumeration->cardinality + numColumns, "Some row has too many nonzeros.\n");

      if (!tooManyNonzeros)
      {
        /* Recurse. */
        CMR_CALL( balancedTestEnumerateColumns(enumeration, numColumns + 1) );
        if (!*enumeration->pisBalanced)
          return CMR_OKAY;
      }

      /* Decrement row nonzero counters. */
      for (size_t i = 0; i < enumeration->cardinality; ++i)
      {
        size_t row = enumeration->subsetRows[i];
        size_t entry;
        CMRchrmatFindEntry(enumeration->matrix, row, column, &entry);
        if (entry != SIZE_MAX)
        {
          enumeration->sumEntries -= enumeration->matrix->entryValues[entry];
          enumeration->rowsNumNonzeros[row]--;
        }
      }
    }
  }
  else
  {
    if (enumeration->stats)
    {
      if (enumeration->isTransposed)
        enumeration->stats->enumeratedRowSubsets++;
      else
        enumeration->stats->enumeratedColumnSubsets++;
    }

    CMRdbgMsg(12 + 2 * enumeration->cardinality, "Sum over nonzeros is %d.\n", enumeration->sumEntries);

    if (enumeration->sumEntries % 4 != 0)
    {
      *(enumeration->pisBalanced) = false;
      if (enumeration->psubmatrix)
      {
        CMR_SUBMAT* submatrix = NULL;
        CMR_CALL( CMRsubmatCreate(enumeration->cmr, enumeration->cardinality, enumeration->cardinality, &submatrix) );
        for (size_t i = 0; i < enumeration->cardinality; ++i)
        {
          submatrix->rows[i] = enumeration->subsetRows[i];
          submatrix->columns[i] = enumeration->usableColumns[enumeration->subsetUsable[i]];
        }

        *(enumeration->psubmatrix) = submatrix;
      }
    }
  }

  return CMR_OKAY;
}

/**
 * \brief Recursive enumeration of row subsets and subsequent testing of balancedness.
 *
 * \returns \c true if we can stop (due to time limit or a found violating submatrix).
 */

static
bool balancedTestEnumerateRows(
  CMR_BALANCED_ENUMERATION* enumeration,  /**< Enumeration information. */
  size_t numRows                          /**< Number of already selected rows. */
)
{
  assert(enumeration);

  if (numRows < enumeration->cardinality)
  {
    /* Recursion step: pick one row and enumerate over the other rows. */

    size_t firstRow = (numRows == 0) ? 0 : enumeration->subsetRows[numRows - 1] + 1;
    size_t beyondRow = enumeration->matrix->numRows - enumeration->cardinality + numRows + 1;
    for (size_t row = firstRow; row < beyondRow; ++row)
    {
      CMRdbgMsg(10 + numRows, "Selecting row %zu as row #%zu.\n", row, numRows);
      enumeration->subsetRows[numRows] = row;

      /* Iterate over chosen row and increment column nonzero counters. */
      size_t first = enumeration->matrix->rowSlice[row];
      size_t beyond = enumeration->matrix->rowSlice[row + 1];
      for (size_t e = first; e < beyond; ++e)
        enumeration->columnsNumNonzeros[enumeration->matrix->entryColumns[e]]++;

      /* Recurse. */
      CMR_CALL( balancedTestEnumerateRows(enumeration, numRows + 1) );
      if (!*(enumeration->pisBalanced) || enumeration->timeLimit <= 0)
        return CMR_OKAY;

      /* Decrement column nonzero counters again. */
      for (size_t e = first; e < beyond; ++e)
        enumeration->columnsNumNonzeros[enumeration->matrix->entryColumns[e]]--;
    }
  }
  else
  {
    if (enumeration->stats)
    {
      if (enumeration->isTransposed)
        enumeration->stats->enumeratedColumnSubsets++;
      else
        enumeration->stats->enumeratedRowSubsets++;
    }

    double remainingTime = enumeration->timeLimit - (clock() - enumeration->startClock) * 1.0 / CLOCKS_PER_SEC;
    if (remainingTime <= 0)
    {
      enumeration->timeLimit = 0;
      return CMR_OKAY;
    }

    assert(enumeration->sumEntries == 0);
    for (size_t i = 0; i < enumeration->cardinality; ++i)
      assert( enumeration->rowsNumNonzeros[enumeration->subsetRows[i]] == 0 );

    enumeration->numUsableColumns = 0;
    for (size_t column = 0; column < enumeration->matrix->numColumns; ++column)
    {
      if (enumeration->columnsNumNonzeros[column] == 2)
        enumeration->usableColumns[enumeration->numUsableColumns++] = column;
    }

    CMRdbgMsg(10 + numRows, "Number of columns with 2 nonzeros: %zu.\n", enumeration->numUsableColumns);

    /* Skip enumeraton of columns if the number of columns with 2 nonzeros is too low. */
    if (enumeration->numUsableColumns >= enumeration->cardinality)
    {
      CMR_CALL( balancedTestEnumerateColumns(enumeration, 0) );
    }
  }

  return CMR_OKAY;
}

/**
 * \brief Tests a connected series-parallel reduced matrix \f$ M \f$ for being [balanced](\ref balanced) using the
 *        enumeration algorithm.
 *
 * If \f$ M \f$ is not balanced and \p psubmatrix != \c NULL, then \p *psubmatrix will indicate a submatrix
 * of \f$ M \f$ with exactly two nonzeros in each row and in each column and with determinant \f$ -2 \f$ or \f$ 2 \f$.
 */

static
CMR_ERROR balancedTestEnumerate(
  CMR* cmr,                     /**< \ref CMR environment */
  CMR_CHRMAT* matrix,           /**< Matrix \f$ M \f$. */
  bool* pisBalanced,            /**< Pointer for storing whether \f$ M \f$ is balanced. */
  CMR_SUBMAT** psubmatrix,      /**< Pointer for storing a minimal nonbalanced submatrix (may be \c NULL). */
  CMR_BALANCED_STATS* stats,    /**< Statistics for the computation (may be \c NULL). */
  double timeLimit,             /**< Time limit to impose. */
  bool isTransposed             /**< Whether we're dealing with the transposed matrix. */
)
{
  assert(cmr);
  assert(matrix);
  assert(pisBalanced);
  assert(timeLimit > 0);

  CMRassertStackConsistency(cmr);

  CMR_ERROR error = CMR_OKAY;

  /* Transpose if more rows than columns. */
  if (matrix->numRows > matrix->numColumns)
  {
    assert(!isTransposed);

    CMR_CHRMAT* transpose = NULL;
    CMR_CALL( CMRchrmatTranspose(cmr, matrix, &transpose) );

    CMRdbgMsg(6, "Transposing matrix to have fewer rows than columns.\n");

    error = balancedTestEnumerate(cmr, transpose, pisBalanced, psubmatrix, stats, timeLimit, true);

    /* Transpose the violator. */
    if (psubmatrix && *psubmatrix)
      CMR_CALL( CMRsubmatTranspose(*psubmatrix) );

    CMR_CALL( CMRchrmatFree(cmr, &transpose) );
    return error;
  }

  CMR_BALANCED_ENUMERATION enumeration;
  enumeration.cmr = cmr;
  enumeration.matrix = matrix;
  enumeration.pisBalanced = pisBalanced;
  enumeration.psubmatrix = psubmatrix;
  enumeration.stats = stats;
  enumeration.timeLimit = timeLimit;
  enumeration.isTransposed = isTransposed;
  enumeration.startClock = clock();
  enumeration.subsetRows = NULL;
  enumeration.usableColumns = NULL;
  enumeration.subsetUsable = NULL;
  enumeration.rowsNumNonzeros = NULL;
  enumeration.columnsNumNonzeros = NULL;
  enumeration.sumEntries = 0;

  CMR_CALL( CMRallocStackArray(cmr, &enumeration.subsetRows, matrix->numRows) );
  CMR_CALL( CMRallocStackArray(cmr, &enumeration.usableColumns, matrix->numColumns) );
  CMR_CALL( CMRallocStackArray(cmr, &enumeration.subsetUsable, matrix->numColumns) );
  CMR_CALL( CMRallocStackArray(cmr, &enumeration.rowsNumNonzeros, matrix->numRows) );
  CMR_CALL( CMRallocStackArray(cmr, &enumeration.columnsNumNonzeros, matrix->numColumns) );
  for (size_t row = 0; row < matrix->numRows; ++row)
    enumeration.rowsNumNonzeros[row] = 0;
  for (size_t column = 0; column < matrix->numColumns; ++column)
    enumeration.columnsNumNonzeros[column] = 0;

  CMRdbgMsg(6, "Starting enumeration algorithm with a time limit of %g.\n", timeLimit);

  CMRassertStackConsistency(cmr);

  for (enumeration.cardinality = 2; enumeration.cardinality <= matrix->numRows; enumeration.cardinality++)
  {
    CMRassertStackConsistency(cmr);
    CMRdbgMsg(8, "Considering submatrices of size %zux%zu.\n", enumeration.cardinality, enumeration.cardinality);
    CMRassertStackConsistency(cmr);
    CMR_CALL( balancedTestEnumerateRows(&enumeration, 0) );
    if (!*pisBalanced || enumeration.timeLimit <= 0)
      break;
  }

  CMRassertStackConsistency(cmr);

  CMR_CALL( CMRfreeStackArray(cmr, &enumeration.columnsNumNonzeros) );
  CMR_CALL( CMRfreeStackArray(cmr, &enumeration.rowsNumNonzeros) );
  CMR_CALL( CMRfreeStackArray(cmr, &enumeration.subsetUsable) );
  CMR_CALL( CMRfreeStackArray(cmr, &enumeration.usableColumns) );
  CMR_CALL( CMRfreeStackArray(cmr, &enumeration.subsetRows) );

  CMRassertStackConsistency(cmr);

  if (enumeration.timeLimit <= 0)
    return CMR_ERROR_TIMEOUT;

  return CMR_OKAY;
}

static
CMR_ERROR balancedBFS(
  CMR* cmr,                     /**< \ref CMR environment */
  CMR_CHRMAT* matrix,           /**< Matrix \f$ M \f$. */
  CMR_CHRMAT* transpose,
  CMR_ELEMENT source,
  CMR_ELEMENT target,
  bool* rowMask,
  bool* colMask,
  bool* rowsInPath,
  bool* colsInPath
)
{
  // Ensure that the source element is valid for this matrix.
  if (CMRelementIsRow(source)) {
    assert(CMRelementToRowIndex(source) < matrix->numRows);
  }
  else if (CMRelementIsColumn(source)){
    assert(CMRelementToColumnIndex(source) < matrix->numColumns);
  }
  // Ensure that the target element is valid for this matrix.
  if (CMRelementIsRow(target)) {
    assert(CMRelementToRowIndex(target) < matrix->numRows);
  }
  else if (CMRelementIsColumn(target)){
    assert(CMRelementToColumnIndex(target) < matrix->numColumns);
  }

  CMR_ELEMENT* queue = NULL;
  CMR_CALL( CMRallocStackArray(cmr, &queue, matrix->numRows + matrix->numColumns));
  queue[0] = source;
  size_t current = 0;
  size_t potential = 1;
  CMR_ELEMENT* backtrack = NULL; /**< Array that stores the CMR_ELEMENT to get back to the source. */
  CMR_CALL( CMRallocStackArray(cmr, &backtrack, matrix->numRows + matrix->numColumns));
  // Initialise the backtrack array, setting all but the source to NULL. The source will backtrack to itself for now.
  for (int i = 0; i < matrix->numRows + matrix->numColumns; i++) {
    backtrack[i] = NULL;
  }
  if (CMRelementIsRow(source)) {
    backtrack[CMRrowToElement(source)] = source;
  } else if (CMRelementIsColumn(source)) {
    backtrack[matrix->numRows + CMRcolumnToElement(source)] = source;
  }
  bool isTargetFound = false;

  while (current < potential && !isTargetFound)
  {

    CMRdbgMsg(1, "queue:     ");
    for (int i = 0; i < potential; i++) {
      CMRdbgMsg(0, "%s ", CMRelementString(queue[i], NULL));
    }
    CMRdbgMsg(0, "\n current:   ");
    for (int i = 0; i < current; i++) {
      CMRdbgMsg(0, "   ");
    }
    CMRdbgMsg(0, "^^\n");


    CMR_ELEMENT currentElement = queue[current];
    if (CMRelementIsRow(currentElement))
    {
      size_t currentRow = CMRelementToRowIndex(currentElement);
      // Loop over the columns that the currentRow is adjacent to.
      for (int i = matrix->rowSlice[currentRow]; i < matrix->rowSlice[currentRow + 1]; i++)
      {
        size_t col = matrix->entryColumns[i];
        if (colMask[col] && (backtrack[matrix->numRows + col] == NULL))
        {
          CMR_ELEMENT foundElement = CMRcolumnToElement(col);
          queue[potential] = foundElement;
          potential++;
          backtrack[matrix->numRows + col] = currentElement;

          if (foundElement == target)
          {
            isTargetFound = true;
          }
          
          CMRdbgMsg(2, "%s->", CMRelementString(currentElement, NULL));
          CMRdbgMsg(0, "%s\n", CMRelementString(foundElement, NULL));
        }
      }
    }
    else if (CMRelementIsColumn(currentElement))
    {
      size_t currentCol = CMRelementToColumnIndex(currentElement);
      // Loop over the rows that the currentCol is adjacent to.
      for (int i = transpose->rowSlice[currentCol]; i < transpose->rowSlice[currentCol + 1]; i++)
      {
        size_t row = transpose->entryColumns[i];
        if (rowMask[row] && (backtrack[row] == NULL))
        {
          CMR_ELEMENT foundElement = CMRrowToElement(row);
          queue[potential] = foundElement;
          potential++;
          backtrack[row] = currentElement;

          if (foundElement == target)
          {
            isTargetFound = true;
          }

          CMRdbgMsg(2, "%s->", CMRelementString(currentElement, NULL));
          CMRdbgMsg(0, "%s\n", CMRelementString(foundElement, NULL));
        }
      }
    }
    current++;


    CMRdbgMsg(1, "backtrack: ");
    for (int i = 0; i < matrix->numRows + matrix->numColumns; i++) {
      CMRdbgMsg(0, "%s ", backtrack[i] != NULL ? CMRelementString(backtrack[i], NULL) : "--");
    }
    CMRdbgMsg(0, "\n");
  }

  CMRdbgMsg(0, "target %s found!\n", CMRelementString(target, NULL));
  CMRdbgMsg(0, "path from target to source: %s ", CMRelementString(target, NULL));

  // Backtrack from the target to the source. 
  // Each element on the path is stored in the rowsInPath and colsInPath arrays, including the target and source.
  CMR_ELEMENT backtrackElement = target;
  while (backtrackElement != source) 
  {
    if (CMRelementIsRow(backtrackElement))
    {
      rowsInPath[CMRelementToRowIndex(backtrackElement)] = true;
      backtrackElement = backtrack[CMRelementToRowIndex(backtrackElement)];
    } 
    else if (CMRelementIsColumn(backtrackElement))
    {
      colsInPath[CMRelementToColumnIndex(backtrackElement)] = true;
      backtrackElement = backtrack[matrix->numRows + CMRelementToColumnIndex(backtrackElement)];
    }
    CMRdbgMsg(0, "%s ", CMRelementString(backtrackElement, NULL));
  }
  // Also store the source in the path arrays.
  if (CMRelementIsRow(backtrackElement))
  {
    rowsInPath[CMRelementToRowIndex(backtrackElement)] = true;
  } 
  else if (CMRelementIsColumn(backtrackElement))
  {
    colsInPath[CMRelementToColumnIndex(backtrackElement)] = true;
  }
  CMRdbgMsg(0, "\n");


  CMR_CALL( CMRfreeStackArray(cmr, &backtrack));
  CMR_CALL( CMRfreeStackArray(cmr, &queue));
  
  return CMR_OKAY;
}

/**
 * \brief Tests a connected series-parallel reduced matrix \f$ M \f$ for being [balanced](\ref balanced) using the
 *        graph-based algorithm.
 *
 * If \f$ M \f$ is not balanced and \p psubmatrix != \c NULL, then \p *psubmatrix will indicate a submatrix
 * of \f$ M \f$ with exactly two nonzeros in each row and in each column and with determinant \f$ -2 \f$ or \f$ 2 \f$.
 */

static
CMR_ERROR balancedTestGraph(
  CMR* cmr,                     /**< \ref CMR environment */
  CMR_CHRMAT* matrix,           /**< Matrix \f$ M \f$. */
  bool* pisBalanced,            /**< Pointer for storing whether \f$ M \f$ is balanced. */
  CMR_SUBMAT** psubmatrix,      /**< Pointer for storing a minimal nonbalanced submatrix (may be \c NULL). */
  CMR_BALANCED_PARAMS* params,  /**< Parameters for the computation. */
  CMR_BALANCED_STATS* stats,    /**< Statistics for the computation (may be \c NULL). */
  double timeLimit              /**< Time limit to impose. */
)
{
  assert(cmr);
  assert(matrix);
  assert(pisBalanced);
  assert(psubmatrix);
  assert(params);
  assert(stats);
  assert(timeLimit > 0);

  assert(!"Not implemented");

  CMR_CHRMAT* transpose = NULL;
  CMRchrmatTranspose(cmr, matrix, &transpose); // also handles the memory allocation for transpose.
  
  CMRdbgMsg(0, "\n");
  CMRdbgMsg(0, "matrix: \n");
  CMRchrmatPrintDense(cmr, matrix, stdout, '0', true);
  CMRdbgMsg(0, "transpose matrix: \n");
  CMRchrmatPrintDense(cmr, transpose, stdout, '0', true);

  CMR_ELEMENT source = NULL;
  CMR_ELEMENT target = NULL;
  // For now use manual input for testing the BFS on the above matrix.
  char inputSource[100];
  char inputTarget[100];
  printf("Enter source element: ");
  scanf("%s", inputSource);
  if (inputSource[0] == 'r') {
    source = CMRrowToElement(atoi(&inputSource[1])-1);
  } else if (inputSource[0] == 'c') {
    source = CMRcolumnToElement(atoi(&inputSource[1])-1);
  } else {
    printf("INVALID INPUT\n");
  }

  printf("Enter target element: ");
  scanf("%s", inputTarget);
  if (inputTarget[0] == 'r') {
    target = CMRrowToElement(atoi(&inputTarget[1])-1);
  } else if (inputTarget[0] == 'c') {
    target = CMRcolumnToElement(atoi(&inputTarget[1])-1);
  } else {
    printf("INVALID INPUT\n");
  }

  
  CMRdbgMsg(0, "source: %s\n", CMRelementString(source, NULL));
  CMRdbgMsg(0, "target: %s\n", CMRelementString(target, NULL));

  bool* rowMask = NULL; /**< boolArray for the mask of the rows allowed to be used in the BFS. */
  bool* colMask = NULL; /**< boolArray for the mask of the cols allowed to be used in the BFS. */
  bool* rowsInPath = NULL; /**< boolArray for the mask of the rows that are in the path found by the BFS. */
  bool* colsInPath = NULL; /**< boolArray for the mask of the cols that are in the path found by the BFS. */
  // CMRdbgMsg(0, "size of bool: %s\n", sizeof(bool)); // what is the size of bool? does alloc not allocate way too much for this boolArray?
  CMR_CALL( CMRallocStackArray(cmr, &rowMask, matrix->numRows) );
  CMR_CALL( CMRallocStackArray(cmr, &colMask, matrix->numColumns) );
  CMR_CALL( CMRallocStackArray(cmr, &rowsInPath, matrix->numRows) );
  CMR_CALL( CMRallocStackArray(cmr, &colsInPath, matrix->numColumns) );
  // Initialize the rowMask and colMask to be all true.
  for (int i = 0; i < matrix->numRows; i++)
  {
    rowMask[i] = true;
    rowsInPath[i] = false;
    // CMRdbgMsg(2, "%s %d %s\n", CMRelementString(CMRrowToElement(i), NULL), rowMask[i], rowMask[i] ? "true" : "false");
  }
  for (int i = 0; i < matrix->numColumns; i++)
  {
    colMask[i] = true;
    colsInPath[i] = false;
    // CMRdbgMsg(2, "%s %d %s\n", CMRelementString(CMRcolToElement(i), NULL), colMask[i], colMask[i] ? "true" : "false");
  }


  balancedBFS(cmr, matrix, transpose, source, target, rowMask, colMask, rowsInPath, colsInPath);

  CMRdbgMsg(0, "\n");
  CMRdbgMsg(0, "matrix: \n");
  CMRchrmatPrintDense(cmr, matrix, stdout, '0', true);
  CMRdbgMsg(0, "Results of BFS:\n");
  for (int row = 0; row < matrix->numRows; row++)
  {
    CMRdbgMsg(1, "%s ", CMRelementString(CMRrowToElement(row), NULL));
    CMRdbgMsg(0, "%s\n", rowsInPath[row] ? "true" : "false");
  }
  for (int col = 0; col < matrix->numColumns; col++)
  {
    CMRdbgMsg(1, "%s ", CMRelementString(CMRcolumnToElement(col), NULL));
    CMRdbgMsg(0, "%s\n", colsInPath[col] ? "true" : "false");
  }

  CMR_CALL( CMRfreeStackArray(cmr, &colsInPath));
  CMR_CALL( CMRfreeStackArray(cmr, &rowsInPath));
  CMR_CALL( CMRfreeStackArray(cmr, &colMask));
  CMR_CALL( CMRfreeStackArray(cmr, &rowMask));

  CMR_CALL( CMRchrmatFree(cmr, &transpose) );
  

  return CMR_OKAY;
}

/**
 * \brief Tests a connected (potentially SP-reduced) matrix \f$ M \f$ for being [balanced](\ref balanced).
 *
 * Calls the algorithm requested by \p params.
 *
 * If \f$ M \f$ is not balanced and \p psubmatrix != \c NULL, then \p *psubmatrix will indicate a submatrix
 * of \f$ M \f$ with exactly two nonzeros in each row and in each column and with determinant \f$ -2 \f$ or \f$ 2 \f$.
 */

static
CMR_ERROR balancedTestChooseAlgorithm(
  CMR* cmr,                     /**< \ref CMR environment */
  CMR_CHRMAT* matrix,           /**< Matrix \f$ M \f$. */
  bool* pisBalanced,            /**< Pointer for storing whether \f$ M \f$ is balanced. */
  CMR_SUBMAT** psubmatrix,      /**< Pointer for storing a minimal nonbalanced submatrix (may be \c NULL). */
  CMR_BALANCED_PARAMS* params,  /**< Parameters for the computation. */
  CMR_BALANCED_STATS* stats,    /**< Statistics for the computation (may be \c NULL). */
  double timeLimit              /**< Time limit to impose. */
)
{
  assert(cmr);
  assert(matrix);

  CMR_ERROR error = CMR_OKAY;

  CMR_BALANCED_ALGORITHM algorithm = params->algorithm;
  if (algorithm == CMR_BALANCED_ALGORITHM_AUTO)
  {
    /* TODO: Conclude experiments with a recommendation for when to use which algorithm. */
    // algorithm = CMR_BALANCED_ALGORITHM_SUBMATRIX;

    /* FOR TESTING PURPOSES AUTO NOW USES GRAPH */
    algorithm = CMR_BALANCED_ALGORITHM_GRAPH;
  }

  if (algorithm == CMR_BALANCED_ALGORITHM_GRAPH)
  {
    error = balancedTestGraph(cmr, matrix, pisBalanced, psubmatrix, params, stats, timeLimit);
    if (error != CMR_ERROR_TIMEOUT)
      CMR_CALL(error);

    // return CMR_ERROR_INVALID;
  }
  else
  {
    assert(algorithm == CMR_BALANCED_ALGORITHM_SUBMATRIX);
    error = balancedTestEnumerate(cmr, matrix, pisBalanced, psubmatrix, stats, timeLimit, false);
    if (error != CMR_ERROR_TIMEOUT)
      CMR_CALL(error);

  }

  return error;
}


/**
 * \brief Tests a connected matrix \f$ M \f$ for being [balanced](\ref balanced).
 *
 * Tries to apply series-parallel reductions and then calls \ref balancedTestSeriesParalelReduced.
 *
 * If \f$ M \f$ is not balanced and \p psubmatrix != \c NULL, then \p *psubmatrix will indicate a submatrix
 * of \f$ M \f$ with exactly two nonzeros in each row and in each column and with determinant \f$ -2 \f$ or \f$ 2 \f$.
 */

static
CMR_ERROR balancedTestConnected(
  CMR* cmr,                     /**< \ref CMR environment */
  CMR_CHRMAT* matrix,           /**< Matrix \f$ M \f$. */
  bool* pisBalanced,            /**< Pointer for storing whether \f$ M \f$ is balanced. */
  CMR_SUBMAT** psubmatrix,      /**< Pointer for storing a minimal nonbalanced submatrix (may be \c NULL). */
  CMR_BALANCED_PARAMS* params,  /**< Parameters for the computation. */
  CMR_BALANCED_STATS* stats,    /**< Statistics for the computation (may be \c NULL). */
  double timeLimit              /**< Time limit to impose. */
)
{
  assert(cmr);
  assert(matrix);
  assert(params);

  clock_t startClock = clock();

  CMR_ERROR error = CMR_OKAY;
  bool isSeriesParallel = true;
  CMR_SUBMAT* reducedSubmatrix = NULL;
  CMR_SUBMAT* violatorSubmatrix = NULL;

  CMRdbgMsg(2, "Called balancedTestConnected for a %zux%zu matrix.\n", matrix->numRows, matrix->numColumns);

  if (params->seriesParallel)
  {
    /* TODO: Consider 2-separations as well; to this end, use CMRdecomposeTernarySeriesParallel instead. */

    CMR_CALL( CMRtestTernarySeriesParallel(cmr, matrix, &isSeriesParallel, NULL, NULL, &reducedSubmatrix,
      &violatorSubmatrix, stats ? &stats->seriesParallel : NULL, timeLimit) );

    /* Stop if the matrix is actually series-parallel. */
    if (isSeriesParallel)
    {
      CMRdbgMsg(4, "Matrix is series-parallel.\n");
      *pisBalanced = true;
      CMR_CALL( CMRsubmatFree(cmr, &reducedSubmatrix) );
      return CMR_OKAY;
    }

    /* Check if violator matrix is unbalanced. */
    bool* violatorSubmatrixColumns = NULL;
    size_t* violatorSubmatrixColumnsNonzeros = NULL;
    CMR_CALL( CMRallocStackArray(cmr, &violatorSubmatrixColumns, matrix->numColumns) );
    CMR_CALL( CMRallocStackArray(cmr, &violatorSubmatrixColumnsNonzeros, matrix->numColumns) );
    for (size_t column = 0; column < matrix->numColumns; ++column)
    {
      violatorSubmatrixColumns[column] = false;
      violatorSubmatrixColumnsNonzeros[column] = 0;
    }
    for (size_t j = 0; j < violatorSubmatrix->numColumns; ++j)
      violatorSubmatrixColumns[violatorSubmatrix->columns[j]] = true;

    int sum = 0;
    bool isEulerian = true;
    for (size_t i = 0; i < violatorSubmatrix->numRows; ++i)
    {
      size_t row = violatorSubmatrix->rows[i];
      size_t begin = matrix->rowSlice[row];
      size_t beyond = matrix->rowSlice[row + 1];
      for (size_t e = begin; e < beyond; ++e)
      {
        size_t column = matrix->entryColumns[e];
        if (violatorSubmatrixColumns[column])
        {
          sum += matrix->entryValues[e];
          violatorSubmatrixColumnsNonzeros[column]++;
        }
      }
    }
    for (size_t j = 0; j < violatorSubmatrix->numColumns; ++j)
    {
      if (violatorSubmatrixColumnsNonzeros[violatorSubmatrix->columns[j]] != 2)
      {
        isEulerian = false;
        break;
      }
    }

    CMR_CALL( CMRfreeStackArray(cmr, &violatorSubmatrixColumnsNonzeros) );
    CMR_CALL( CMRfreeStackArray(cmr, &violatorSubmatrixColumns) );

    if (isEulerian && sum % 4 != 0)
    {
      CMRdbgMsg(4, "Matrix is not series-parallel, and the violating submatrix is unbalanced.\n");
      *pisBalanced = false;
      if (psubmatrix)
        *psubmatrix = violatorSubmatrix;
      else
        CMR_CALL( CMRsubmatFree(cmr, &violatorSubmatrix) );
      CMR_CALL( CMRsubmatFree(cmr, &reducedSubmatrix) );
      return CMR_OKAY;
    }
    CMR_CALL( CMRsubmatFree(cmr, &violatorSubmatrix) );

    CMRdbgMsg(4, "Matrix is not series-parallel, and the reduced matrix is %zux%zu.\n", reducedSubmatrix->numRows,
    reducedSubmatrix->numColumns);

    /* If we were not lucky, then we need to deal with the SP-reduced submatrix. */

    CMR_CHRMAT* reducedMatrix = NULL;
    CMR_CALL( CMRchrmatZoomSubmat(cmr, matrix, reducedSubmatrix, &reducedMatrix) );

    double time = (clock() - startClock) * 1.0 / CLOCKS_PER_SEC;
    if (time >= timeLimit)
    {
      CMR_CALL( CMRchrmatFree(cmr, &reducedMatrix) );
      CMR_CALL( CMRsubmatFree(cmr, &reducedSubmatrix) );
      return CMR_ERROR_TIMEOUT;
    }

    CMR_SUBMAT* submatrix = NULL;
    error = balancedTestChooseAlgorithm(cmr, reducedMatrix, pisBalanced, psubmatrix ? &submatrix : NULL, params,
      stats, timeLimit - time);
    if (error != CMR_ERROR_TIMEOUT)
      CMR_CALL(error);

    CMRdbgMsg(4, "Matrix %s balanced.\n", (*pisBalanced) ? "IS" : "is NOT" );

    /* Turn the submatrix of the reduced matrix into a submatrix of the input matrix. */
    if (submatrix)
    {
      CMRsubmatZoomSubmat(cmr, reducedSubmatrix, submatrix, psubmatrix);
      CMR_CALL( CMRsubmatFree(cmr, &submatrix) );
    }

    CMR_CALL( CMRchrmatFree(cmr, &reducedMatrix) );
    CMR_CALL( CMRsubmatFree(cmr, &reducedSubmatrix) );
  }
  else
  {
    error = balancedTestChooseAlgorithm(cmr, matrix, pisBalanced, psubmatrix, params, stats, timeLimit);

    CMRdbgMsg(4, "Matrix %s balanced.\n", (*pisBalanced) ? "IS" : "is NOT" );

    if (error != CMR_ERROR_TIMEOUT)
      CMR_CALL(error);
  }

  return error;
}

/**
 * \brief Compares two matrix blocks according to min(#rows, #columns).
 */

int compareBlockComponents(const void* a, const void* b)
{
  CMR_BLOCK* ca = (CMR_BLOCK*)a;
  CMR_BLOCK* cb = (CMR_BLOCK*)b;
  int min_a = (int)(ca->matrix->numRows < ca->matrix->numColumns ? ca->matrix->numRows : ca->matrix->numColumns);
  int min_b = (int)(cb->matrix->numRows < cb->matrix->numColumns ? cb->matrix->numRows : cb->matrix->numColumns);
  return min_a - min_b;
}

CMR_ERROR CMRbalancedTest(CMR* cmr, CMR_CHRMAT* matrix, bool* pisBalanced, CMR_SUBMAT** psubmatrix,
  CMR_BALANCED_PARAMS* params, CMR_BALANCED_STATS* stats, double timeLimit)
{
  assert(cmr);
  assert(matrix);

  /* Initialize default params if not passed. */
  CMR_BALANCED_PARAMS localParams;
  if (!params)
  {
    CMR_CALL( CMRbalancedParamsInit(&localParams) );
    params = &localParams;
  }

  CMRdbgMsg(0, "Called CMRbalancedTest for a %zux%zu matrix.\n", matrix->numRows, matrix->numColumns);

  clock_t startClock = clock();

  if (!CMRchrmatIsTernary(cmr, matrix, psubmatrix))
  {
    if (stats)
    {
      stats->totalCount++;
      stats->totalTime += (clock() - startClock) * 1.0 / CLOCKS_PER_SEC;
    }
    return CMR_OKAY;
  }

  /* Perform a block decomposition. */

  size_t numComponents;
  CMR_BLOCK* components = NULL;
  CMR_CALL( CMRdecomposeBlocks(cmr, (CMR_MATRIX*) matrix, sizeof(char), sizeof(char), &numComponents, &components, NULL,
    NULL, NULL, NULL) );

  CMRdbgMsg(2, "Found %zu blocks.\n", numComponents);

  /* We create an intermediate array for sorting the components by number of nonzeros. */

  CMR_BLOCK** orderedComponents = NULL;
  CMR_CALL( CMRallocStackArray(cmr, &orderedComponents, numComponents) );
  for (size_t comp = 0; comp < numComponents; ++comp)
    orderedComponents[comp] = &components[comp];
  CMR_CALL( CMRsort(cmr, numComponents, orderedComponents, sizeof(CMR_BLOCK*), &compareBlockComponents) );

  *pisBalanced = true;
  for (size_t comp = 0; comp < numComponents; ++comp)
  {
    CMR_BLOCK* component = orderedComponents[comp];
    CMR_CHRMAT* matrix = (CMR_CHRMAT*) component->matrix;
    CMR_CHRMAT* transpose = (CMR_CHRMAT*) component->transpose;

    CMRdbgMsg(2, "Processing block %zu.\n", comp);

    double time = ((clock() - startClock) * 1.0 / CLOCKS_PER_SEC);
    if (*pisBalanced && time < timeLimit)
    {
      CMR_CALL( balancedTestConnected(cmr, matrix, pisBalanced, psubmatrix, params, stats, timeLimit - time) );

      /* If the component was not balanced, then we modify its violating submatrix to be one of the input matrix. */
      if (!*pisBalanced && psubmatrix)
      {
        CMR_SUBMAT* submatrix = *psubmatrix;
        for (size_t row = 0; row < submatrix->numRows; ++row)
          submatrix->rows[row] = component->rowsToOriginal[submatrix->rows[row]];
        for (size_t column = 0; column < submatrix->numColumns; ++column)
          submatrix->columns[column] = component->columnsToOriginal[submatrix->columns[column]];
      }
    }

    CMR_CALL( CMRchrmatFree(cmr, &matrix) );
    CMR_CALL( CMRchrmatFree(cmr, &transpose) );
    CMR_CALL( CMRfreeBlockArray(cmr, &component->rowsToOriginal) );
    CMR_CALL( CMRfreeBlockArray(cmr, &component->columnsToOriginal) );
  }

  CMR_CALL( CMRfreeStackArray(cmr, &orderedComponents) );
  CMR_CALL( CMRfreeBlockArray(cmr, &components) );

  double time = ((clock() - startClock) * 1.0 / CLOCKS_PER_SEC);
  if (stats)
  {
    stats->totalCount++;
    stats->totalTime += time;
  }

  return time < timeLimit ? CMR_OKAY : CMR_ERROR_TIMEOUT;
}

