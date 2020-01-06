#include <tu/matrix.h>

#include <assert.h>
#include <stdlib.h>
#include <math.h>

void TUcreateDoubleMatrix(TU* tu, TU_DOUBLE_MATRIX** matrix, int numRows, int numColumns, 
  int numNonzeros)
{
  assert(matrix);
  TUallocBlock(tu, matrix);
  (*matrix)->numRows = numRows;
  (*matrix)->numColumns = numColumns;
  (*matrix)->numNonzeros = numNonzeros;
  TUallocBlockArray(tu, &(*matrix)->rowStarts, numRows);
  if (numNonzeros > 0)
  {
    TUallocBlockArray(tu, &(*matrix)->entryColumns, numNonzeros);
    TUallocBlockArray(tu, &(*matrix)->entryValues, numNonzeros);
  }
  else
  {
    (*matrix)->entryColumns = NULL;
    (*matrix)->entryValues = NULL;
  }
}

void TUfreeDoubleMatrix(TU* tu, TU_DOUBLE_MATRIX** matrix)
{
  assert(matrix);

  TUfreeBlock(tu, matrix);
  if ((*matrix)->rowStarts)
    TUfreeBlockArray(tu, &(*matrix)->rowStarts);
  if ((*matrix)->numNonzeros > 0)
  {
    TUfreeBlockArray(tu, &(*matrix)->entryColumns);
    TUfreeBlockArray(tu, &(*matrix)->entryValues);
  }
}

void TUcreateIntMatrix(TU* tu, TU_INT_MATRIX** matrix, int numRows, int numColumns, 
  int numNonzeros)
{
  assert(matrix);
  TUallocBlock(tu, matrix);
  (*matrix)->numRows = numRows;
  (*matrix)->numColumns = numColumns;
  (*matrix)->numNonzeros = numNonzeros;
  TUallocBlockArray(tu, &(*matrix)->rowStarts, numRows);
  if (numNonzeros > 0)
  {
    TUallocBlockArray(tu, &(*matrix)->entryColumns, numNonzeros);
    TUallocBlockArray(tu, &(*matrix)->entryValues, numNonzeros);
  }
  else
  {
    (*matrix)->entryColumns = NULL;
    (*matrix)->entryValues = NULL;
  }
}

void TUfreeIntMatrix(TU* tu, TU_INT_MATRIX** matrix)
{
  assert(matrix);

  TUfreeBlock(tu, matrix);
  if ((*matrix)->rowStarts)
    TUfreeBlockArray(tu, &(*matrix)->rowStarts);
  if ((*matrix)->numNonzeros > 0)
  {
    TUfreeBlockArray(tu, &(*matrix)->entryColumns);
    TUfreeBlockArray(tu, &(*matrix)->entryValues);
  }
}

void TUcreateCharMatrix(TU* tu, TU_CHAR_MATRIX** matrix, int numRows, int numColumns, 
  int numNonzeros)
{
  assert(matrix);
  TUallocBlock(tu, matrix);
  (*matrix)->numRows = numRows;
  (*matrix)->numColumns = numColumns;
  (*matrix)->numNonzeros = numNonzeros;
  TUallocBlockArray(tu, &(*matrix)->rowStarts, numRows);
  if (numNonzeros > 0)
  {
    TUallocBlockArray(tu, &(*matrix)->entryColumns, numNonzeros);
    TUallocBlockArray(tu, &(*matrix)->entryValues, numNonzeros);
  }
  else
  {
    (*matrix)->entryColumns = NULL;
    (*matrix)->entryValues = NULL;
  }
}

void TUfreeCharMatrix(TU* tu, TU_CHAR_MATRIX** matrix)
{
  assert(matrix);

  TUfreeBlock(tu, matrix);
  if ((*matrix)->rowStarts)
    TUfreeBlockArray(tu, &(*matrix)->rowStarts);
  if ((*matrix)->numNonzeros > 0)
  {
    TUfreeBlockArray(tu, &(*matrix)->entryColumns);
    TUfreeBlockArray(tu, &(*matrix)->entryValues);
  }
}

void TUprintDoubleMatrixDense(FILE* stream, TU_DOUBLE_MATRIX* sparse, char zeroChar, bool header)
{
  assert(stream != NULL);
  assert(sparse != NULL);
  double* rowEntries = (double*) calloc(sparse->numColumns, sizeof(double));

  fprintf(stream, "%d %d\n", sparse->numRows, sparse->numColumns);
  if (header)
  {
    fputs("   ", stream);
    for (int column = 0; column < sparse->numColumns; ++column)
      fprintf(stream, "%d ", column % 10);
    fputs("\n  ", stream);
    for (int column = 0; column < sparse->numColumns; ++column)
      fputs("--", stream);
    fputc('\n', stream);
  }
  for (int row = 0; row < sparse->numRows; ++row)
  {
    if (header)
      fprintf(stream, "%d| ", row % 10);
    int start = sparse->rowStarts[row];
    int end = row + 1 < sparse->numRows ? sparse->rowStarts[row + 1] : sparse->numNonzeros;
    for (int i = start; i < end; ++i)
      rowEntries[sparse->entryColumns[i]] = sparse->entryValues[i];
    for (int column = 0; column < sparse->numColumns; ++column)
    {
      double x = rowEntries[column];
      if (x == 0.0)
        fprintf(stream, "%c ", zeroChar);
      else
        fprintf(stream, "%f ", x);
    }
    for (int i = start; i < end; ++i)
      rowEntries[sparse->entryColumns[i]] = 0.0;
    fputc('\n', stream);
  }

  free(rowEntries);
}

void TUprintIntMatrixDense(FILE* stream, TU_INT_MATRIX* sparse, char zeroChar, bool header)
{
  assert(stream != NULL);
  assert(sparse != NULL);
  int* rowEntries = (int*) calloc(sparse->numColumns, sizeof(int));

  fprintf(stream, "%d %d\n", sparse->numRows, sparse->numColumns);
  if (header)
  {
    fputs("   ", stream);
    for (int column = 0; column < sparse->numColumns; ++column)
      fprintf(stream, "%d ", column % 10);
    fputs("\n  ", stream);
    for (int column = 0; column < sparse->numColumns; ++column)
      fputs("--", stream);
    fputc('\n', stream);
  }
  for (int row = 0; row < sparse->numRows; ++row)
  {
    if (header)
      fprintf(stream, "%d| ", row % 10);
    int start = sparse->rowStarts[row];
    int end = row + 1 < sparse->numRows ? sparse->rowStarts[row + 1] : sparse->numNonzeros;
    for (int i = start; i < end; ++i)
      rowEntries[sparse->entryColumns[i]] = sparse->entryValues[i];
    for (int column = 0; column < sparse->numColumns; ++column)
    {
      int x = rowEntries[column];
      if (x == 0.0)
        fprintf(stream, "%c ", zeroChar);
      else
        fprintf(stream, "%d ", x);
    }
    for (int i = start; i < end; ++i)
      rowEntries[sparse->entryColumns[i]] = 0.0;
    fputc('\n', stream);
  }

  free(rowEntries);
}

void TUprintCharMatrixDense(FILE* stream, TU_CHAR_MATRIX* sparse, char zeroChar, bool header)
{
  assert(stream != NULL);
  assert(sparse != NULL);
  char* rowEntries = (char*) calloc(sparse->numColumns, sizeof(char));

  fprintf(stream, "%d %d\n", sparse->numRows, sparse->numColumns);
  if (header)
  {
    fputs("   ", stream);
    for (int column = 0; column < sparse->numColumns; ++column)
      fprintf(stream, "%d ", column % 10);
    fputs("\n  ", stream);
    for (int column = 0; column < sparse->numColumns; ++column)
      fputs("--", stream);
    fputc('\n', stream);
  }
  for (int row = 0; row < sparse->numRows; ++row)
  {
    if (header)
      fprintf(stream, "%d| ", row % 10);
    int start = sparse->rowStarts[row];
    int end = row + 1 < sparse->numRows ? sparse->rowStarts[row + 1] : sparse->numNonzeros;
    for (int i = start; i < end; ++i)
      rowEntries[sparse->entryColumns[i]] = sparse->entryValues[i];
    for (int column = 0; column < sparse->numColumns; ++column)
    {
      char x = rowEntries[column];
      if (x == 0.0)
        fprintf(stream, "%c ", zeroChar);
      else
        fprintf(stream, "%d ", x);
    }
    for (int i = start; i < end; ++i)
      rowEntries[sparse->entryColumns[i]] = 0.0;
    fputc('\n', stream);
  }

  free(rowEntries);
}

bool TUcheckDoubleMatrixEqual(TU_DOUBLE_MATRIX* matrix1, TU_DOUBLE_MATRIX* matrix2)
{
  assert(TUcheckDoubleMatrixSorted(matrix1));
  assert(TUcheckDoubleMatrixSorted(matrix2));

  if (matrix1->numRows != matrix2->numRows)
    return false;
  if (matrix1->numColumns != matrix2->numColumns)
    return false;
  if (matrix1->numColumns != matrix2->numColumns)
    return false;

  for (int row = 0; row < matrix1->numRows; ++row)
  {
    int start1 = matrix1->rowStarts[row];
    int start2 = matrix2->rowStarts[row];
    if (start1 != start2)
      return false;
    int end1 = row + 1 < matrix1->numRows ? matrix1->rowStarts[row] : matrix1->numNonzeros;
    int end2 = row + 1 < matrix2->numRows ? matrix2->rowStarts[row] : matrix2->numNonzeros;
    if (end1 != end2)
      return false;

    for (int i = start1; i < end1; ++i)
    {
      if (matrix1->entryColumns[i] != matrix2->entryColumns[i])
        return false;
      if (matrix1->entryValues[i] != matrix2->entryValues[i])
        return false;
    }
  }

  return true;
}

bool TUcheckIntMatrixEqual(TU_INT_MATRIX* matrix1, TU_INT_MATRIX* matrix2)
{
  assert(TUcheckIntMatrixSorted(matrix1));
  assert(TUcheckIntMatrixSorted(matrix2));

  if (matrix1->numRows != matrix2->numRows)
    return false;
  if (matrix1->numColumns != matrix2->numColumns)
    return false;
  if (matrix1->numColumns != matrix2->numColumns)
    return false;

  for (int row = 0; row < matrix1->numRows; ++row)
  {
    int start1 = matrix1->rowStarts[row];
    int start2 = matrix2->rowStarts[row];
    if (start1 != start2)
      return false;
    int end1 = row + 1 < matrix1->numRows ? matrix1->rowStarts[row] : matrix1->numNonzeros;
    int end2 = row + 1 < matrix2->numRows ? matrix2->rowStarts[row] : matrix2->numNonzeros;
    if (end1 != end2)
      return false;

    for (int i = start1; i < end1; ++i)
    {
      if (matrix1->entryColumns[i] != matrix2->entryColumns[i])
        return false;
      if (matrix1->entryValues[i] != matrix2->entryValues[i])
        return false;
    }
  }

  return true;
}

bool TUcheckCharMatrixEqual(TU_CHAR_MATRIX* matrix1, TU_CHAR_MATRIX* matrix2)
{
  assert(TUcheckCharMatrixSorted(matrix1));
  assert(TUcheckCharMatrixSorted(matrix2));

  if (matrix1->numRows != matrix2->numRows)
    return false;
  if (matrix1->numColumns != matrix2->numColumns)
    return false;
  if (matrix1->numColumns != matrix2->numColumns)
    return false;

  for (int row = 0; row < matrix1->numRows; ++row)
  {
    int start1 = matrix1->rowStarts[row];
    int start2 = matrix2->rowStarts[row];
    if (start1 != start2)
      return false;
    int end1 = row + 1 < matrix1->numRows ? matrix1->rowStarts[row + 1] : matrix1->numNonzeros;
    int end2 = row + 1 < matrix2->numRows ? matrix2->rowStarts[row + 1] : matrix2->numNonzeros;
    if (end1 != end2)
      return false;

    for (int i = start1; i < end1; ++i)
    {
      if (matrix1->entryColumns[i] != matrix2->entryColumns[i])
        return false;
      if (matrix1->entryValues[i] != matrix2->entryValues[i])
        return false;
    }
  }

  return true;
}

bool TUcheckDoubleMatrixTranspose(TU_DOUBLE_MATRIX* matrix1, TU_DOUBLE_MATRIX* matrix2)
{
  assert(matrix1 != NULL);
  assert(matrix2 != NULL);

  if (matrix1->numRows != matrix2->numColumns)
    return false;
  if (matrix1->numColumns != matrix2->numRows)
    return false;
  if (matrix1->numNonzeros != matrix2->numNonzeros)
    return false;

  int* currentColumnEntries = (int*) malloc(matrix1->numColumns * sizeof(int) );
  for (int column = 0; column < matrix2->numRows; ++column)
    currentColumnEntries[column] = matrix2->rowStarts[column];

  for (int row = 0; row < matrix1->numRows; ++row)
  {
    int begin = matrix1->rowStarts[row];
    int end = row + 1 < matrix1->numRows ? matrix1->rowStarts[row + 1] : matrix1->numNonzeros;
    for (int entry = begin; entry < end; ++entry)
    {
      int column = matrix1->entryColumns[entry];
      int entry2 = currentColumnEntries[column];
      if (matrix2->entryColumns[entry2] != row)
        return false;
      if (matrix2->entryValues[entry2] != matrix1->entryValues[entry])
        return false;
    }
  }

  free(currentColumnEntries);

  return true;
}

bool TUcheckIntMatrixTranspose(TU_INT_MATRIX* matrix1, TU_INT_MATRIX* matrix2)
{
  assert(matrix1 != NULL);
  assert(matrix2 != NULL);

  if (matrix1->numRows != matrix2->numColumns)
    return false;
  if (matrix1->numColumns != matrix2->numRows)
    return false;
  if (matrix1->numNonzeros != matrix2->numNonzeros)
    return false;

  int* currentColumnEntries = (int*) malloc(matrix1->numColumns * sizeof(int) );
  for (int column = 0; column < matrix2->numRows; ++column)
    currentColumnEntries[column] = matrix2->rowStarts[column];

  for (int row = 0; row < matrix1->numRows; ++row)
  {
    int begin = matrix1->rowStarts[row];
    int end = row + 1 < matrix1->numRows ? matrix1->rowStarts[row + 1] : matrix1->numNonzeros;
    for (int entry = begin; entry < end; ++entry)
    {
      int column = matrix1->entryColumns[entry];
      int entry2 = currentColumnEntries[column];
      if (matrix2->entryColumns[entry2] != row)
        return false;
      if (matrix2->entryValues[entry2] != matrix1->entryValues[entry])
        return false;
    }
  }

  free(currentColumnEntries);

  return true;
}

bool TUcheckCharMatrixTranspose(TU_CHAR_MATRIX* matrix1, TU_CHAR_MATRIX* matrix2)
{
  assert(matrix1 != NULL);
  assert(matrix2 != NULL);

  if (matrix1->numRows != matrix2->numColumns)
    return false;
  if (matrix1->numColumns != matrix2->numRows)
    return false;
  if (matrix1->numNonzeros != matrix2->numNonzeros)
    return false;

  int* currentColumnEntries = (int*) malloc(matrix1->numColumns * sizeof(int) );
  for (int column = 0; column < matrix2->numRows; ++column)
    currentColumnEntries[column] = matrix2->rowStarts[column];

  for (int row = 0; row < matrix1->numRows; ++row)
  {
    int begin = matrix1->rowStarts[row];
    int end = row + 1 < matrix1->numRows ? matrix1->rowStarts[row + 1] : matrix1->numNonzeros;
    for (int entry1 = begin; entry1 < end; ++entry1)
    {
      int column = matrix1->entryColumns[entry1];
      int entry2 = currentColumnEntries[column];
      if (matrix2->entryColumns[entry2] != row)
        return false;
      if (matrix2->entryValues[entry2] != matrix1->entryValues[entry1])
        return false;
      currentColumnEntries[column]++;
    }
  }

  free(currentColumnEntries);

  return true;
}

bool TUcheckDoubleMatrixSorted(TU_DOUBLE_MATRIX* sparse)
{
  assert(sparse != NULL);

  for (int row = 0; row < sparse->numRows; ++row)
  {
    int start = sparse->rowStarts[row];
    int end = row + 1 < sparse->numRows ? sparse->rowStarts[row+1] : sparse->numNonzeros;
    for (int i = start + 1; i < end; ++i)
    {
      if (sparse->entryColumns[i-1] > sparse->entryColumns[i])
        return false;
    }
  }

  return true;
}

bool TUcheckIntMatrixSorted(TU_INT_MATRIX* sparse)
{
  return TUcheckDoubleMatrixSorted((TU_DOUBLE_MATRIX*) sparse);
}

bool TUcheckCharMatrixSorted(TU_CHAR_MATRIX* sparse)
{
  return TUcheckDoubleMatrixSorted((TU_DOUBLE_MATRIX*) sparse);
}

bool TUisBinaryDouble(TU* tu, TU_DOUBLE_MATRIX* sparse, double epsilon, TU_SUBMATRIX** submatrix)
{
  assert(sparse != NULL);

  for (int row = 0; row < sparse->numRows; ++row)
  {
    int begin = sparse->rowStarts[row];
    int end = row + 1 < sparse->numRows ? sparse->rowStarts[row + 1] : sparse->numNonzeros;
    for (int entry = begin; entry < end; ++entry)
    {
      double value = sparse->entryValues[entry];
      int rounded = (int)(value + 0.5);
      if (rounded < 0 || rounded > +1 || fabs(value - rounded) > epsilon)
      {
        if (submatrix)
          TUcreateSubmatrix1x1(tu, submatrix, row, sparse->entryColumns[entry]);
        return false;
      }
    }
  }

  return true;
}

bool TUisBinaryInt(TU* tu, TU_INT_MATRIX* sparse, TU_SUBMATRIX** submatrix)
{
  assert(sparse != NULL);

  for (int row = 0; row < sparse->numRows; ++row)
  {
    int begin = sparse->rowStarts[row];
    int end = row + 1 < sparse->numRows ? sparse->rowStarts[row + 1] : sparse->numNonzeros;
    for (int entry = begin; entry < end; ++entry)
    {
      int value = sparse->entryValues[entry];
      if (value < 0 || value > 1)
      {
        if (submatrix)
          TUcreateSubmatrix1x1(tu, submatrix, row, sparse->entryColumns[entry]);
        return false;
      }
    }
  }

  return true;
}

bool TUisBinaryChar(TU* tu, TU_CHAR_MATRIX* sparse, TU_SUBMATRIX** submatrix)
{
  assert(sparse != NULL);

  for (int row = 0; row < sparse->numRows; ++row)
  {
    int begin = sparse->rowStarts[row];
    int end = row + 1 < sparse->numRows ? sparse->rowStarts[row + 1] : sparse->numNonzeros;
    for (int entry = begin; entry < end; ++entry)
    {
      char value = sparse->entryValues[entry];
      if (value < 0 || value > 1)
      {
        if (submatrix)
          TUcreateSubmatrix1x1(tu, submatrix, row, sparse->entryColumns[entry]);
        return false;
      }
    }
  }

  return true;
}

bool TUisTernaryDouble(TU* tu, TU_DOUBLE_MATRIX* sparse, double epsilon, TU_SUBMATRIX** submatrix)
{
  assert(sparse != NULL);

  for (int row = 0; row < sparse->numRows; ++row)
  {
    int begin = sparse->rowStarts[row];
    int end = row + 1 < sparse->numRows ? sparse->rowStarts[row + 1] : sparse->numNonzeros;
    for (int entry = begin; entry < end; ++entry)
    {
      double value = sparse->entryValues[entry];
      int rounded = (int)(value + 0.5);
      if (rounded < -1 || rounded > +1 || fabs(value - rounded) > epsilon)
      {
        if (submatrix)
          TUcreateSubmatrix1x1(tu, submatrix, row, sparse->entryColumns[entry]);
        return false;
      }
    }
  }

  return true;
}

bool TUisTernaryInt(TU* tu, TU_INT_MATRIX* sparse, TU_SUBMATRIX** submatrix)
{
  assert(sparse != NULL);

  for (int row = 0; row < sparse->numRows; ++row)
  {
    int begin = sparse->rowStarts[row];
    int end = row + 1 < sparse->numRows ? sparse->rowStarts[row + 1] : sparse->numNonzeros;
    for (int entry = begin; entry < end; ++entry)
    {
      int value = sparse->entryValues[entry];
      if (value < -1 || value > +1)
      {
        if (submatrix)
          TUcreateSubmatrix1x1(tu, submatrix, row, sparse->entryColumns[entry]);
        return false;
      }
    }
  }

  return true;
}

bool TUisTernaryChar(TU* tu, TU_CHAR_MATRIX* sparse, TU_SUBMATRIX** submatrix)
{
  assert(sparse != NULL);

  for (int row = 0; row < sparse->numRows; ++row)
  {
    int begin = sparse->rowStarts[row];
    int end = row + 1 < sparse->numRows ? sparse->rowStarts[row + 1] : sparse->numNonzeros;
    for (int entry = begin; entry < end; ++entry)
    {
      char value = sparse->entryValues[entry];
      if (value < -1 || value > +1)
      {
        if (submatrix)
          TUcreateSubmatrix1x1(tu, submatrix, row, sparse->entryColumns[entry]);
        return false;
      }
    }
  }

  return true;
}

void TUcreateSubmatrix(TU* tu, TU_SUBMATRIX** submatrix, int numRows, int numColumns)
{
  assert(submatrix != NULL);
  
  TUallocBlock(tu, submatrix);
  (*submatrix)->numRows = numRows;
  (*submatrix)->numColumns = numColumns;
  TUallocBlockArray(tu, &(*submatrix)->rows, numRows);
  TUallocBlockArray(tu, &(*submatrix)->columns, numColumns);
}

void TUcreateSubmatrix1x1(TU* tu, TU_SUBMATRIX** submatrix, int row, int column)
{
  TUcreateSubmatrix(tu, submatrix, 1, 1);
  (*submatrix)->rows[0] = row;
  (*submatrix)->columns[0] = column;
}

void TUfreeSubmatrix(TU* tu, TU_SUBMATRIX** submatrix)
{
  assert(submatrix);

  if ((*submatrix)->rows)
    TUfreeBlockArray(tu, &(*submatrix)->rows);
  if ((*submatrix)->columns)
    TUfreeBlockArray(tu, &(*submatrix)->columns);
  TUfreeBlockArray(tu, submatrix);
  *submatrix = NULL;
}

static int TUsortSubmatrixCompare(const void* p1, const void* p2)
{
  return *(int*)p1 - *(int*)p2;
}

void TUsortSubmatrix(TU_SUBMATRIX* submatrix)
{
  assert(submatrix);

  qsort(submatrix->rows, submatrix->numRows, sizeof(int), TUsortSubmatrixCompare);
  qsort(submatrix->columns, submatrix->numColumns, sizeof(int), TUsortSubmatrixCompare);
}

void TUfilterCharSubmatrix(TU* tu, TU_CHAR_MATRIX* matrix, TU_SUBMATRIX* submatrix,
  TU_CHAR_MATRIX** result)
{
  assert(matrix);
  assert(submatrix);
  assert(result);

  int* columnMap = NULL;
  TUallocBlockArray(tu, &columnMap, matrix->numColumns);
  for (int c = 0; c < matrix->numColumns; ++c)
    columnMap[c] = -1;
  for (int j = 0; j < submatrix->numColumns; ++j)
  {
    assert(submatrix->columns[j] < matrix->numColumns);
    columnMap[submatrix->columns[j]] = j;
  }

  TUcreateCharMatrix(tu, result, submatrix->numRows, submatrix->numColumns, 0);

  /* Count nonzeros. */
  int numNonzeros = 0;
  for (int i = 0; i < submatrix->numRows; ++i)
  {
    int r = submatrix->rows[i];
    assert(r < matrix->numRows);

    int begin = matrix->rowStarts[r];
    int end = r + 1 < matrix->numRows ? matrix->rowStarts[r+1] : matrix->numNonzeros;
    for (int e = begin; e < end; ++e)
    {
      int c = matrix->entryColumns[e];
      if (columnMap[c] >= 0)
        ++numNonzeros;
    }
  }
  
  (*result)->entryColumns = (int*) malloc( numNonzeros * sizeof(int) );
  (*result)->entryValues = (char*) malloc( numNonzeros * sizeof(char) );

  /* Copy nonzeros. */
  for (int i = 0; i < submatrix->numRows; ++i)
  {
    (*result)->rowStarts[i] = (*result)->numNonzeros;
    int r = submatrix->rows[i];
    assert(r < matrix->numRows);

    int begin = matrix->rowStarts[r];
    int end = r + 1 < matrix->numRows ? matrix->rowStarts[r+1] : matrix->numNonzeros;
    for (int e = begin; e < end; ++e)
    {
      int c = matrix->entryColumns[e];
      if (columnMap[c] >= 0)
      {
        (*result)->entryColumns[(*result)->numNonzeros] = columnMap[c];
        (*result)->entryValues[(*result)->numNonzeros] = matrix->entryValues[e];
        (*result)->numNonzeros++;
      }
    }
  }
  (*result)->rowStarts[(*result)->numRows] = (*result)->numNonzeros;

  if (columnMap)
    free(columnMap);
}
