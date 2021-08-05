#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include <cmr/matrix.h>
#include <cmr/series_parallel.h>

typedef enum
{
  FILEFORMAT_UNDEFINED = 0,
  FILEFORMAT_MATRIX_DENSE = 1,
  FILEFORMAT_MATRIX_SPARSE = 2
} FileFormat;

int printUsage(const char* program)
{
  printf("Usage: %s [OPTION]... FILE\n\n", program);
  puts("Applies all possible series-parallel reductions to the ternary or binary matrix in FILE.");
  puts("Options:");
  puts("  -i FORMAT  Format of input FILE; default: `dense'.");
  puts("  -o FORMAT  Format of output matrices; default: `dense'.");
  puts("  -sp        Output the list of series-parallel reductions.");
  puts("  -r         Output the elements of the reduced matrix.");
  puts("  -R         Output the reduced matrix.");
  puts("  -w         Output the elements of a wheel matrix if not series-parallel.");
  puts("  -W         Output a wheel matrix if not series-parallel.");
  puts("  -N NUM     Repeat the computation N times.");
  puts("Matrix formats: dense, sparse");
  puts("If FILE is `-', then the input will be read from stdin.");
  return EXIT_FAILURE;
}

CMR_ERROR matrixSeriesParallel2Sums(
  const char* instanceFileName, /**< File name of instance. */
  FileFormat inputFormat,       /**< Format of input matrix. */
  FileFormat outputFormat,      /**< Format of output matrices. */
  bool outputReductions,        /**< Whether to output the list of series-parallel reductions. */
  bool outputReducedElements,   /**< Whether to output the elements of the reduced matrix. */
  bool outputReducedMatrix,     /**< Whether to output the reduced matrix. */
  bool outputWheelElements,     /**< Whether to output the elements of a wheel matrix if not series-parallel. */
  bool outputWheelMatrix        /**< Whether to output a wheel matrix if not series-parallel. */
)
{
  clock_t startClock, endTime;
  FILE* instanceFile = strcmp(instanceFileName, "-") ? fopen(instanceFileName, "r") : stdin;
  if (!instanceFile)
    return CMR_ERROR_INPUT;

  CMR* cmr = NULL;
  CMR_CALL( CMRcreateEnvironment(&cmr) );

  /* Read matrix. */

  startClock = clock();
  CMR_CHRMAT* matrix = NULL;
  if (inputFormat == FILEFORMAT_MATRIX_DENSE)
    CMR_CALL( CMRchrmatCreateFromDenseStream(cmr, &matrix, instanceFile) );
  else if (inputFormat == FILEFORMAT_MATRIX_SPARSE)
    CMR_CALL( CMRchrmatCreateFromSparseStream(cmr, &matrix, instanceFile) );
  if (instanceFile != stdin)
    fclose(instanceFile);
  fprintf(stderr, "Read %dx%d matrix with %d nonzeros in %f seconds.\n", matrix->numRows, matrix->numColumns,
    matrix->numNonzeros, (clock() - startClock) * 1.0 / CLOCKS_PER_SEC);

  /* Run the search. */

  CMR_SP_REDUCTION* reductions = NULL;
  size_t numReductions = 0;
  CMR_CALL( CMRallocBlockArray(cmr, &reductions, matrix->numRows + matrix->numColumns) );
  CMR_SUBMAT* reducedSubmatrix = NULL;
  CMR_SUBMAT* wheelSubmatrix = NULL;

  CMR_SP_STATISTICS stats;
  CMR_CALL( CMRspInitStatistics(&stats) );
  CMR_CALL( CMRtestTernarySeriesParallel(cmr, matrix, true, NULL, reductions, &numReductions,
    (outputReducedElements || outputReducedMatrix) ? &reducedSubmatrix : NULL,
    (outputWheelElements || outputWheelMatrix) ? &wheelSubmatrix : NULL, &stats) );

  fprintf(stderr, "Recognition done in %f seconds with %f for reduction and %f for wheel search. Matrix %sseries-parallel; %ld reductions can be applied.\n",
    stats.totalTime, stats.reduceTime, stats.wheelTime,
    numReductions == matrix->numRows + matrix->numColumns ? "IS " : "is NOT ", numReductions);

  if (outputReductions)
  {
    fprintf(stderr, "Printing %ld series-parallel reductions.\n", numReductions);
    printf("%ld\n", numReductions);
    for (size_t i = 0; i < numReductions; ++i)
      printf("%s\n", CMRspReductionString(reductions[i], NULL));
  }

  if (outputReducedElements)
  {
    fprintf(stderr, "\nReduced submatrix consists of these elements:\n");
    printf("%ld rows:", reducedSubmatrix->numRows);
    for (size_t r = 0; r < reducedSubmatrix->numRows; ++r)
      printf(" %ld", reducedSubmatrix->rows[r]+1);
    printf("\n%ld columns: ", reducedSubmatrix->numColumns);
    for (size_t c = 0; c < reducedSubmatrix->numColumns; ++c)
      printf(" %ld", reducedSubmatrix->columns[c]+1);
    printf("\n");
  }

  if (outputReducedMatrix)
  {
    startClock = clock();
    CMR_CHRMAT* reducedMatrix = NULL;
    CMR_CALL( CMRchrmatFilterSubmat(cmr, matrix, reducedSubmatrix, &reducedMatrix) );
    endTime = clock();
    fprintf(stderr, "\nExtracted reduced %dx%d matrix with %d nonzeros in %f seconds.\n", reducedMatrix->numRows,
      reducedMatrix->numColumns, reducedMatrix->numNonzeros, (endTime - startClock) * 1.0 / CLOCKS_PER_SEC );
    if (outputFormat == FILEFORMAT_MATRIX_DENSE)
      CMR_CALL( CMRchrmatPrintDense(cmr, stdout, reducedMatrix, '0', false) );
    else if (outputFormat == FILEFORMAT_MATRIX_SPARSE)
      CMR_CALL( CMRchrmatPrintSparse(stdout, reducedMatrix) );
    CMR_CALL( CMRchrmatFree(cmr, &reducedMatrix) );
  }

  if (wheelSubmatrix && outputWheelElements)
  {
    fprintf(stderr, "\nWheel submatrix of order %ld consists of these elements of the input matrix:\n",
      wheelSubmatrix->numRows);
    printf("%ld rows:", wheelSubmatrix->numRows);
    for (size_t r = 0; r < wheelSubmatrix->numRows; ++r)
      printf(" %ld", wheelSubmatrix->rows[r]+1);
    printf("\n%ld columns: ", wheelSubmatrix->numColumns);
    for (size_t c = 0; c < wheelSubmatrix->numColumns; ++c)
      printf(" %ld", wheelSubmatrix->columns[c]+1);
    printf("\n");
  }

  if (wheelSubmatrix && outputWheelMatrix)
  {
    startClock = clock();
    CMR_CHRMAT* wheelMatrix = NULL;
    CMR_CALL( CMRchrmatFilterSubmat(cmr, matrix, wheelSubmatrix, &wheelMatrix) );
    endTime = clock();
    fprintf(stderr, "\nExtracted %dx%d wheel matrix with %d nonzeros in %f seconds.\n", wheelMatrix->numRows,
      wheelMatrix->numColumns, wheelMatrix->numNonzeros, (endTime - startClock) * 1.0 / CLOCKS_PER_SEC );
    if (outputFormat == FILEFORMAT_MATRIX_DENSE)
      CMR_CALL( CMRchrmatPrintDense(cmr, stdout, wheelMatrix, '0', false) );
    else if (outputFormat == FILEFORMAT_MATRIX_SPARSE)
      CMR_CALL( CMRchrmatPrintSparse(stdout, wheelMatrix) );
    CMR_CALL( CMRchrmatFree(cmr, &wheelMatrix) );
  }
  
  /* Cleanup. */

  CMR_CALL( CMRsubmatFree(cmr, &wheelSubmatrix) );
  CMR_CALL( CMRsubmatFree(cmr, &reducedSubmatrix) );
  CMR_CALL( CMRfreeBlockArray(cmr, &reductions) );
  CMR_CALL( CMRchrmatFree(cmr, &matrix) );
  CMR_CALL( CMRfreeEnvironment(&cmr) );

  return CMR_OKAY;
}

int main(int argc, char** argv)
{
  FileFormat inputFormat = FILEFORMAT_MATRIX_DENSE;
  FileFormat outputFormat = FILEFORMAT_MATRIX_DENSE;
  char* instanceFileName = NULL;
  bool outputReductions = false;
  bool outputReducedElements = false;
  bool outputReducedMatrix = false;
  bool outputWheelElements = false;
  bool outputWheelMatrix = false;
  for (int a = 1; a < argc; ++a)
  {
    if (!strcmp(argv[a], "-h"))
    {
      printUsage(argv[0]);
      return EXIT_SUCCESS;
    }
    else if (!strcmp(argv[a], "-sp"))
      outputReductions = true;
    else if (!strcmp(argv[a], "-r"))
      outputReducedElements = true;
    else if (!strcmp(argv[a], "-R"))
      outputReducedMatrix = true;
    else if (!strcmp(argv[a], "-w"))
      outputWheelElements = true;
    else if (!strcmp(argv[a], "-W"))
      outputWheelMatrix = true;
    else if (!strcmp(argv[a], "-i") && a+1 < argc)
    {
      if (!strcmp(argv[a+1], "dense"))
        inputFormat = FILEFORMAT_MATRIX_DENSE;
      else if (!strcmp(argv[a+1], "sparse"))
        inputFormat = FILEFORMAT_MATRIX_SPARSE;
      else
      {
        printf("Error: unknown input file format <%s>.\n\n", argv[a+1]);
        return printUsage(argv[0]);
      }
      ++a;
    }
    else if (!strcmp(argv[a], "-o") && a+1 < argc)
    {
      if (!strcmp(argv[a+1], "dense"))
        outputFormat = FILEFORMAT_MATRIX_DENSE;
      else if (!strcmp(argv[a+1], "sparse"))
        outputFormat = FILEFORMAT_MATRIX_SPARSE;
      else
      {
        printf("Error: unknown output format <%s>.\n\n", argv[a+1]);
        return printUsage(argv[0]);
      }
      ++a;
    }
    else if (!instanceFileName)
      instanceFileName = argv[a];
    else
    {
      printf("Error: Two input files <%s> and <%s> specified.\n\n", instanceFileName, argv[a]);
      return printUsage(argv[0]);
    }
  }

  if (!instanceFileName)
  {
    puts("No input file specified.\n");
    return printUsage(argv[0]);
  }

  CMR_ERROR error = matrixSeriesParallel2Sums(instanceFileName, inputFormat, outputFormat, outputReductions,
    outputReducedElements, outputReducedMatrix, outputWheelElements, outputWheelMatrix);
  switch (error)
  {
  case CMR_ERROR_INPUT:
    puts("Input error.");
    return EXIT_FAILURE;
  case CMR_ERROR_MEMORY:
    puts("Memory error.");
    return EXIT_FAILURE;
  default:
    return EXIT_SUCCESS;
  }
}
