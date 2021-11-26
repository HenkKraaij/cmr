#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include <cmr/matrix.h>
#include <cmr/camion.h>

typedef enum
{
  FILEFORMAT_UNDEFINED = 0,       /**< Whether the file format of input/output was defined by the user. */
  FILEFORMAT_MATRIX_DENSE = 1,    /**< Dense matrix format. */
  FILEFORMAT_MATRIX_SPARSE = 2,   /**< Sparse matrix format. */
} FileFormat;

/**
 * \brief Prints the usage of the \p program to stdout.
 * 
 * \returns \c EXIT_FAILURE.
 */

int printUsage(const char* program)
{
  printf("Usage: %s [OPTION]... FILE\n\n", program);
  puts("Checks whether matrix in FILE is Camion-signed.");
  puts("Options:");
  puts("  -i FORMAT  Format of input FILE; default: `dense'.");
  puts("  -o FORMAT  Format of output; default: `dense'.");
  puts("  -n         Output the elements of a minimal non-Camion submatrix.");
  puts("  -N         Output a minimal non-Camion submatrix.");
  puts("  -s         Print statistics about the computation to stderr.");
  puts("Formats for matrices: dense, sparse");
  puts("If FILE is `-', then the input will be read from stdin.");
  return EXIT_FAILURE;
}

/**
 * \brief Tests matrix from a file for total unimodularity.
 */

static
CMR_ERROR testCamionSigned(
  const char* instanceFileName, /**< File name containing the input matrix (may be `-' for stdin). */
  FileFormat inputFormat,       /**< Format of the input matrix. */
  FileFormat outputFormat,      /**< Format of the output submatrix. */
  bool outputSubmatrixElements, /**< Whether to print the elements of a non-camion submatrix. */
  bool outputSubmatrix,         /**< Whether to print a non-camion submatrix. */
  bool printStats               /**< Whether to print statistics to stderr. */
)
{
  clock_t readClock = clock();
  FILE* instanceFile = strcmp(instanceFileName, "-") ? fopen(instanceFileName, "r") : stdin;
  if (!instanceFile)
    return CMR_ERROR_INPUT;

  CMR* cmr = NULL;
  CMR_CALL( CMRcreateEnvironment(&cmr) );

  /* Read matrix. */

  CMR_CHRMAT* matrix = NULL;
  if (inputFormat == FILEFORMAT_MATRIX_DENSE)
    CMR_CALL( CMRchrmatCreateFromDenseStream(cmr, instanceFile, &matrix) );
  else if (inputFormat == FILEFORMAT_MATRIX_SPARSE)
    CMR_CALL( CMRchrmatCreateFromSparseStream(cmr, instanceFile, &matrix) );
  if (instanceFile != stdin)
    fclose(instanceFile);
  fprintf(stderr, "Read %lux%lu matrix with %lu nonzeros in %f seconds.\n", matrix->numRows, matrix->numColumns,
    matrix->numNonzeros, (clock() - readClock) * 1.0 / CLOCKS_PER_SEC);

  /* Actual test. */

  bool isCamion;
  CMR_SUBMAT* submatrix = NULL;
  CMR_CAMION_STATISTICS stats;
  CMR_CALL( CMRstatsCamionInit(&stats) );
  CMR_CALL( CMRtestCamionSigned(cmr, matrix, &isCamion,
    (outputFormat || outputSubmatrixElements) ? &submatrix : NULL, printStats ? &stats : NULL) );

  fprintf(stderr, "Matrix %sCamion-signed.\n", isCamion ? "IS " : "IS NOT ");
  if (printStats)
    CMR_CALL( CMRstatsCamionPrint(stderr, &stats, NULL) );

  if (submatrix)
  {
    if (outputSubmatrixElements)
    {
      fprintf(stderr, "\nNon-camion submatrix consists of these elements:\n");
      printf("%ld rows:", submatrix->numRows);
      for (size_t r = 0; r < submatrix->numRows; ++r)
        printf(" %ld", submatrix->rows[r]+1);
      printf("\n%ld columns: ", submatrix->numColumns);
      for (size_t c = 0; c < submatrix->numColumns; ++c)
        printf(" %ld", submatrix->columns[c]+1);
      printf("\n");
    }

    if (outputSubmatrix)
    {
      CMR_CHRMAT* violatorMatrix = NULL;
      CMR_CALL( CMRchrmatZoomSubmat(cmr, matrix, submatrix, &violatorMatrix) );
      fprintf(stderr, "\nExtracted %lux%lu non-camion submatrix with %lu nonzeros.\n", violatorMatrix->numRows,
        violatorMatrix->numColumns, violatorMatrix->numNonzeros);
      if (outputFormat == FILEFORMAT_MATRIX_DENSE)
        CMR_CALL( CMRchrmatPrintDense(cmr, violatorMatrix, stdout, '0', false) );
      else if (outputFormat == FILEFORMAT_MATRIX_SPARSE)
        CMR_CALL( CMRchrmatPrintSparse(cmr, violatorMatrix, stdout) );
      CMR_CALL( CMRchrmatFree(cmr, &violatorMatrix) );
    }

    CMR_CALL( CMRsubmatFree(cmr, &submatrix) );
  }

  /* Cleanup. */

  CMR_CALL( CMRchrmatFree(cmr, &matrix) );
  CMR_CALL( CMRfreeEnvironment(&cmr) );

  return CMR_OKAY;
}

int main(int argc, char** argv)
{
  FileFormat inputFormat = FILEFORMAT_UNDEFINED;
  FileFormat outputFormat = FILEFORMAT_MATRIX_DENSE;
  bool outputSubmatrixElements = false;
  bool outputSubmatrix = false;
  bool printStats = false;
  char* instanceFileName = NULL;
  for (int a = 1; a < argc; ++a)
  {
    if (!strcmp(argv[a], "-h"))
    {
      printUsage(argv[0]);
      return EXIT_SUCCESS;
    }
    else if (!strcmp(argv[a], "-n"))
      outputSubmatrixElements = true;
    else if (!strcmp(argv[a], "-N"))
      outputSubmatrix = true;
    else if (!strcmp(argv[a], "-s"))
      printStats = true;
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

  if (inputFormat == FILEFORMAT_UNDEFINED)
    inputFormat = FILEFORMAT_MATRIX_DENSE;

  CMR_ERROR error;
  error = testCamionSigned(instanceFileName, inputFormat, outputFormat, outputSubmatrixElements, outputFormat,
    printStats);

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
