#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include <cmr/matrix.h>
#include <cmr/sign.h>

typedef enum
{
  UNDEFINED = 0,
  DENSE = 1,
  SPARSE = 2
} Format;

typedef enum
{
  COPY = 0,
  SUPPORT = 1,
  SIGNED_SUPPORT = 2
} Task;

static
CMR_ERROR printDbl(CMR* cmr, CMR_DBLMAT* matrix, Format outputFormat, bool transpose)
{
  assert(matrix);

  CMR_DBLMAT* output = NULL;
  if (transpose)
  {
    CMR_CALL( CMRdblmatTranspose(cmr, matrix, &output) );
  }
  else
    output = matrix;

  CMR_ERROR error = CMR_OKAY;
  if (outputFormat == SPARSE)
    CMR_CALL( CMRdblmatPrintSparse(stdout, output) );
  else if (outputFormat == DENSE)
    CMR_CALL( CMRdblmatPrintDense(stdout, output, '0', false) );
  else
    error = CMR_ERROR_INPUT;

  if (transpose)
    CMR_CALL( CMRdblmatFree(cmr, &output) );

  return error;
}

static
CMR_ERROR printInt(CMR* cmr, CMR_INTMAT* matrix, Format outputFormat, bool transpose)
{
  assert(matrix);

  CMR_INTMAT* output = NULL;
  if (transpose)
  {
    CMR_CALL( CMRintmatTranspose(cmr, matrix, &output) );
  }
  else
    output = matrix;

  CMR_ERROR error = CMR_OKAY;
  if (outputFormat == SPARSE)
    CMR_CALL( CMRintmatPrintSparse(stdout, output) );
  else if (outputFormat == DENSE)
    CMR_CALL( CMRintmatPrintDense(stdout, output, '0', false) );
  else
    error = CMR_ERROR_INPUT;

  if (transpose)
    CMR_CALL( CMRintmatFree(cmr, &output) );

  return error;
}

static
CMR_ERROR printChr(CMR* cmr, CMR_CHRMAT* matrix, Format outputFormat, bool transpose)
{
  assert(matrix);

  CMR_CHRMAT* output = NULL;
  if (transpose)
  {
    CMR_CALL( CMRchrmatTranspose(cmr, matrix, &output) );
  }
  else
    output = matrix;

  CMR_ERROR error = CMR_OKAY;
  if (outputFormat == SPARSE)
    CMR_CALL( CMRchrmatPrintSparse(stdout, output) );
  else if (outputFormat == DENSE)
    CMR_CALL( CMRchrmatPrintDense(cmr, stdout, output, '0', false) );
  else
    error = CMR_ERROR_INPUT;

  if (transpose)
    CMR_CALL( CMRchrmatFree(cmr, &output) );

  return error;
}

CMR_ERROR runDbl(const char* instanceFileName, Format inputFormat, Format outputFormat, Task task, bool transpose)
{
  FILE* instanceFile = strcmp(instanceFileName, "-") ? fopen(instanceFileName, "r") : stdin;
  if (!instanceFile)
    return CMR_ERROR_INPUT;

  CMR* cmr = NULL;
  CMR_CALL( CMRcreateEnvironment(&cmr) );

  CMR_DBLMAT* matrix = NULL;
  if (inputFormat == SPARSE)
    CMR_CALL( CMRdblmatCreateFromSparseStream(cmr, &matrix, instanceFile) );
  else if (inputFormat == DENSE)
    CMR_CALL( CMRdblmatCreateFromDenseStream(cmr, &matrix, instanceFile) );
  else
    return CMR_ERROR_INPUT;
  if (instanceFile != stdin)
    fclose(instanceFile);

  if (task == SUPPORT)
  {
    CMR_CHRMAT* result = NULL;
    CMR_CALL( CMRsupportDbl(cmr, matrix, 1.0e-9, &result) );
    CMR_CALL( printChr(cmr, result, outputFormat, transpose) );
    CMR_CALL( CMRchrmatFree(cmr, &result) );
  }
  else if (task == SIGNED_SUPPORT)
  {
    CMR_CHRMAT* result = NULL;
    CMR_CALL( CMRsignedSupportDbl(cmr, matrix, 1.0e-9, &result) );
    CMR_CALL( printChr(cmr, result, outputFormat, transpose) );
    CMR_CALL( CMRchrmatFree(cmr, &result) );
  }
  else
  {
    CMR_CALL( printDbl(cmr, matrix, outputFormat, transpose) );
  }

  CMR_CALL( CMRdblmatFree(cmr, &matrix) );

  CMR_CALL( CMRfreeEnvironment(&cmr) );

  return CMR_OKAY;
}

CMR_ERROR runInt(const char* instanceFileName, Format inputFormat, Format outputFormat, Task task, bool transpose)
{
  FILE* instanceFile = strcmp(instanceFileName, "-") ? fopen(instanceFileName, "r") : stdin;
  if (!instanceFile)
    return CMR_ERROR_INPUT;

  CMR* cmr = NULL;
  CMR_CALL( CMRcreateEnvironment(&cmr) );

  CMR_INTMAT* matrix = NULL;
  if (inputFormat == SPARSE)
    CMR_CALL( CMRintmatCreateFromSparseStream(cmr, &matrix, instanceFile) );
  else if (inputFormat == DENSE)
    CMR_CALL( CMRintmatCreateFromDenseStream(cmr, &matrix, instanceFile) );
  else
    return CMR_ERROR_INPUT;
  if (instanceFile != stdin)
    fclose(instanceFile);

  if (task == SUPPORT)
  {
    CMR_CHRMAT* result = NULL;
    CMR_CALL( CMRsupportInt(cmr, matrix, &result) );
    CMR_CALL( printChr(cmr, result, outputFormat, transpose) );
    CMR_CALL( CMRchrmatFree(cmr, &result) );
  }
  else if (task == SIGNED_SUPPORT)
  {
    CMR_CHRMAT* result = NULL;
    CMR_CALL( CMRsignedSupportInt(cmr, matrix, &result) );
    CMR_CALL( printChr(cmr, result, outputFormat, transpose) );
    CMR_CALL( CMRchrmatFree(cmr, &result) );
  }
  else
  {
    CMR_CALL( printInt(cmr, matrix, outputFormat, transpose) );
  }

  CMR_CALL( CMRintmatFree(cmr, &matrix) );

  CMR_CALL( CMRfreeEnvironment(&cmr) );

  return CMR_OKAY;
}

int printUsage(const char* program)
{
  printf("Usage: %s [OPTION]... MATRIX\n\n", program);
  puts("Copies MATRIX, potentially applying an operation.");
  puts("\nOptions:");
  puts("  -i, --input FORMAT  Format of MATRIX file, among {dense, sparse}; default: dense.");
  puts("  -o, --output FORMAT Format of output, among {dense, sparse}; default: same as input.");
  puts("  -s, --support       Create support matrix instead of copying.");
  puts("  -t, --transpose     Output transposed matrix (can be combined with other operations).");
  puts("  -S, --sign          Create signed support matrix instead of copying.");
  puts("  -d, --double        Use double arithmetic.");
  puts("If MATRIX is `-', then the matrix will be read from stdin.");
  
  return EXIT_FAILURE;
}

int main(int argc, char** argv)
{
  /* Parse command line options. */
  Format inputFormat = DENSE;
  Format outputFormat = UNDEFINED;
  Task task = COPY;
  bool transpose = false;
  bool doubleArithmetic = false;
  char* instanceFileName = NULL;
  for (int a = 1; a < argc; ++a)
  {
    if (!strcmp(argv[a], "-h") || !strcmp(argv[a], "--help"))
    {
      printUsage(argv[0]);
      return EXIT_SUCCESS;
    }
    else if ((!strcmp(argv[a], "-i") || !strcmp(argv[a], "--input")) && a+1 < argc)
    {
      if (!strcmp(argv[a+1], "dense"))
        inputFormat = DENSE;
      else if (!strcmp(argv[a+1], "sparse"))
        inputFormat = SPARSE;
      else
      {
        printf("Error: unknown input format <%s>.\n\n", argv[a+1]);
        return printUsage(argv[0]);
      }
      ++a;
    }
    else if ((!strcmp(argv[a], "-o") || !strcmp(argv[a], "--output")) && a+1 < argc)
    {
      if (!strcmp(argv[a+1], "dense"))
        outputFormat = DENSE;
      else if (!strcmp(argv[a+1], "sparse"))
        outputFormat = SPARSE;
      else
      {
        printf("Error: unknown output format <%s>.\n\n", argv[a+1]);
        return printUsage(argv[0]);
      }
      ++a;
    }
    else if (!strcmp(argv[a], "-s") || !strcmp(argv[a], "--support"))
      task = SUPPORT;
    else if (!strcmp(argv[a], "-S") || !strcmp(argv[a], "--sign"))
      task = SIGNED_SUPPORT;
    else if (!strcmp(argv[a], "-t") || !strcmp(argv[a], "--transpose"))
      transpose = true;
    else if (!strcmp(argv[a], "-d") || !strcmp(argv[a], "--double"))
      doubleArithmetic = true;
    else if (!instanceFileName)
      instanceFileName = argv[a];
    else
    {
      printf("Error: Two input matrix files <%s> and <%s> specified.\n\n", instanceFileName, argv[a]);
      return printUsage(argv[0]);
    }
  }

  if (!instanceFileName)
  {
    puts("No input matrix specified.\n");
    return printUsage(argv[0]);
  }
  if (outputFormat == UNDEFINED)
    outputFormat = inputFormat;

  CMR_ERROR error;
  if (doubleArithmetic)
    error = runDbl(instanceFileName, inputFormat, outputFormat, task, transpose);
  else
    error = runInt(instanceFileName, inputFormat, outputFormat, task, transpose);
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
