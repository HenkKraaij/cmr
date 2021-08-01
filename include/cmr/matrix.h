#ifndef TU_MATRIX_H
#define TU_MATRIX_H

#include <cmr/env.h>

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif


/**
 * \brief Row and column indices for a submatrix
 *
 * Does not contain information about the matrix it refers to.
 */
typedef struct
{
  /**
   * \brief Number of rows
   */
  int numRows;
  /**
   * \brief Array with row indices
   */
  int* rows;
  /**
   * \brief Number of columns
   */
  int numColumns;
  /**
   * \brief Array with column indices
   */
  int* columns;
} TU_SUBMAT;

/**
 * \brief Row-wise representation of sparse double matrix.
 * 
 * The columns and values of all nonzeros are stored in \ref entryColumns and \ref entryValues,
 * respectively.
 * Those of row \c r are stored from \ref rowStarts[r] until (but not including)
 * \ref rowStarts[r+1]. The last row is an exception, since \ref rowStarts[\ref numRows] need not
 * be defined.
 * For convenience, one may store this additional entry.
 * In particular \ref TUdblmatCreate allocates sufficient space for it.
 * However, all public methods use \ref numRows to determine the last row's number of nonzeros via
 * \ref numNonzeros.
 */

typedef struct
{
  int numRows;          /**< \brief Number of rows. */
  int numColumns;       /**< \brief Number of columns. */
  int numNonzeros;      /**< \brief Number of and memory allocated for nonzeros. */
  int* rowStarts;       /**< \brief Array mapping each row to the index of its first entry. */
  int* entryColumns;    /**< \brief Array mapping each entry to its column.*/
  double* entryValues;  /**< \brief Array mapping each entry to its value. */
} TU_DBLMAT;

/**
 * \brief Row-wise representation of sparse int matrix.
 * 
 * The columns and values of all nonzeros are stored in \ref entryColumns and \ref entryValues,
 * respectively.
 * Those of row \c r are stored from \ref rowStarts[r] until (but not including)
 * \ref rowStarts[r+1]. The last row is an exception, since \ref rowStarts[\ref numRows] need not
 * be defined.
 * For convenience, one may store this additional entry.
 * In particular \ref TUintmatCreate allocates sufficient space for it.
 * However, all public methods use \ref numRows to determine the last row's number of nonzeros via
 * \ref numNonzeros.
 */

typedef struct
{
  int numRows;        /**< \brief Number of rows. */
  int numColumns;     /**< \brief Number of columns. */
  int numNonzeros;    /**< \brief Number of and memory allocated for nonzeros. */
  int* rowStarts;     /**< \brief Array mapping each row to the index of its first entry. */
  int* entryColumns;  /**< \brief Array mapping each entry to its column.*/
  int* entryValues;   /**< \brief Array mapping each entry to its value. */
} TU_INTMAT;

/**
 * \brief Row-wise representation of sparse char matrix.
 * 
 * The columns and values of all nonzeros are stored in \ref entryColumns and \ref entryValues,
 * respectively.
 * Those of row \c r are stored from \ref rowStarts[r] until (but not including)
 * \ref rowStarts[r+1]. The last row is an exception, since \ref rowStarts[\ref numRows] need not
 * be defined.
 * For convenience, one may store this additional entry.
 * In particular \ref TUchrmatCreate allocates sufficient space for it.
 * However, all public methods use \ref numRows to determine the last row's number of nonzeros via
 * \ref numNonzeros.
 */

typedef struct
{
  int numRows;        /**< \brief Number of rows. */
  int numColumns;     /**< \brief Number of columns. */
  int numNonzeros;    /**< \brief Number of and memory allocated for nonzeros. */
  int* rowStarts;     /**< \brief Array mapping each row to the index of its first entry. */
  int* entryColumns;  /**< \brief Array mapping each entry to its column.*/
  char* entryValues;  /**< \brief Array mapping each entry to its value. */
} TU_CHRMAT;



















/**
 * \brief Creates a double matrix of size \p numRows times \p numColumns with \p numNonzeros
 *        nonzeros. The row starts and entries are allocated but not initialized.
 */

CMR_EXPORT
CMR_ERROR TUdblmatCreate(
  TU* tu,              /**< \ref TU environment. */
  TU_DBLMAT** matrix,  /**< Pointer for storing the created matrix. */
  int numRows,         /**< Number of rows. */
  int numColumns,      /**< Number of columns. */
  int numNonzeros      /**< Number of nonzeros. */
);





/**
 * \brief Creates a submatrix of given size.
 *
 * Only allocates the memory. Use rows and columns attributes of *\p psubmatrix to actually set the row and column
 * indices, respectively.
 */
CMR_EXPORT
CMR_ERROR TUsubmatCreate(
  TU* tu,                 /**< \ref TU environment. */
  TU_SUBMAT** psubmatrix, /**< Pointer to where the submatrix is to be stored. */
  int numRows,            /**< Number of rows */
  int numColumns          /**< Number of columns */
);

/**
 * \brief Creates a 1x1 submatrix.
 */

CMR_EXPORT
void TUsubmatCreate1x1(
  TU* tu,                 /**< \ref TU environment. */
  TU_SUBMAT** psubmatrix, /**< Pointer to submatrix */
  int row,                /**< Row of entry */
  int column              /**< Column of entry */
);

/**
 * \brief Frees a submatrix.
 */
CMR_EXPORT
CMR_ERROR TUsubmatFree(
  TU* tu,                 /**< \ref TU environment. */
  TU_SUBMAT** psubmatrix  /**< Pointer to submatrix. */
);




















/**
 * \brief Frees the memory of a double matrix.
 */

CMR_EXPORT
CMR_ERROR TUdblmatFree(
  TU* tu,             /**< \ref TU environment. */
  TU_DBLMAT** matrix  /**< Pointer to matrix. */
);

/**
 * \brief Changes the number of nonzeros and reallocates corresponding arrays.
 */

CMR_EXPORT
CMR_ERROR TUdblmatChangeNumNonzeros(
  TU* tu,             /**< \ref TU environment. */
  TU_DBLMAT* matrix,  /**< Given matrix. */
  int newNumNonzeros  /**< New number of nonzeros. */ 
);

/**
 * \brief Copies a double matrix to a newly allocated one.
 * 
 * Allocates *\p result and copies \p matrix there.
 */
CMR_EXPORT
CMR_ERROR TUdblmatCopy(
  TU* tu,             /**< \ref TU environment. */
  TU_DBLMAT* matrix,  /**< Given matrix. */
  TU_DBLMAT** result  /**< Pointer to store a copy of \p matrix. */
);

/**
 * \brief Creates the transpose of a double matrix.
 */
CMR_EXPORT
CMR_ERROR TUdblmatTranspose(
  TU* tu,             /**< \ref TU environment. */
  TU_DBLMAT* matrix,  /**< Given matrix. */
  TU_DBLMAT** result  /**< Pointer to store the transpose of \p matrix. */
);

/**
 * \brief Prints a double matrix.
 */

CMR_EXPORT
CMR_ERROR TUdblmatPrintSparse(
  FILE* stream,       /**< File stream to print to. */
  TU_DBLMAT* matrix   /**< Double matrix. */
);


/**
 * \brief Prints a double matrix.
 */

CMR_EXPORT
CMR_ERROR TUdblmatPrintDense(
  FILE* stream,       /**< File stream to print to. */
  TU_DBLMAT* matrix,  /**< Double matrix. */
  char zeroChar,      /**< Character to print for a zero. */
  bool header         /**< Whether to print row and column indices. */
);

/**
 * \brief Reads a sparse double matrix from a file \p stream.
 * 
 * Zero entries are ignored, and multiple occurences of (row,column) pairs are considered as errors.
 * Returns \ref CMR_ERROR_INPUT in case of errors. In this case, *\p pmatrix will be \c NULL.
 */

CMR_EXPORT
CMR_ERROR TUdblmatCreateFromSparseStream(
  TU* tu,               /**< \ref TU environment. */
  TU_DBLMAT** pmatrix,  /**< Pointer for storing the matrix. */
  FILE* stream          /**< File stream to read from. */
);

/**
 * \brief Reads a densely stored double matrix from a file \p stream.
 */

CMR_EXPORT
CMR_ERROR TUdblmatCreateFromDenseStream(
  TU* tu,               /**< \ref TU environment. */
  TU_DBLMAT** pmatrix,  /**< Pointer for storing the matrix. */
  FILE* stream          /**< File stream to read from. */
);

/**
 * \brief Checks whether two double matrices are equal.
 */

CMR_EXPORT
bool TUdblmatCheckEqual(
  TU_DBLMAT* matrix1,  /**< First matrix */
  TU_DBLMAT* matrix2   /**< Second matrix */
);

/**
 * \brief Checks whether two double matrices are transposes of each other.
 */

CMR_EXPORT
bool TUdblmatCheckTranspose(
  TU_DBLMAT* matrix1,  /**< First matrix */
  TU_DBLMAT* matrix2   /**< Second matrix */
);

/**
 * \brief Checks whether double matrix has each row sorted by minor.
 */

CMR_EXPORT
bool TUdblmatCheckSorted(
  TU_DBLMAT* matrix /**< Double matrix */
);

/**
 * \brief Creates a submatrix of a double matrix as an explicit matrix.
 */
CMR_EXPORT
CMR_ERROR TUdblmatFilterSubmat(
  TU* tu,               /**< \ref TU environment. */
  TU_DBLMAT* matrix,    /**< Given matrix */
  TU_SUBMAT* submatrix, /**< Specified submatrix */
  TU_DBLMAT** result    /**< Pointer for storing the resulting double matrix. */
);

/**
 * \brief Checks if double matrix has only entries in {0, 1} with absolute error tolerance \p epsilon.
 */

CMR_EXPORT
bool TUisBinaryDbl(
  TU* tu,                 /**< \ref TU environment. */
  TU_DBLMAT* matrix,      /**< Double matrix */
  double epsilon,         /**< Absolute error tolerance */
  TU_SUBMAT** psubmatrix  /**< Pointer for storing a non-binary entry as a submatrix (may be \c NULL). */
);

/**
 * \brief Checks if double matrix has only entries in {-1, 0, +1} with absolute error tolerance \p epsilon.
 */

CMR_EXPORT
bool TUisTernaryDbl(
  TU* tu,                 /**< \ref TU environment. */
  TU_DBLMAT* matrix,      /**< Double matrix */
  double epsilon,         /**< Absolute error tolerance */
  TU_SUBMAT** psubmatrix  /**< Pointer for storing a non-ternary entry as a submatrix (may be \c NULL). */
);

/**
 * \brief Creates the support matrix of a double \p matrix as a char matrix.
 */

CMR_EXPORT
CMR_ERROR TUsupportDbl(
  TU* tu,               /**< \ref TU environment. */
  TU_DBLMAT* matrix,    /**< Double matrix */
  double epsilon,       /**< Absolute error tolerance */
  TU_CHRMAT** psupport  /**< Pointer for storing the support matrix of \p matrix. */
);

/**
 * \brief Creates the signed support matrix of a double \p matrix as a char matrix.
 */

CMR_EXPORT
CMR_ERROR TUsignedSupportDbl(
  TU* tu,               /**< \ref TU environment. */
  TU_DBLMAT* matrix,    /**< Double matrix */
  double epsilon,       /**< Absolute error tolerance */
  TU_CHRMAT** psupport  /**< Pointer for storing the support matrix of \p matrix. */
);











/**
 * \brief Creates a double matrix of size \p numRows times \p numColumns with \p numNonzeros
 *        nonzeros. The row starts and entries are allocated but not initialized.
 */

CMR_EXPORT
CMR_ERROR TUintmatCreate(
  TU* tu,              /**< \ref TU environment. */
  TU_INTMAT** matrix,  /**< Pointer for storing the created matrix. */
  int numRows,         /**< Number of rows. */
  int numColumns,      /**< Number of columns. */
  int numNonzeros      /**< Number of nonzeros. */
);

/**
 * \brief Frees the memory of an int matrix.
 */

CMR_EXPORT
CMR_ERROR TUintmatFree(
  TU* tu,             /**< \ref TU environment. */
  TU_INTMAT** matrix  /**< Pointer to matrix. */
);

/**
 * \brief Changes the number of nonzeros and reallocates corresponding arrays.
 */

CMR_EXPORT
CMR_ERROR TUintmatChangeNumNonzeros(
  TU* tu,             /**< \ref TU environment. */
  TU_INTMAT* matrix,  /**< Given matrix. */
  int newNumNonzeros  /**< New number of nonzeros. */ 
);

/**
 * \brief Copies an int matrix to a newly allocated one.
 * 
 * Allocates *\p result and copies \p matrix there.
 */
CMR_EXPORT
CMR_ERROR TUintmatCopy(
  TU* tu,             /**< \ref TU environment. */
  TU_INTMAT* matrix,  /**< Given matrix. */
  TU_INTMAT** result  /**< Pointer to store a copy of \p matrix. */
);

/**
 * \brief Creates the transpose of an int matrix.
 */
CMR_EXPORT
CMR_ERROR TUintmatTranspose(
  TU* tu,             /**< \ref TU environment. */
  TU_INTMAT* matrix,  /**< Given matrix. */
  TU_INTMAT** result  /**< Pointer to store the transpose of \p matrix. */
);

/**
 * \brief Prints an int matrix.
 */

CMR_EXPORT
CMR_ERROR TUintmatPrintSparse(
  FILE* stream,       /**< File stream to print to. */
  TU_INTMAT* matrix   /**< Int matrix. */
);

/**
 * \brief Prints an int matrix.
 */

CMR_EXPORT
CMR_ERROR TUintmatPrintDense(
  FILE* stream,       /**< File stream to print to. */
  TU_INTMAT* matrix,  /**< Int matrix. */
  char zeroChar,      /**< Character to print for a zero. */
  bool header         /**< Whether to print row and column indices. */
);

/**
 * \brief Reads a sparse int matrix from a file \p stream.
 *  
 * Zero entries are ignored, and multiple occurences of (row,column) pairs are considered as errors.
 * Returns \ref CMR_ERROR_INPUT in case of errors. In this case, *\p pmatrix will be \c NULL.
 */

CMR_EXPORT
CMR_ERROR TUintmatCreateFromSparseStream(
  TU* tu,               /**< \ref TU environment. */
  TU_INTMAT** pmatrix,  /**< Pointer for storing the matrix. */
  FILE* stream          /**< File stream to read from. */
);

/**
 * \brief Reads a densely stored int matrix from a file \p stream.
 */

CMR_EXPORT
CMR_ERROR TUintmatCreateFromDenseStream(
  TU* tu,               /**< \ref TU environment. */
  TU_INTMAT** pmatrix,  /**< Pointer for storing the matrix. */
  FILE* stream          /**< File stream to read from. */
);

/**
 * \brief Checks whether two int matrices are equal.
 */

CMR_EXPORT
bool TUintmatCheckEqual(
  TU_INTMAT* matrix1, /**< First matrix */
  TU_INTMAT* matrix2  /**< Second matrix */
);

/**
 * \brief Checks whether two int matrices are transposes of each other.
 */

CMR_EXPORT
bool TUintmatCheckTranspose(
  TU_INTMAT* matrix1, /**< First matrix */
  TU_INTMAT* matrix2  /**< Second matrix */
);

/**
 * \brief Checks whether int matrix has each row sorted by minor.
 */

CMR_EXPORT
bool TUintmatCheckSorted(
  TU_INTMAT* matrix /**< Int matrix */
);

/**
 * \brief Creates a submatrix of an int matrix as an explicit matrix.
 */
CMR_EXPORT
CMR_ERROR TUintmatFilterSubmat(
  TU* tu,               /**< \ref TU environment. */
  TU_INTMAT* matrix,    /**< Given matrix */
  TU_SUBMAT* submatrix, /**< Specified submatrix */
  TU_INTMAT** result    /**< Pointer for storing the resulting int matrix. */
);

/**
 * \brief Checks if int matrix has only entries in {0, 1}.
 */

CMR_EXPORT
bool TUisBinaryInt(
  TU* tu,                 /**< \ref TU environment. */
  TU_INTMAT* matrix,      /**< Int matrix */
  TU_SUBMAT** psubmatrix  /**< Pointer for storing a non-binary entry as a submatrix (may be \c NULL). */
);

/**
 * \brief Checks if int matrix has only entries in {-1, 0, +1}.
 */

CMR_EXPORT
bool TUisTernaryInt(
  TU* tu,                 /**< \ref TU environment. */
  TU_INTMAT* matrix,      /**< Int matrix */
  TU_SUBMAT** psubmatrix  /**< Pointer for storing a non-ternary entry as a submatrix (may be \c NULL). */
);

/**
 * \brief Creates the support matrix of an int \p matrix as a char matrix.
 */

CMR_EXPORT
CMR_ERROR TUsupportInt(
  TU* tu,               /**< \ref TU environment. */
  TU_INTMAT* matrix,    /**< Int matrix */
  TU_CHRMAT** psupport  /**< Pointer for storing the support matrix of \p matrix. */
);

/**
 * \brief Creates the signed support matrix of an int \p matrix as a char matrix.
 */

CMR_EXPORT
CMR_ERROR TUsignedSupportInt(
  TU* tu,               /**< \ref TU environment. */
  TU_INTMAT* matrix,    /**< Int matrix */
  TU_CHRMAT** psupport  /**< Pointer for storing the support matrix of \p matrix. */
);

















/**
 * \brief Creates a char matrix of size \p numRows times \p numColumns with \p numNonzeros
 *        nonzeros. The row starts and entries are allocated but not initialized.
 */

CMR_EXPORT
CMR_ERROR TUchrmatCreate(
  TU* tu,              /**< \ref TU environment. */
  TU_CHRMAT** matrix,  /**< Pointer for storing the created matrix. */
  int numRows,         /**< Number of rows. */
  int numColumns,      /**< Number of columns. */
  int numNonzeros      /**< Number of nonzeros. */
);

/**
 * \brief Frees the memory of an int matrix.
 */

CMR_EXPORT
CMR_ERROR TUchrmatFree(
  TU* tu,             /**< \ref TU environment. */
  TU_CHRMAT** matrix  /**< Pointer to matrix. */
);

/**
 * \brief Changes the number of nonzeros and reallocates corresponding arrays.
 */

CMR_EXPORT
CMR_ERROR TUchrmatChangeNumNonzeros(
  TU* tu,             /**< \ref TU environment. */
  TU_CHRMAT* matrix,  /**< Given matrix. */
  int newNumNonzeros  /**< New number of nonzeros. */ 
);

/**
 * \brief Copies an int matrix to a newly allocated one.
 * 
 * Allocates *\p result and copies \p matrix there.
 */
CMR_EXPORT
CMR_ERROR TUchrmatCopy(
  TU* tu,             /**< \ref TU environment. */
  TU_CHRMAT* matrix,  /**< Given matrix. */
  TU_CHRMAT** result  /**< Pointer to store a copy of \p matrix. */
);

/**
 * \brief Creates the transpose of an int matrix.
 */
CMR_EXPORT
CMR_ERROR TUchrmatTranspose(
  TU* tu,             /**< \ref TU environment. */
  TU_CHRMAT* matrix,  /**< Given matrix. */
  TU_CHRMAT** result  /**< Pointer to store the transpose of \p matrix. */
);



/**
 * \brief Prints a char matrix.
 */

CMR_EXPORT
CMR_ERROR TUchrmatPrintSparse(
  FILE* stream,       /**< File stream to print to. */
  TU_CHRMAT* matrix   /**< Char matrix. */
);



/**
 * \brief Prints a char matrix.
 */

CMR_EXPORT
CMR_ERROR TUchrmatPrintDense(
  FILE* stream,       /**< File stream to print to. */
  TU_CHRMAT* matrix,  /**< Char matrix. */
  char zeroChar,      /**< Character to print for a zero. */
  bool header         /**< Whether to print row and column indices. */
);



/**
 * \brief Reads a sparse char matrix from a file \p stream.
 *  
 * Zero entries are ignored, and multiple occurences of (row,column) pairs are considered as errors.
 * Returns \ref CMR_ERROR_INPUT in case of errors. In this case, *\p pmatrix will be \c NULL.
 */

CMR_EXPORT
CMR_ERROR TUchrmatCreateFromSparseStream(
  TU* tu,               /**< \ref TU environment. */
  TU_CHRMAT** pmatrix,  /**< Pointer for storing the matrix. */
  FILE* stream          /**< File stream to read from. */
);


/**
 * \brief Reads a densely stored char matrix from a file \p stream.
 */

CMR_EXPORT
CMR_ERROR TUchrmatCreateFromDenseStream(
  TU* tu,               /**< \ref TU environment. */
  TU_CHRMAT** pmatrix,  /**< Pointer for storing the matrix. */
  FILE* stream          /**< File stream to read from. */
);

/**
 * \brief Checks whether two char matrices are equal.
 */

CMR_EXPORT
bool TUchrmatCheckEqual(
  TU_CHRMAT* matrix1,  /**< First matrix */
  TU_CHRMAT* matrix2   /**< Second matrix */
);

/**
 * \brief Checks whether two char matrices are transposes of each other.
 */

CMR_EXPORT
bool TUchrmatCheckTranspose(
  TU_CHRMAT* matrix1, /**< First matrix */
  TU_CHRMAT* matrix2  /**< Second matrix */
);

/**
 * \brief Checks whether char matrix has each row sorted by minor.
 */

CMR_EXPORT
bool TUchrmatCheckSorted(
  TU_CHRMAT* matrix /**< Char matrix */
);

/**
 * \brief Creates a submatrix of a char matrix as an explicit matrix.
 */
CMR_EXPORT
CMR_ERROR TUchrmatFilterSubmat(
  TU* tu,               /**< \ref TU environment. */
  TU_CHRMAT* matrix,    /**< Given matrix */
  TU_SUBMAT* submatrix, /**< Specified submatrix */
  TU_CHRMAT** result    /**< Pointer for storing the resulting char matrix. */
);

/**
 * \brief Checks if matrix has only entries in {0, 1}.
 */

CMR_EXPORT
bool TUisBinaryChr(
  TU* tu,                 /**< \ref TU environment. */
  TU_CHRMAT* matrix,      /**< Char matrix */
  TU_SUBMAT** psubmatrix  /**< Pointer for storing a non-binary entry as a submatrix (may be \c NULL). */
);

/**
 * \brief Checks if char matrix has only entries in {-1, 0, +1}.
 */

CMR_EXPORT
bool TUisTernaryChr(
  TU* tu,                 /**< \ref TU environment. */
  TU_CHRMAT* matrix,      /**< Char matrix */
  TU_SUBMAT** psubmatrix  /**< Pointer for storing a non-ternary entry as a submatrix (may be \c NULL). */
);

/**
 * \brief Creates the support matrix of a char \p matrix as a char matrix.
 */

CMR_EXPORT
CMR_ERROR TUsupportChr(
  TU* tu,               /**< \ref TU environment. */
  TU_CHRMAT* matrix,    /**< Char matrix */
  TU_CHRMAT** psupport  /**< Pointer for storing the support matrix of \p matrix. */
);

/**
 * \brief Creates the signed support matrix of a char \p matrix as a char matrix.
 */

CMR_EXPORT
CMR_ERROR TUsignedSupportChr(
  TU* tu,               /**< \ref TU environment. */
  TU_CHRMAT* matrix,    /**< Char matrix */
  TU_CHRMAT** psupport  /**< Pointer for storing the support matrix of \p matrix. */
);

#ifdef __cplusplus
}
#endif

#endif /* TU_MATRIX_H */
