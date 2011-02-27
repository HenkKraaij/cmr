#include <Integer.h>
#include <Matrix.h>
#include <iostream>

#include "total_unimodularity.hpp"
#include "unimodularity.hpp"

namespace polymake
{
  namespace common
  {

    bool is_totally_unimodular_plain(const Matrix <Integer>& matrix)
    {
      unimod::integer_matrix input_matrix(matrix.rows(), matrix.cols());
      for (size_t r = 0; r < input_matrix.size1(); ++r)
      {
        for (size_t c = 0; c < input_matrix.size2(); ++c)
        {
          input_matrix(r, c) = matrix(r, c);
        }
      }

      return unimod::is_totally_unimodular(input_matrix);
    }

    bool is_totally_unimodular_violator(const Matrix <Integer>& matrix, Vector <Integer>& rows, Vector <Integer>& columns)
    {
      unimod::integer_matrix input_matrix(matrix.rows(), matrix.cols());
      for (size_t r = 0; r < input_matrix.size1(); ++r)
      {
        for (size_t c = 0; c < input_matrix.size2(); ++c)
          input_matrix(r, c) = matrix(r, c);
      }

      unimod::submatrix_indices indices;

      bool result = unimod::is_totally_unimodular(input_matrix, indices);
      if (!result)
      {
        rows.resize(indices.rows.size());
        for (size_t i = 0; i < indices.rows.size(); ++i)
          rows[i] = (int) indices.rows[i];
        columns.resize(indices.columns.size());
        for (size_t i = 0; i < indices.columns.size(); ++i)
          columns[i] = (int) indices.columns[i];
      }
      return result;
    }

    bool is_unimodular_plain(const Matrix <Integer>& matrix)
    {
      size_t rank;
      unimod::integer_matrix input_matrix(matrix.rows(), matrix.cols());
      for (size_t r = 0; r < input_matrix.size1(); ++r)
      {
        for (size_t c = 0; c < input_matrix.size2(); ++c)
        {
          input_matrix(r, c) = matrix(r, c);
        }
      }

      return unimod::is_unimodular(input_matrix, rank);
    }

    bool is_strongly_unimodular_plain(const Matrix <Integer>& matrix)
    {
      size_t rank;
      unimod::integer_matrix input_matrix(matrix.rows(), matrix.cols());
      for (size_t r = 0; r < input_matrix.size1(); ++r)
      {
        for (size_t c = 0; c < input_matrix.size2(); ++c)
        {
          input_matrix(r, c) = matrix(r, c);
        }
      }

      return unimod::is_strongly_unimodular(input_matrix, rank);
    }

    bool is_k_modular_plain(const Matrix <Integer>& matrix)
    {
      size_t rank;
      unimod::integer_matrix input_matrix(matrix.rows(), matrix.cols());
      for (size_t r = 0; r < input_matrix.size1(); ++r)
      {
        for (size_t c = 0; c < input_matrix.size2(); ++c)
        {
          input_matrix(r, c) = matrix(r, c);
        }
      }

      return unimod::is_k_modular(input_matrix, rank);
    }

    bool is_k_modular_compute(const Matrix <Integer>& matrix, int& ko)
    {
      size_t rank;
      unsigned int k;
      bool result;
      unimod::integer_matrix input_matrix(matrix.rows(), matrix.cols());
      for (size_t r = 0; r < input_matrix.size1(); ++r)
      {
        for (size_t c = 0; c < input_matrix.size2(); ++c)
        {
          input_matrix(r, c) = matrix(r, c);
        }
      }

      result = unimod::is_k_modular(input_matrix, rank, k);
      ko = (int) k;
      return result;
    }

    bool is_strongly_k_modular_plain(const Matrix <Integer>& matrix)
    {
      size_t rank;
      unimod::integer_matrix input_matrix(matrix.rows(), matrix.cols());
      for (size_t r = 0; r < input_matrix.size1(); ++r)
      {
        for (size_t c = 0; c < input_matrix.size2(); ++c)
        {
          input_matrix(r, c) = matrix(r, c);
        }
      }

      return unimod::is_strongly_k_modular(input_matrix, rank);
    }

  UserFunction4perl("# Tests a given //matrix// for total unimodularity, without certificates."
      "# @param Matrix<int> matrix"
      "# @return Bool",
      &is_totally_unimodular_plain,"is_totally_unimodular(Matrix<Integer>)");
  UserFunction4perl("# Tests a given //matrix// for total unimodularity, setting row/column indices to a violating submatrix if the result is negative."
      "# @param Matrix<int> matrix"
      "# @param Vector<int> row_indices of violating submatrix."
      "# @param Vector<int> column_indices of violating submatrix."
      "# @return Bool",
      &is_totally_unimodular_violator,"is_totally_unimodular(Matrix<Integer>,Vector<Integer>,Vector<Integer>)");
  UserFunction4perl("# Tests a given //matrix// for unimodularity, without certificates."
      "# @param Matrix<int> matrix"
      "# @return Bool",
      &is_unimodular_plain,"is_unimodular(Matrix<Integer>)");
  UserFunction4perl("# Tests a given //matrix// for strong unimodularity, without certificates."
      "# @param Matrix<int> matrix"
      "# @return Bool",
      &is_strongly_unimodular_plain,"is_strongly_unimodular(Matrix<Integer>)");
  UserFunction4perl("# Tests a given //matrix// for k-modularity, without certificates."
      "# @param Matrix<int> matrix"
      "# @return Bool",
      &is_k_modular_plain,"is_k_modular(Matrix<Integer>)");
  UserFunction4perl("# Tests a given //matrix// for k-modularity, without certificates."
      "# @param Matrix<int> matrix"
      "# @param int& k"
      "# @return Bool",
      &is_k_modular_compute,"is_k_modular(Matrix<Integer>, $)");
  UserFunction4perl("# Tests a given //matrix// for strong k-modularity, without certificates."
      "# @param Matrix<int> matrix"
      "# @return Bool",
      &is_strongly_k_modular_plain,"is_strongly_k_modular(Matrix<Integer>)");
}}

