#ifndef LINALG_NAIVE_H
#define LINALG_NAIVE_H
#include <vector>
#include <iostream>
#include <utility>
#include <cmath>
namespace linalg_naive {
class Matrix {
public:
  Matrix(const Matrix &other) {
    data = other.getData();
  }

  Matrix(Matrix &&other) {
    data.swap(other.getData());
  }

  Matrix(int _Rows, int _Cols, double _val = 0.0) {
    data.assign(_Rows, std::vector<double>(_Cols, _val));
  }

  Matrix(const std::vector<std::vector<double>> &_data) {
    data = _data;
  }

  // Matrix(const boost::python::list &iterable);

  const int n_row() const {
    return data.size();
  }

  const int n_col() const {
    return data[0].size();
  }

  static Matrix eye(int _size) {
    /*Exception*/
    Matrix mt(_size, _size);
    auto &_mt_data = mt.getData();
    for (int i = 0; i < _size; ++i)
      _mt_data[i][i] = 1.0;
    return mt;
  }

   const std::vector<std::vector<double>> &getData() const {
     return data;
   }

   std::vector<std::vector<double>> &getData() {
     return data;
   }

  Matrix &operator=(const Matrix &other) {
    data = other.getData();
    return *this;
  }

  Matrix &operator=(Matrix &&other) {
    data.swap(other.getData());
    return *this;
  }

  friend std::ostream &operator << (std::ostream &output, const Matrix &mt) {
    auto &_mt_data = mt.getData();
    for (auto &row : _mt_data) {
      int sz = row.size();
      for (int j = 0; j < sz; ++j)
        output << row[j] << " \n"[j == sz - 1];
    }
    return output;
  }

  Matrix slice(int row_start, int row_end, int row_step, int col_start, int col_end, int col_step) {
    /*
    */ 
    int _new_n_row = (row_end - row_start + row_step - 1) / row_step;
    int _new_n_col = (col_end - col_start + col_step - 1) / col_step;
    Matrix mt(_new_n_row, _new_n_col);
    auto &_mt_data = mt.getData();
    for (int i = 0; i < _new_n_row; ++i)
      for (int j = 0; j < _new_n_col; ++j)
        _mt_data[i][j] = data[row_start + i * row_step][col_start + j * col_step];
    return mt;
  }

  Matrix transpose() const {
    int _n_row = n_row();
    int _n_col = n_col();
    Matrix mt(_n_col, _n_row);
    auto &_mt_data = mt.getData();
    for (int i = 0; i < _n_col; ++i)
      for (int j = 0; j < _n_row; ++j)
        _mt_data[i][j] = data[j][i];
    return mt;
  }

  Matrix reshape(int _row, int _col) {
    /* Exception
    */
    Matrix mt(_row, _col);
    auto &_mt_data = mt.getData();
    int _n_col = n_col();
    for (int i = 0; i < _row; ++i) {
      for (int j = 0; j < _col; ++j) {
        int tmp = i * _col + j;
        int _i = tmp / _n_col;
        int _j = tmp % _n_col;
        _mt_data[i][j] = data[_i][_j];
      }
    }
    return mt;
  }



protected:
  std::vector<std::vector<double>> data;
};

Matrix operator+(const Matrix &a, const Matrix &b) {
  int _n_row = a.n_row();
  int _n_col = a.n_col();
  /* MatrixSizeDismatchException
  */
  Matrix mt(_n_row, _n_col); 
  auto &_mt_data = mt.getData();
  auto &_a_data = a.getData();
  auto &_b_data = b.getData();
  for (int i = 0; i < _n_row; ++i)
    for (int j = 0; j < _n_col; ++j)
      _mt_data[i][j] = _a_data[i][j] + _b_data[i][j];
  return mt;
}

Matrix operator+(const Matrix &a, const double &val) {
  int _n_row = a.n_row();
  int _n_col = a.n_col();
  /* Exception
  */
  Matrix mt(_n_row, _n_col); 
  auto &_mt_data = mt.getData();
  auto &_a_data = a.getData();
  for (int i = 0; i < _n_row; ++i)
    for (int j = 0; j < _n_col; ++j)
      _mt_data[i][j] = _a_data[i][j] + val;
  return mt;
}

Matrix operator+(const double &val, const Matrix &a) {
  int _n_row = a.n_row();
  int _n_col = a.n_col();
  /* Exception
  */
  Matrix mt(_n_row, _n_col); 
  auto &_mt_data = mt.getData();
  auto &_a_data = a.getData();
  for (int i = 0; i < _n_row; ++i)
    for (int j = 0; j < _n_col; ++j)
      _mt_data[i][j] = _a_data[i][j] + val;
  return mt;
}

Matrix operator-(const Matrix &a, const Matrix &b) {
  int _n_row = a.n_row();
  int _n_col = a.n_col();
  /* Exception
  */
  Matrix mt(_n_row, _n_col); 
  auto &_mt_data = mt.getData();
  auto &_a_data = a.getData();
  auto &_b_data = b.getData();
  for (int i = 0; i < _n_row; ++i)
    for (int j = 0; j < _n_col; ++j)
      _mt_data[i][j] = _a_data[i][j] - _b_data[i][j];
  return mt;
}

Matrix operator-(const Matrix &a, const double &val) {
  int _n_row = a.n_row();
  int _n_col = a.n_col();
  /* Exception
  */
  Matrix mt(_n_row, _n_col); 
  auto &_mt_data = mt.getData();
  auto &_a_data = a.getData();
  for (int i = 0; i < _n_row; ++i)
    for (int j = 0; j < _n_col; ++j)
      _mt_data[i][j] = _a_data[i][j] - val;
  return mt;
}

Matrix operator-(const double &val, const Matrix &a) {
  int _n_row = a.n_row();
  int _n_col = a.n_col();
  /* Exception
  */
  Matrix mt(_n_row, _n_col); 
  auto &_mt_data = mt.getData();
  auto &_a_data = a.getData();
  for (int i = 0; i < _n_row; ++i)
    for (int j = 0; j < _n_col; ++j)
      _mt_data[i][j] = val - _a_data[i][j];
  return mt;
}

Matrix operator*(const Matrix &a, const Matrix &b) {
  int _n_row = a.n_row();
  int _n_col = a.n_col();
  /* Exception
  */
  Matrix mt(_n_row, _n_col); 
  auto &_mt_data = mt.getData();
  auto &_a_data = a.getData();
  auto &_b_data = b.getData();
  for (int i = 0; i < _n_row; ++i)
    for (int j = 0; j < _n_col; ++j)
      _mt_data[i][j] = _a_data[i][j] * _b_data[i][j];
  return mt;
}

Matrix operator*(const Matrix &a, const double &val) {
  int _n_row = a.n_row();
  int _n_col = a.n_col();
  /* Exception
  */
  Matrix mt(_n_row, _n_col); 
  auto &_mt_data = mt.getData();
  auto &_a_data = a.getData();
  for (int i = 0; i < _n_row; ++i)
    for (int j = 0; j < _n_col; ++j)
      _mt_data[i][j] = _a_data[i][j] * val;
  return mt;
}

Matrix operator*(const double &val, const Matrix &a) {
  int _n_row = a.n_row();
  int _n_col = a.n_col();
  /* Exception
  */
  Matrix mt(_n_row, _n_col); 
  auto &_mt_data = mt.getData();
  auto &_a_data = a.getData();
  for (int i = 0; i < _n_row; ++i)
    for (int j = 0; j < _n_col; ++j)
      _mt_data[i][j] = val * _a_data[i][j];
  return mt;
}

Matrix operator/(const Matrix &a, const Matrix &b) {
  int _n_row = a.n_row();
  int _n_col = a.n_col();
  /* Exception
  */
  Matrix mt(_n_row, _n_col); 
  auto &_mt_data = mt.getData();
  auto &_a_data = a.getData();
  auto &_b_data = b.getData();
  for (int i = 0; i < _n_row; ++i)
    for (int j = 0; j < _n_col; ++j)
      _mt_data[i][j] = _a_data[i][j] / _b_data[i][j];
  return mt;
}

Matrix operator/(const Matrix &a, const double &val) {
  int _n_row = a.n_row();
  int _n_col = a.n_col();
  /* Exception
  */
  Matrix mt(_n_row, _n_col); 
  auto &_mt_data = mt.getData();
  auto &_a_data = a.getData();
  for (int i = 0; i < _n_row; ++i)
    for (int j = 0; j < _n_col; ++j)
      _mt_data[i][j] = _a_data[i][j] / val;
  return mt;
}

Matrix operator/(const double &val, const Matrix &a) {
  int _n_row = a.n_row();
  int _n_col = a.n_col();
  /* Exception
  */
  Matrix mt(_n_row, _n_col); 
  auto &_mt_data = mt.getData();
  auto &_a_data = a.getData();
  for (int i = 0; i < _n_row; ++i)
    for (int j = 0; j < _n_col; ++j)
      _mt_data[i][j] = val / _a_data[i][j];
  return mt;
}

Matrix operator-(const Matrix &a) {
  int _n_row = a.n_row();
  int _n_col = a.n_col();
  Matrix mt(_n_row, _n_col); 
  auto &_mt_data = mt.getData();
  auto &_a_data = a.getData();
  for (int i = 0; i < _n_row; ++i)
    for (int j = 0; j < _n_col; ++j)
      _mt_data[i][j] = -_a_data[i][j];
  return mt;
}

Matrix matmul(const Matrix &a, const Matrix &b) {
  int _a_row = a.n_row();
  int _a_col = a.n_col();
  int _b_row = b.n_row();
  int _b_col = b.n_col();
  /* Exception
  */
  Matrix mt(_a_row, _b_col);
  auto &_mt_data = mt.getData();
  auto &_a_data = a.getData();
  auto &_b_data = b.getData();
  for (int i = 0; i < _a_row; ++i) {
    for (int j = 0; j < _b_col; ++j) {
      double val = 0;
      for (int k = 0; k < _a_col; ++k) {
        val += _a_data[i][k] * _b_data[k][j];
      }
      _mt_data[i][j] = val;
    }
  }
  return mt;
}
 
double sum(const Matrix &mt) {
  auto &_mt_data = mt.getData();
  double res = 0;
  for (auto &row : _mt_data)
    for (auto &val : row)
      res += val;
  return res;
}

Matrix sum(const Matrix &mt, int row_major) {
  auto &_mt_data = mt.getData();
  int n_rows = mt.n_row();
  int n_cols = mt.n_col();
  if (row_major) {
    Matrix ret(n_rows, 1);
    auto &_ret_data = ret.getData();
    for (int i = 0; i < n_rows; ++i) {
      double sum = 0;
      for (auto &val : _mt_data[i]) sum += val;
      _ret_data[i][0] = sum;
    }
    return std::move(ret);
  } else {
    Matrix ret(1, n_cols);
    auto &_ret_data = ret.getData();
    for (int i = 0; i < n_cols; ++i) {
      double sum = 0;
      for (int j = 0; j < n_rows; ++j)  sum += _mt_data[j][i];
      _ret_data[0][i] = sum;
    }
    return std::move(ret);
  }
}

double mean(const Matrix &mt) {
  /* Empty_Matrix_Exception
  */
  auto &_mt_data = mt.getData();
  int n_rows = _mt_data.size();
  int n_cols = _mt_data[0].size();
  double sum = 0;
  for (auto &_row : _mt_data) {
    for (auto val : _row) {
      sum += val;
    }
  }
  return sum / (n_rows * n_cols);
}
  
Matrix mean(const Matrix &mt, int row_major) {
  auto &_mt_data = mt.getData();
  int n_rows = _mt_data.size();
  int n_cols = _mt_data[0].size();
  if (row_major) {
    Matrix ret(n_rows, 1);
    auto &_ret_data = ret.getData();
    for (int i = 0; i < n_rows; ++i) {
      double sum = 0;
      for (auto &val : _mt_data[i]) sum += val;
      _ret_data[i][0] = sum / n_cols;
    }
    return std::move(ret);
  } else {
    Matrix ret(1, n_cols);
    auto &_ret_data = ret.getData();
    for (int i = 0; i < n_cols; ++i) {
      double sum = 0;
      for (int j = 0; j < n_rows; ++j)  sum += _mt_data[j][i];
      _ret_data[0][i] = sum / n_rows;
    }
    return std::move(ret);
  }
}
 
Matrix tile(const Matrix &origin, int row_count, int col_count) {
  auto &_origin_data = origin.getData();
  int _n_row = origin.n_row();
  int _n_col = origin.n_col();
  int _new_n_row = _n_row * row_count;
  int _new_n_col = _n_col * col_count;
  Matrix mt(_new_n_row, _new_n_col);
  auto &_mt_data = mt.getData();
  for (int i = 0; i < _new_n_row; ++i)
    for (int j = 0; j < _new_n_col; ++j)
      _mt_data[i][j] = _origin_data[i % _n_row][j % _n_col];
  return mt;
}

Matrix inv(const Matrix &origin) {
  /*Exception*/
  int n = origin.n_row();
  Matrix res(n, n);
  std::vector<std::vector<double>> a(origin.getData());
  auto &_res_data = res.getData();
  for (int i = 0; i < n; ++i)
    _res_data[i][i] = 1;
  for (int k = 0; k < n; ++k) {
    int pivot = k;
    for (int i = k + 1; i < n; ++i) {
      if (fabs(a[i][k]) > fabs(a[pivot][k]))
        pivot = i;
    }
    if (pivot != k) {
      a[k].swap(a[pivot]);
      _res_data[k].swap(_res_data[pivot]);
    }
    if (fabs(a[k][k]) < 1e-6) {
      puts("1");
    }
    /* NotInvertibleException
    */
    for (int i = k + 1; i < n; ++i) {
      a[k][i] /= a[k][k];
    }
    for (int i = 0; i < n; ++i) {
      _res_data[k][i] /= a[k][k];
    }
    a[k][k] = 1;
    for (int i = k + 1; i < n; ++i) {
      double tmp = a[i][k];
      if (fabs(tmp) > 1e-7) {
        for (int j = k; j < n; ++j) {
          a[i][j] -= a[k][j] * tmp;
        }
        for (int j = 0; j < n; ++j) {
          _res_data[i][j] -= _res_data[k][j] * tmp;
        }
      }
    }
  }
  for (int k = n - 1; k; --k) {
    for (int i = 0; i < k; ++i) {
      double tmp = a[i][k];
      for (int j = 0; j < n; ++j) {
        _res_data[i][j] -= _res_data[k][j] * tmp;
      }
      for (int j = k; j < n; ++j) {
        a[i][j] -= a[k][j] * tmp;
      }
    }
  }
  return res;
}
}
#endif
