#ifndef LINALG_H
#define LINALG_H
#include <vector>
#include <iostream>
#include <utility>
#include <cmath>
#include "simd.h"
namespace linalg {
class Matrix {
public:
  Matrix(const Matrix &other) {
    data = other.getData();
  }

  Matrix(Matrix &&other) {
    data.swap(other.getData());
  }

  Matrix(int _Rows, int _Cols, double _val = 0.0) {
    /*Exception*/
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
#pragma omp parallel for
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
#pragma omp parallel for
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
  Matrix res(_n_row, _n_col); 
  auto &_res_data = res.getData();
  auto &_a_data = a.getData();
  auto &_b_data = b.getData();
#pragma omp parallel for
  for (int i = 0; i < _n_row; ++i) {
    const double *p_a = _a_data[i].data();
    const double *p_b = _b_data[i].data();
    double *p_res = _res_data[i].data();
    int num_block = _n_col / DVEC_SIZE;
    int num_rem = _n_col % DVEC_SIZE;
    for (int j = 0; j < num_block; ++j) {
      vec_store(p_res, vec_add(vec_load(p_a), vec_load(p_b)));
      p_a += DVEC_SIZE;
      p_b += DVEC_SIZE;
      p_res += DVEC_SIZE;
    }
    for (int j = 0; j < num_rem; ++j) {
      p_res[j] = p_a[j] + p_b[j];
    }
  }
  return res;
}

Matrix operator+(const Matrix &a, const double &val) {
  int _n_row = a.n_row();
  int _n_col = a.n_col();
  Matrix res(_n_row, _n_col); 
  auto &_res_data = res.getData();
  auto &_a_data = a.getData();
#pragma omp parallel for
  for (int i = 0; i < _n_row; ++i) {
    const double *p_a = _a_data[i].data();
    double *p_res = _res_data[i].data();
    int num_block = _n_col / DVEC_SIZE;
    int num_rem = _n_col % DVEC_SIZE;
    dvec val_vec = dvec_set1(val);
    for (int j = 0; j < num_block; ++j) {
      vec_store(p_res, vec_add(vec_load(p_a), val_vec));
      p_a += DVEC_SIZE;
      p_res += DVEC_SIZE;
    }
    for (int j = 0; j < num_rem; ++j) {
      p_res[j] = p_a[j] + val;
    }
  }
  return res;
}

Matrix operator+(const double &val, const Matrix &a) {
  int _n_row = a.n_row();
  int _n_col = a.n_col();
  Matrix res(_n_row, _n_col); 
  auto &_res_data = res.getData();
  auto &_a_data = a.getData();
#pragma omp parallel for
  for (int i = 0; i < _n_row; ++i) {
    const double *p_a = _a_data[i].data();
    double *p_res = _res_data[i].data();
    int num_block = _n_col / DVEC_SIZE;
    int num_rem = _n_col % DVEC_SIZE;
    dvec val_vec = dvec_set1(val);
    for (int j = 0; j < num_block; ++j) {
      vec_store(p_res, vec_add(vec_load(p_a), val_vec));
      p_a += DVEC_SIZE;
      p_res += DVEC_SIZE;
    }
    for (int j = 0; j < num_rem; ++j) {
      p_res[j] = p_a[j] + val;
    }
  }
  return res;
}

Matrix operator-(const Matrix &a, const Matrix &b) {
  int _n_row = a.n_row();
  int _n_col = a.n_col();
  /* MatrixSizeDismatchException
  */
  Matrix res(_n_row, _n_col); 
  auto &_res_data = res.getData();
  auto &_a_data = a.getData();
  auto &_b_data = b.getData();
#pragma omp parallel for
  for (int i = 0; i < _n_row; ++i) {
    const double *p_a = _a_data[i].data();
    const double *p_b = _b_data[i].data();
    double *p_res = _res_data[i].data();
    int num_block = _n_col / DVEC_SIZE;
    int num_rem = _n_col % DVEC_SIZE;
    for (int j = 0; j < num_block; ++j) {
      vec_store(p_res, vec_sub(vec_load(p_a), vec_load(p_b)));
      p_a += DVEC_SIZE;
      p_b += DVEC_SIZE;
      p_res += DVEC_SIZE;
    }
    for (int j = 0; j < num_rem; ++j) {
      p_res[j] = p_a[j] - p_b[j];
    }
  }
  return res;
}

Matrix operator-(const Matrix &a, const double &val) {
  int _n_row = a.n_row();
  int _n_col = a.n_col();
  Matrix res(_n_row, _n_col); 
  auto &_res_data = res.getData();
  auto &_a_data = a.getData();
#pragma omp parallel for
  for (int i = 0; i < _n_row; ++i) {
    const double *p_a = _a_data[i].data();
    double *p_res = _res_data[i].data();
    int num_block = _n_col / DVEC_SIZE;
    int num_rem = _n_col % DVEC_SIZE;
    dvec val_vec = dvec_set1(val);
    for (int j = 0; j < num_block; ++j) {
      dvec sum = vec_sub(vec_load(p_a), val_vec);
      vec_store(p_res, sum);
      p_a += DVEC_SIZE;
      p_res += DVEC_SIZE;
    }
    for (int j = 0; j < num_rem; ++j) {
      p_res[j] = p_a[j] - val;
    }
  }
  return res;
}

Matrix operator-(const double &val, const Matrix &a) {
  int _n_row = a.n_row();
  int _n_col = a.n_col();
  Matrix res(_n_row, _n_col); 
  auto &_res_data = res.getData();
  auto &_a_data = a.getData();
#pragma omp parallel for
  for (int i = 0; i < _n_row; ++i) {
    const double *p_a = _a_data[i].data();
    double *p_res = _res_data[i].data();
    int num_block = _n_col / DVEC_SIZE;
    int num_rem = _n_col % DVEC_SIZE;
    dvec val_vec = dvec_set1(val);
    for (int j = 0; j < num_block; ++j) {
      vec_store(p_res, vec_sub(val_vec, vec_load(p_a)));
      p_a += DVEC_SIZE;
      p_res += DVEC_SIZE;
    }
    for (int j = 0; j < num_rem; ++j) {
      p_res[j] = val - p_a[j];
    }
  }
  return res;
}

Matrix operator*(const Matrix &a, const Matrix &b) {
  int _n_row = a.n_row();
  int _n_col = a.n_col();
  /* MatrixSizeDismatchException
  */
  Matrix res(_n_row, _n_col); 
  auto &_res_data = res.getData();
  auto &_a_data = a.getData();
  auto &_b_data = b.getData();
#pragma omp parallel for
  for (int i = 0; i < _n_row; ++i) {
    const double *p_a = _a_data[i].data();
    const double *p_b = _b_data[i].data();
    double *p_res = _res_data[i].data();
    int num_block = _n_col / DVEC_SIZE;
    int num_rem = _n_col % DVEC_SIZE;
    for (int j = 0; j < num_block; ++j) {
      vec_store(p_res, vec_mul(vec_load(p_a), vec_load(p_b)));
      p_a += DVEC_SIZE;
      p_b += DVEC_SIZE;
      p_res += DVEC_SIZE;
    }
    for (int j = 0; j < num_rem; ++j) {
      p_res[j] = p_a[j] * p_b[j];
    }
  }
  return res;
}

Matrix operator*(const Matrix &a, const double &val) {
  int _n_row = a.n_row();
  int _n_col = a.n_col();
  Matrix res(_n_row, _n_col); 
  auto &_res_data = res.getData();
  auto &_a_data = a.getData();
#pragma omp parallel for
  for (int i = 0; i < _n_row; ++i) {
    const double *p_a = _a_data[i].data();
    double *p_res = _res_data[i].data();
    int num_block = _n_col / DVEC_SIZE;
    int num_rem = _n_col % DVEC_SIZE;
    dvec val_vec = dvec_set1(val);
    for (int j = 0; j < num_block; ++j) {
      dvec sum = vec_mul(vec_load(p_a), val_vec);
      vec_store(p_res, sum);
      p_a += DVEC_SIZE;
      p_res += DVEC_SIZE;
    }
    for (int j = 0; j < num_rem; ++j) {
      p_res[j] = p_a[j] * val;
    }
  }
  return res;
}

Matrix operator*(const double &val, const Matrix &a) {
  int _n_row = a.n_row();
  int _n_col = a.n_col();
  Matrix res(_n_row, _n_col); 
  auto &_res_data = res.getData();
  auto &_a_data = a.getData();
#pragma omp parallel for
  for (int i = 0; i < _n_row; ++i) {
    const double *p_a = _a_data[i].data();
    double *p_res = _res_data[i].data();
    int num_block = _n_col / DVEC_SIZE;
    int num_rem = _n_col % DVEC_SIZE;
    dvec val_vec = dvec_set1(val);
    for (int j = 0; j < num_block; ++j) {
      dvec sum = vec_mul(vec_load(p_a), val_vec);
      vec_store(p_res, sum);
      p_a += DVEC_SIZE;
      p_res += DVEC_SIZE;
    }
    for (int j = 0; j < num_rem; ++j) {
      p_res[j] = p_a[j] * val;
    }
  }
  return res;
}

Matrix operator/(const Matrix &a, const Matrix &b) {
  int _n_row = a.n_row();
  int _n_col = a.n_col();
  /* MatrixSizeDismatchException
  */
  Matrix res(_n_row, _n_col); 
  auto &_res_data = res.getData();
  auto &_a_data = a.getData();
  auto &_b_data = b.getData();
#pragma omp parallel for
  for (int i = 0; i < _n_row; ++i) {
    const double *p_a = _a_data[i].data();
    const double *p_b = _b_data[i].data();
    double *p_res = _res_data[i].data();
    int num_block = _n_col / DVEC_SIZE;
    int num_rem = _n_col % DVEC_SIZE;
    for (int j = 0; j < num_block; ++j) {
      vec_store(p_res, vec_div(vec_load(p_a), vec_load(p_b)));
      p_a += DVEC_SIZE;
      p_b += DVEC_SIZE;
      p_res += DVEC_SIZE;
    }
    for (int j = 0; j < num_rem; ++j) {
      p_res[j] = p_a[j] / p_b[j];
    }
  }
  return res;
}

Matrix operator/(const Matrix &a, const double &val) {
  int _n_row = a.n_row();
  int _n_col = a.n_col();
  Matrix res(_n_row, _n_col); 
  auto &_res_data = res.getData();
  auto &_a_data = a.getData();
#pragma omp parallel for
  for (int i = 0; i < _n_row; ++i) {
    const double *p_a = _a_data[i].data();
    double *p_res = _res_data[i].data();
    int num_block = _n_col / DVEC_SIZE;
    int num_rem = _n_col % DVEC_SIZE;
    dvec val_vec = dvec_set1(val);
    for (int j = 0; j < num_block; ++j) {
      vec_store(p_res, vec_div(vec_load(p_a), val_vec));
      p_a += DVEC_SIZE;
      p_res += DVEC_SIZE;
    }
    for (int j = 0; j < num_rem; ++j) {
      p_res[j] = p_a[j] / val;
    }
  }
  return res;
}

Matrix operator/(const double &val, const Matrix &a) {
  int _n_row = a.n_row();
  int _n_col = a.n_col();
  /* Exception
  */
  Matrix res(_n_row, _n_col); 
  auto &_res_data = res.getData();
  auto &_a_data = a.getData();
#pragma omp parallel for
  for (int i = 0; i < _n_row; ++i) {
    const double *p_a = _a_data[i].data();
    double *p_res = _res_data[i].data();
    int num_block = _n_col / DVEC_SIZE;
    int num_rem = _n_col % DVEC_SIZE;
    dvec val_vec = dvec_set1(val);
    for (int j = 0; j < num_block; ++j) {
      vec_store(p_res, vec_div(val_vec, vec_load(p_a)));
      p_a += DVEC_SIZE;
      p_res += DVEC_SIZE;
    }
    for (int j = 0; j < num_rem; ++j) {
      p_res[j] = val / p_a[j];
    }
  }
  return res;
}

Matrix operator-(const Matrix &a) {
  int _n_row = a.n_row();
  int _n_col = a.n_col();
  Matrix res(_n_row, _n_col); 
  auto &_res_data = res.getData();
  auto &_a_data = a.getData();
#pragma omp parallel for
  for (int i = 0; i < _n_row; ++i) {
    const double *p_a = _a_data[i].data();
    double *p_res = _res_data[i].data();
    int num_block = _n_col / DVEC_SIZE;
    int num_rem = _n_col % DVEC_SIZE;
    for (int j = 0; j < num_block; ++j) {
      vec_store(p_res, vec_sub(dvec_setzero(), vec_load(p_a)));
      p_a += DVEC_SIZE;
      p_res += DVEC_SIZE;
    }
    for (int j = 0; j < num_rem; ++j) {
      p_res[j] = -p_a[j];
    }
  }
  return res;
}

Matrix matmul(const Matrix &a, const Matrix &b) {
  int _a_row = a.n_row();
  int _a_col = a.n_col();
  int _b_row = b.n_row();
  int _b_col = b.n_col();
  /* MatrixScaleNotMatchedException
  */
  Matrix res(_a_row, _b_col);
  auto &_res_data = res.getData();
  auto &_a_data = a.getData();
  auto &_b_data = b.getData();
  std::vector<std::vector<double>> _b_transpose(_b_col, std::vector<double>(_b_row));
  for (int i = 0; i < _b_col; ++i) {
    for (int j = 0; j < _b_row; ++j) {
      _b_transpose[i][j] = _b_data[j][i];
    }
  }
#pragma omp parallel for
  for (int i = 0; i < _a_row; ++i) {
    for (int j = 0; j < _b_col; ++j) {
      int num_block = _a_col / DVEC_SIZE;
      int num_rem = _a_col % DVEC_SIZE;
      const double *p_a = _a_data[i].data();
      const double *p_b = _b_transpose[j].data();
      dvec sum = dvec_setzero();
      for (int k = 0; k < num_block; ++k) {
        sum = vec_add(sum, vec_mul(vec_load(p_a), vec_load(p_b)));
        p_a += DVEC_SIZE;
        p_b += DVEC_SIZE;
      }
      _res_data[i][j] = sum[0] + sum[1] + sum[2] + sum[3];
      for (int k = 0; k < num_rem; ++k) {
        _res_data[i][j] += p_a[k] * p_b[k];
      }
    }
  }
  return res;
}
 
double sum(const Matrix &mt) {
  /* Empty_Matrix_Exception
  */
  auto &_mt_data = mt.getData();
  int n_rows = _mt_data.size();
  int n_cols = _mt_data[0].size();
  std::vector<double> sum_of_row(n_rows);
  int num_block = n_cols / DVEC_SIZE;
  int num_rem = n_cols % DVEC_SIZE;
#pragma omp parallel for
  for (int i = 0; i < n_rows; ++i) {
    dvec sum = dvec_setzero();
    const double *p = _mt_data[i].data();
    for (int j = 0; j < num_block; ++j) {
      sum = vec_add(sum, vec_load(p));
      p += DVEC_SIZE;
    }
    sum_of_row[i] = sum[0] + sum[1] + sum[2] + sum[3];
    for (int j = 0; j < num_rem; ++j) {
      sum_of_row[i] += p[j];
    }
  }
  double sum = 0;
  for (int i = 0; i < n_rows; ++i)
    sum += sum_of_row[i];
  return sum;
}

Matrix sum(const Matrix &mt, int row_major) {
  auto &_mt_data = mt.getData();
  int n_rows = mt.n_row();
  int n_cols = mt.n_col();
  if (row_major) {
    Matrix res(n_rows, 1);
    auto &_res_data = res.getData();
    int num_block = n_cols / DVEC_SIZE;
    int num_rem = n_cols % DVEC_SIZE;
#pragma omp parallel for
    for (int i = 0; i < n_rows; ++i) {
      dvec sum = dvec_setzero();
      const double *p = _mt_data[i].data();
      for (int j = 0; j < num_block; ++j) {
        sum = vec_add(sum, vec_load(p));
        p += DVEC_SIZE;
      }
      _res_data[i][0] = sum[0] + sum[1] + sum[2] + sum[3];
      for (int j = 0; j < num_rem; ++j) {
        _res_data[i][0] += p[j];
      }
    }
    return res;
  } else {
    Matrix res(1, n_cols);
    auto &_res_data = res.getData()[0];
    int num_block = n_cols / DVEC_SIZE;
    int num_rem = n_cols % DVEC_SIZE;
#pragma omp paralleo for reduction(+:_res_data[0: n])
    for (int i = 0; i < n_rows; ++i) {
      double *p_res = _res_data.data();
      const double *p_mt = _mt_data[i].data();
      for (int j = 0; j < num_block; ++j) {
        vec_store(p_res, vec_add(vec_load(p_mt), vec_load(p_res)));
        p_res += DVEC_SIZE;
        p_mt += DVEC_SIZE;
      }
      for (int j = 0; j < num_rem; ++j) {
        p_res[j] += p_mt[j];
      }
    }
    return res;
  }
}

double mean(const Matrix &mt) {
  /* Empty_Matrix_Exception
  */
  auto &_mt_data = mt.getData();
  int n_rows = _mt_data.size();
  int n_cols = _mt_data[0].size();
  return sum(mt) / (n_rows * n_cols);
}
  
Matrix mean(const Matrix &mt, int row_major) {
  auto &_mt_data = mt.getData();
  int n_rows = _mt_data.size();
  int n_cols = _mt_data[0].size();
  Matrix ret = sum(mt, row_major);
  return ret / (row_major ? (n_cols) : (n_rows));
}
 
Matrix tile(const Matrix &mt, int row_count, int col_count) {
  auto &_mt_data = mt.getData();
  int _n_row = mt.n_row();
  int _n_col = mt.n_col();
  int _new_n_row = _n_row * row_count;
  int _new_n_col = _n_col * col_count;

  Matrix res(_new_n_row, _new_n_col);
  auto &_res_data = res.getData();
#pragma omp parallel for
  for (int i = 0; i < _new_n_row; ++i) {
    for (int cc = 0; cc < col_count; ++cc) {
      int num_block = _n_col / DVEC_SIZE;
      int num_rem = _n_col % DVEC_SIZE;
      double *p_res = _res_data[i].data() + cc * _n_col;
      const double *p_mt = _mt_data[i % _n_row].data();
      for (int j = 0; j < num_block; ++j) {
        vec_store(p_res, vec_load(p_mt));
        p_res += DVEC_SIZE;
        p_mt += DVEC_SIZE;
      }
      for (int j = 0; j < num_rem; ++j) {
        p_res[j] = p_mt[j];
      }
    }
  }
  return res;
}

Matrix inv(const Matrix &mt) {
  /*Exception*/
  int n = mt.n_row();
  auto &_mt_data = mt.getData();
  std::vector<std::vector<double>> aug_data(n, std::vector<double>(n * 2, 0));
  for (int i = 0; i < n; ++i)
    aug_data[i][i + n] = 1;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      aug_data[i][j] = _mt_data[i][j];
    }
  }
  // for (int i = 0; i < n; ++i) {
  //   for (int j = 0; j < n * 2; ++j) {
  //     printf("%10.4f", aug_data[i][j]);
  //   }
  //   puts("");
  // }
  // puts("");
  for (int k = 0; k < n; ++k) {
    int pivot = k;
    for (int i = k + 1; i < n; ++i) {
      if (fabs(aug_data[i][k]) > fabs(aug_data[pivot][k]))
        pivot = i;
    }
    if (pivot != k) {
      aug_data[k].swap(aug_data[pivot]);
    }
    if (fabs(aug_data[k][k]) < 1e-6) {
      puts("1");
    }
    /* NotInvertibleException
    */
    for (int i = k + 1; i < n * 2; ++i) {
      aug_data[k][i] /= aug_data[k][k];
    }
    aug_data[k][k] = 1;
#pragma omp parallel for
    for (int i = k + 1; i < n; ++i) {
      double tmp = aug_data[i][k];
      if (fabs(tmp) > 1e-7) {
        int num_block = (n * 2 - k) / DVEC_SIZE;
        int num_rem = (n * 2 - k) % DVEC_SIZE;
        double *p1 = &aug_data[i][k];
        const double *p2 = &aug_data[k][k];
        dvec tmp_vec = dvec_set1(tmp);
        for (int j = 0; j < num_block; ++j) {
          vec_store(p1, vec_sub(vec_load(p1), vec_mul(vec_load(p2), tmp_vec)));
          p1 += DVEC_SIZE;
          p2 += DVEC_SIZE;
        }
        for (int j = 0; j < num_rem; ++j) {
          p1[j] -= p2[j] * tmp;
        }
      }
    }
    // for (int i = 0; i < n; ++i) {
    //   for (int j = 0; j < n * 2; ++j) {
    //     printf("%10.4f", aug_data[i][j]);
    //   }
    //   puts("");
    // }
    // puts("");
  }
  for (int k = n - 1; k; --k) {
#pragma omp parallel for
    for (int i = 0; i < k; ++i) {
      double tmp = aug_data[i][k];
      int num_block = (n * 2 - k) / DVEC_SIZE;
      int num_rem = (n * 2 - k) % DVEC_SIZE;
      double *p1 = &aug_data[i][k];
      const double *p2 = &aug_data[k][k];
      dvec tmp_vec = dvec_set1(tmp);
      for (int j = 0; j < num_block; ++j) {
        vec_store(p1, vec_sub(vec_load(p1), vec_mul(vec_load(p2), tmp_vec)));
        p1 += DVEC_SIZE;
        p2 += DVEC_SIZE;
      }
      for (int j = 0; j < num_rem; ++j) {
        p1[j] -= p2[j] * tmp;
      }
    }
  }
  Matrix res(n, n);
  auto &_res_data = res.getData();
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      _res_data[i][j] = aug_data[i][j + n];
    }
  }
  return res;
}
}
#endif
