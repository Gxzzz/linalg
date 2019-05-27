#ifndef CSR_MATRIX_H
#define CSR_MATRIX_H
#include "matrix.h"
namespace linalg {
class csr_matrix {
public:
  csr_matrix(int _n_rows, int _n_cols) {
    n_rows = _n_rows;
    n_cols = _n_cols;
    row_ptr.assign(_n_rows + 1, 0);
  }

  csr_matrix(const std::vector<std::vector<double>> &mt) {
    n_rows = mt.size();
    n_cols = mt.size() > 0 ? mt[0].size() : 0;
    row_ptr.resize(n_rows + 1);
    for (int i = 0; i < n_rows; ++i) {
      row_ptr[i] = col_ind.size();
      for (int j = 0; j < n_cols; ++j) {
        if (fabs(mt[i][j]) > eps) {
          val.emplace_back(mt[i][j]);
          col_ind.emplace_back(j);
        }
      }
    }
    row_ptr[n_rows] = col_ind.size();
  }

  csr_matrix(const csr_matrix &other) {
    n_rows = other.n_row();
    n_cols = other.n_col();
    val = other.getVal();
    row_ptr = other.getRowPtr();
    col_ind = other.getColInd();
  }

  csr_matrix(csr_matrix &&other) {
    n_rows = other.n_row();
    n_cols = other.n_col();
    val.swap(other.getVal());
    row_ptr.swap(other.getRowPtr());
    col_ind.swap(other.getColInd());
  }

  const int n_row() const {
    return n_rows;
  }

  const int n_col() const {
    return n_cols;
  }

  const std::vector<double> &getVal() const {
    return val;
  }
  
  std::vector<double> &getVal() {
    return val;
  }

  const std::vector<int> &getRowPtr() const {
    return row_ptr;
  }
  
  std::vector<int> &getRowPtr() {
    return row_ptr;
  }

  const std::vector<int> &getColInd() const {
    return col_ind;
  }
  
  std::vector<int> &getColInd() {
    return col_ind;
  }

  csr_matrix &operator=(const csr_matrix &other) {
    n_rows = other.n_row();
    n_cols = other.n_col();
    val = other.getVal();
    row_ptr = other.getRowPtr();
    col_ind = other.getColInd();
    return *this;
  }

  csr_matrix &operator=(csr_matrix &&other) {
    n_rows = other.n_row();
    n_cols = other.n_col();
    val.swap(other.getVal());
    row_ptr.swap(other.getRowPtr());
    col_ind.swap(other.getColInd());
    return *this;
  }

  std::vector<double> dot(const std::vector<double> &other);
  csr_matrix matmul(const csr_matrix &other);

  friend std::ostream &operator<<(std::ostream &output, const csr_matrix &mt) {
    int n_rows = mt.n_row();
    int n_cols = mt.n_col();
    const auto &row_ptr = mt.getRowPtr();
    const auto &col_ind = mt.getColInd();
    const auto &val = mt.getVal();
    for (int i = 0; i < n_rows; ++i) {
      for (int p = row_ptr[i]; p < row_ptr[i + 1]; ++p) {
        output << "(" << i << ", " << col_ind[p] << ")" << ": " << val[p] << "\n";
      }
    }
    return output;
  }

private:
  int n_rows;
  int n_cols;
  std::vector<double> val;
  std::vector<int> row_ptr, col_ind;
};

std::vector<double> csr_matrix::dot(const std::vector<double> &other) {
  std::vector<double> res(n_rows);
  for (int i = 0; i < n_rows; ++i) {
    double v = 0;
    int l = row_ptr[i], r = row_ptr[i + 1];
    for (int j = l; j < r; ++j) {
      v += val[j] * other[col_ind[j]];
    }
    res[i] = v;
  }
  return res;
}

csr_matrix csr_matrix::matmul(const csr_matrix &other) {
  /*Exception*/
  csr_matrix res(n_rows, other.n_col());
  const auto &a_val = val;
  const auto &a_col_ind = col_ind;
  const auto &a_row_ptr = row_ptr;
  const auto &b_val = other.getVal();
  const auto &b_col_ind = other.getColInd();
  const auto &b_row_ptr = other.getRowPtr();
  auto &res_val = res.getVal();
  auto &res_col_ind = res.getColInd();
  auto &res_row_ptr = res.getRowPtr();
  for (int i = 0; i < n_rows; ++i) {
    res_row_ptr[i] = res_val.size();
    int l_a = a_row_ptr[i], r_a = a_row_ptr[i + 1];
    std::vector<std::pair<int, double>> tmp;
    for (int p = l_a; p < r_a; ++p) {
      double v = a_val[p];
      int l_b = b_row_ptr[a_col_ind[p]], r_b = b_row_ptr[a_col_ind[p] + 1];
      for (int j = l_b; j < r_b; ++j) {
        tmp.push_back({b_col_ind[j], v * b_val[j]});
      }
    }
    sort(tmp.begin(), tmp.end());
    for (int j = 0, pre = 0; j <= tmp.size(); ++j) {
      if (j == tmp.size() || tmp[j].first != tmp[pre].first) {
        double val = 0;
        for (int k = pre; k < j; ++k)
          val += tmp[k].second;
        if (fabs(val) > eps) {
          res_val.emplace_back(val);
          res_col_ind.emplace_back(tmp[pre].first);
        }
        pre = j;
      }
    }
  }
  res_row_ptr[n_rows] = res_val.size();
  return res;
}

csr_matrix operator+(const csr_matrix &a, const csr_matrix &b) {
  int n_rows = a.n_row();
  int n_cols = a.n_col();
  /*Exception*/
  csr_matrix res(n_rows, n_cols);
  const auto &a_val = a.getVal();
  const auto &a_col_ind = a.getColInd();
  const auto &a_row_ptr = a.getRowPtr();
  const auto &b_val = b.getVal();
  const auto &b_col_ind = b.getColInd();
  const auto &b_row_ptr = b.getRowPtr();
  auto &res_val = res.getVal();
  auto &res_col_ind = res.getColInd();
  auto &res_row_ptr = res.getRowPtr();
  for (int i = 0; i < n_rows; ++i) {
    res_row_ptr[i] = res_col_ind.size();
    int p1 = a_row_ptr[i], r1 = a_row_ptr[i + 1];
    int p2 = b_row_ptr[i], r2 = b_row_ptr[i + 1];
    while (p1 < r1 && p2 < r2) {
      if (a_col_ind[p1] < b_col_ind[p2]) {
        res_val.emplace_back(a_val[p1]);
        res_col_ind.emplace_back(a_col_ind[p1]);
        ++p1;
      } else if (a_col_ind[p1] > b_col_ind[p2]) {
        res_val.emplace_back(b_val[p2]);
        res_col_ind.emplace_back(b_col_ind[p2]);
        ++p2;
      } else {
        if (fabs(a_val[p1] + b_val[p2]) > eps) {
          res_val.emplace_back(a_val[p1] + b_val[p2]);
          res_col_ind.emplace_back(a_col_ind[p1]);
        }
        ++p1, ++p2;
      }
    }
    for (; p1 < r1; ++p1) {
      res_val.emplace_back(a_val[p1]);
      res_col_ind.emplace_back(a_col_ind[p1]);
    }
    for (; p2 < r2; ++p2) {
      res_val.emplace_back(b_val[p2]);
      res_col_ind.emplace_back(b_col_ind[p2]);
    }
  }
  res_row_ptr[n_rows] = res_col_ind.size();
  return res;
}

csr_matrix operator-(const csr_matrix &a, const csr_matrix &b) {
  int n_rows = a.n_row();
  int n_cols = a.n_col();
  /*Exception*/
  csr_matrix res(n_rows, n_cols);
  const auto &a_val = a.getVal();
  const auto &a_col_ind = a.getColInd();
  const auto &a_row_ptr = a.getRowPtr();
  const auto &b_val = b.getVal();
  const auto &b_col_ind = b.getColInd();
  const auto &b_row_ptr = b.getRowPtr();
  auto &res_val = res.getVal();
  auto &res_col_ind = res.getColInd();
  auto &res_row_ptr = res.getRowPtr();
  for (int i = 0; i < n_rows; ++i) {
    res_row_ptr[i] = res_col_ind.size();
    int p1 = a_row_ptr[i], r1 = a_row_ptr[i + 1];
    int p2 = b_row_ptr[i], r2 = b_row_ptr[i + 1];
    while (p1 < r1 && p2 < r2) {
      if (a_col_ind[p1] < b_col_ind[p2]) {
        res_val.emplace_back(a_val[p1]);
        res_col_ind.emplace_back(a_col_ind[p1]);
        ++p1;
      } else if (a_col_ind[p1] > b_col_ind[p2]) {
        res_val.emplace_back(-b_val[p2]);
        res_col_ind.emplace_back(b_col_ind[p2]);
        ++p2;
      } else {
        if (fabs(a_val[p1] - b_val[p2]) > eps) {
          res_val.emplace_back(a_val[p1] - b_val[p2]);
          res_col_ind.emplace_back(a_col_ind[p1]);
        }
        ++p1, ++p2;
      }
    }
    for (; p1 < r1; ++p1) {
      res_val.emplace_back(a_val[p1]);
      res_col_ind.emplace_back(a_col_ind[p1]);
    }
    for (; p2 < r2; ++p2) {
      res_val.emplace_back(-b_val[p2]);
      res_col_ind.emplace_back(b_col_ind[p2]);
    }
  }
  res_row_ptr[n_rows] = res_col_ind.size();
  return res;
}

csr_matrix operator*(const csr_matrix &a, const csr_matrix &b) {
  int n_rows = a.n_row();
  int n_cols = a.n_col();
  /*Exception*/
  csr_matrix res(n_rows, n_cols);
  const auto &a_val = a.getVal();
  const auto &a_col_ind = a.getColInd();
  const auto &a_row_ptr = a.getRowPtr();
  const auto &b_val = b.getVal();
  const auto &b_col_ind = b.getColInd();
  const auto &b_row_ptr = b.getRowPtr();
  auto &res_val = res.getVal();
  auto &res_col_ind = res.getColInd();
  auto &res_row_ptr = res.getRowPtr();
  for (int i = 0; i < n_rows; ++i) {
    res_row_ptr[i] = res_col_ind.size();
    int p1 = a_row_ptr[i], r1 = a_row_ptr[i + 1];
    int p2 = b_row_ptr[i], r2 = b_row_ptr[i + 1];
    while (p1 < r1 && p2 < r2) {
      if (a_col_ind[p1] < b_col_ind[p2]) {
        ++p1;
      } else if (a_col_ind[p1] > b_col_ind[p2]) {
        ++p2;
      } else {
        res_val.emplace_back(a_val[p1] * b_val[p2]);
        res_col_ind.emplace_back(a_col_ind[p1]);
        ++p1, ++p2;
      }
    }
  }
  res_row_ptr[n_rows] = res_col_ind.size();
  return res;
}

csr_matrix operator*(const double &val, const csr_matrix &a) {
  int n_rows = a.n_row();
  int n_cols = a.n_col();
  if (fabs(val) < eps)
    return csr_matrix(n_rows, n_cols);
  csr_matrix res(a);
  auto &res_val = res.getVal();
  for (double &v : res_val) {
    v *= val;
  }
  return res;
}

csr_matrix operator*(const csr_matrix &a, const double &val) {
  return val * a;
}

}
#endif
