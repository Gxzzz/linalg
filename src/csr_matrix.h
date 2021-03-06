#ifndef CSR_MATRIX_H
#define CSR_MATRIX_H
#include <vector>
#include <algorithm>
#include <cmath>
#include <utility>
#include <iostream>
#include "matrix.h"
namespace linalg {
class csr_matrix {
public:
  csr_matrix(int _n_rows, int _n_cols);
  csr_matrix(const std::vector<std::vector<double>> &mt);
  csr_matrix(const csr_matrix &other);
  csr_matrix(csr_matrix &&other);

  const int n_row() const;
  const int n_col() const;
  const std::vector<double> &getVal() const;
  std::vector<double> &getVal();
  const std::vector<int> &getRowPtr() const;
  std::vector<int> &getRowPtr();
  const std::vector<int> &getColInd() const;
  std::vector<int> &getColInd();
  csr_matrix &operator=(const csr_matrix &other);
  csr_matrix &operator=(csr_matrix &&other);

  static csr_matrix eye(int _size);
  csr_matrix transpose() const;
  csr_matrix map(std::function<double(double)>) const;

  std::vector<double> dot(const std::vector<double> &other);
  csr_matrix matmul(const csr_matrix &other);

  matrix todense();

  friend std::ostream &operator<<(std::ostream &output, const csr_matrix &mt);

private:
  int n_rows;
  int n_cols;
  std::vector<double> val;
  std::vector<int> row_ptr, col_ind;
};

csr_matrix::csr_matrix(int _n_rows, int _n_cols) {
  n_rows = _n_rows;
  n_cols = _n_cols;
  row_ptr.assign(_n_rows + 1, 0);
}

csr_matrix::csr_matrix(const std::vector<std::vector<double>> &mt) {
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

csr_matrix::csr_matrix(const csr_matrix &other) {
  n_rows = other.n_row();
  n_cols = other.n_col();
  val = other.getVal();
  row_ptr = other.getRowPtr();
  col_ind = other.getColInd();
}

csr_matrix::csr_matrix(csr_matrix &&other) {
  n_rows = other.n_row();
  n_cols = other.n_col();
  val.swap(other.getVal());
  row_ptr.swap(other.getRowPtr());
  col_ind.swap(other.getColInd());
}

const int csr_matrix::n_row() const {
  return n_rows;
}

const int csr_matrix::n_col() const {
  return n_cols;
}

const std::vector<double> &csr_matrix::getVal() const {
  return val;
}

std::vector<double> &csr_matrix::getVal() {
  return val;
}

const std::vector<int> &csr_matrix::getRowPtr() const {
  return row_ptr;
}

std::vector<int> &csr_matrix::getRowPtr() {
  return row_ptr;
} 

const std::vector<int> &csr_matrix::getColInd() const {
  return col_ind;
}

std::vector<int> &csr_matrix::getColInd() {
  return col_ind;
}

csr_matrix &csr_matrix::operator=(const csr_matrix &other) {
  n_rows = other.n_row();
  n_cols = other.n_col();
  val = other.getVal();
  row_ptr = other.getRowPtr();
  col_ind = other.getColInd();
  return *this;
}

csr_matrix &csr_matrix::operator=(csr_matrix &&other) {
  n_rows = other.n_row();
  n_cols = other.n_col();
  val.swap(other.getVal());
  row_ptr.swap(other.getRowPtr());
  col_ind.swap(other.getColInd());
  return *this;
}

csr_matrix csr_matrix::transpose() const {
  csr_matrix res(n_cols, n_rows);
  auto &res_val = res.getVal();
  auto &res_col_ind = res.getColInd();
  auto &res_row_ptr = res.getRowPtr();
  std::vector<std::pair<std::pair<int, int>, double>> tmp;
  for (int i = 0; i < n_rows; ++i) {
    int l = row_ptr[i], r = row_ptr[i + 1];
    for (int j = l; j < r; ++j) {
      tmp.push_back({{col_ind[j], i}, val[j]});
    }
  }
  sort(tmp.begin(), tmp.end());
  int row_number = -1;
  for (auto &x : tmp) {
    if (x.first.first > row_number) {
      for (int k = row_number + 1; k <= x.first.first; ++k)
        res_row_ptr[k] = res_val.size();
      row_number = x.first.first;
    }
    res_col_ind.push_back(x.first.second);
    res_val.push_back(x.second);
  }
  res_row_ptr[n_cols] = res_val.size();
  return res;
}

csr_matrix csr_matrix::map(std::function<double(double)> func) const {
  csr_matrix res(*this);
  auto &res_val = res.getVal();
  auto &res_col_ind = res.getColInd();
  auto &res_row_ptr = res.getRowPtr();
  for (int i = 0; i < n_rows; ++i) {
    int l = row_ptr[i], r = row_ptr[i + 1];
    for (int j = l; j < r; ++j) {
      res_val[j] = func(res_val[j]);
    }
  }
  return res;
}

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
    std::sort(tmp.begin(), tmp.end());
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

csr_matrix csr_matrix::eye(int _size) {
  csr_matrix res(_size, _size);
  auto &res_val = res.getVal();
  auto &res_col_ind = res.getColInd();
  auto &res_row_ptr = res.getRowPtr();
  res_val.assign(_size, 1);
  res_col_ind.resize(_size);
  for (int i = 0; i < _size; ++i)
    res_col_ind[i] = i;
  for (int i = 0; i <= _size; ++i)
    res_row_ptr[i] = i;
  return res;
}

matrix csr_matrix::todense() {
  matrix res(n_rows, n_cols);
  for (int r = 0; r < n_rows; ++r) {
    int left = row_ptr[r], right = row_ptr[r + 1];
    for (int i = left; i < right; ++i) {
      res[r][col_ind[i]] = val[i];
    }
  }
  return res;
}

std::ostream &operator<<(std::ostream &output, const csr_matrix &mt) {
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

csr_matrix operator-(const csr_matrix &a) {
  csr_matrix res(a);
  auto &res_val = res.getVal();
  for (auto &v : res_val) {
    v = -v;
  }
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

csr_matrix load_csr(const char *filename) {
  FILE *fp = fopen(filename, "r");
  int n_rows, n_cols;
  fscanf(fp, "%d%d", &n_rows, &n_cols);
  csr_matrix res(n_rows, n_cols);
  auto &res_val = res.getVal();
  auto &res_col_ind = res.getColInd();
  auto &res_row_ptr = res.getRowPtr();
  int r, c;
  double v;
  std::vector<std::pair<std::pair<int, int>, double>> triples;
  while (fscanf(fp, " (%d, %d): %lf", &r, &c, &v) == 3) {
    triples.push_back({{r, c}, v});
  }
  fclose(fp);
  sort(triples.begin(), triples.end());
  for (auto x : triples) {
    res_col_ind.push_back(x.first.second);
    res_val.push_back(x.second);
    ++res_row_ptr[x.first.first + 1];
  }
  for (int i = 1; i <= n_rows; ++i)
    res_row_ptr[i] += res_row_ptr[i - 1];
  return res;
}

void store_csr(const csr_matrix &mt, const char *filename) {
  FILE *fp = fopen(filename, "w");
  int n_rows = mt.n_row();
  int n_cols = mt.n_col();
  const auto &row_ptr = mt.getRowPtr();
  const auto &col_ind = mt.getColInd();
  const auto &val = mt.getVal();
  fprintf(fp, "%d %d\n", n_rows, n_cols);
  for (int i = 0; i < n_rows; ++i) {
    for (int p = row_ptr[i]; p < row_ptr[i + 1]; ++p) {
      fprintf(fp, "(%d, %d): %f\n", i, col_ind[p], val[p]);
    }
  }
  fclose(fp);
}

}
#endif
