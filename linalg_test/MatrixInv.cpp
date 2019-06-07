#include <bits/stdc++.h>
#include <sys/time.h>
#include "../src/linalg.h"
using namespace std;
const double eps = 1e-7;
int test(vector<vector<double>> &a, vector<vector<double>> &b, int n, int m) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      if (fabs(a[i][j] - b[i][j]) > eps)
        return 0;
    }
  }
  return 1;
}
int main() {
  srand(0x037f28);
  int n = 1000;
  vector<vector<double>> a(n, vector<double>(n, 0));
  for (int i = 0; i < n; ++i)
    a[i][i] = 1;
  for (int k = 0; k < 3000; ++k) {
    int op = rand() % 3;
    if (op == 0) {
      int i = rand() % n, j = rand() % n;
      a[i].swap(a[j]);
    } else if (op == 1) {
      int i = rand() % n;
      double scalar = 10.0 * rand() / RAND_MAX;
      for (int p = 0; p < n; ++p) {
        a[i][p] *= scalar;
      }
    } else {
      int i = rand() % n, j = rand() % n;
      double scalar = 10.0 * rand() / RAND_MAX;
      for (int p = 0; p < n; ++p) {
        a[i][p] += scalar * a[j][p];
      }
    }
  }
  linalg::matrix ma(a);

  struct timeval start_time, end_time;
  double time;
  
  // test for linalg
  gettimeofday(&start_time, NULL);
  auto res = linalg::inv(ma);
  gettimeofday(&end_time, NULL);
  time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
    (end_time.tv_usec - start_time.tv_usec) / 1000.0;
  printf("time: %.3f ms\n", time);

  auto I_matrix = linalg::matrix::eye(n);
  res = linalg::matmul(res, ma);
  printf("correctness: %d\n", test(I_matrix.getData(), res.getData(), n, n));
  return 0;
}
