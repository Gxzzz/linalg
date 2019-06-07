#include <bits/stdc++.h>
#include <sys/time.h>
#include "../src/linalg.h"
using namespace std;
const double eps = 1e-8;
int test(const vector<vector<double>> &a, const vector<vector<double>> b, int n, int m) {
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
  int n = 5000;
  vector<vector<double>> a(n, vector<double>(n));
  vector<vector<double>> b(n, vector<double>(n));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      a[i][j] = 10.0 * rand() / RAND_MAX;
    }
  }
  linalg::matrix ma(a);

  struct timeval start_time, end_time;
  double time;

  // test for simple implementation
  gettimeofday(&start_time, NULL);
  vector<vector<double>> c1(1, vector<double>(n));
  vector<vector<double>> c2(n, vector<double>(1));
  double c3 = 0;
  for (int i = 0; i < n; ++i) {
    double val = 0;
    for (int j = 0; j < n; ++j) {
      val += a[j][i];
    }
    c1[0][i] = val / n;
  }
  for (int i = 0; i < n; ++i) {
    double val = 0;
    for (int j = 0; j < n; ++j) {
      val += a[i][j];
    }
    c2[i][0] = val / n;
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      c3 += a[i][j];
    }
  }
  c3 /= (n * n);
  gettimeofday(&end_time, NULL);
  time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
    (end_time.tv_usec - start_time.tv_usec) / 1000.0;
  printf("time: %.3f ms\n", time);

  // test for linalg
  gettimeofday(&start_time, NULL);
  auto res1 = linalg::mean(ma, 0);
  auto res2 = linalg::mean(ma, 1);
  auto res = linalg::mean(ma);
  gettimeofday(&end_time, NULL);
  time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
    (end_time.tv_usec - start_time.tv_usec) / 1000.0;
  printf("time: %.3f ms\n", time);

  printf("%d\n", test(c1, res1.getData(), 1, n));
  printf("%d\n", test(c2, res2.getData(), n, 1));
  printf("%d\n", fabs(c3 - res) < eps);
  return 0;
}
