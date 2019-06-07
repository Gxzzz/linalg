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
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      a[i][j] = 10.0 * rand() / RAND_MAX;
    }
  }
  double val = 10.0 * rand() / RAND_MAX;
  linalg::matrix ma(a);

  struct timeval start_time, end_time;
  double time;
  
  // test for linalg
  gettimeofday(&start_time, NULL);
  vector<vector<double>> c1(n, vector<double>(n));
  vector<vector<double>> c2(n, vector<double>(n));
  vector<vector<double>> c3(n, vector<double>(n));
  vector<vector<double>> c4(n, vector<double>(n));
  vector<vector<double>> c5(n, vector<double>(n));
  vector<vector<double>> c6(n, vector<double>(n));
  vector<vector<double>> c7(n, vector<double>(n));
  vector<vector<double>> c8(n, vector<double>(n));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      c1[i][j] = val + a[i][j];
    }
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      c2[i][j] = a[i][j] + val;
    }
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      c3[i][j] = val - a[i][j];
    }
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      c4[i][j] = a[i][j] - val;
    }
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      c5[i][j] = val * a[i][j];
    }
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      c6[i][j] = a[i][j] * val;
    }
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      c7[i][j] = val / a[i][j];
    }
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      c8[i][j] = a[i][j] / val;
    }
  }
  gettimeofday(&end_time, NULL);
  time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
    (end_time.tv_usec - start_time.tv_usec) / 1000.0;
  printf("time: %.3f ms\n", time);

  // test for linalg
  gettimeofday(&start_time, NULL);
  auto mc1 = val + ma;
  auto mc2 = ma + val;
  auto mc3 = val - ma;
  auto mc4 = ma - val;
  auto mc5 = val * ma;
  auto mc6 = ma * val;
  auto mc7 = val / ma;
  auto mc8 = ma / val;
  gettimeofday(&end_time, NULL);
  time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
    (end_time.tv_usec - start_time.tv_usec) / 1000.0;
  printf("time: %.3f ms\n", time);

  printf("%d\n", test(c1, mc1.getData(), n, n));
  printf("%d\n", test(c2, mc2.getData(), n, n));
  printf("%d\n", test(c3, mc3.getData(), n, n));
  printf("%d\n", test(c4, mc4.getData(), n, n));
  printf("%d\n", test(c5, mc5.getData(), n, n));
  printf("%d\n", test(c6, mc6.getData(), n, n));
  printf("%d\n", test(c7, mc7.getData(), n, n));
  printf("%d\n", test(c8, mc8.getData(), n, n));
  return 0;
}
