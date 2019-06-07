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
      b[i][j] = 10.0 * rand() / RAND_MAX;
    }
  }
  linalg::matrix ma(a);
  linalg::matrix mb(b);

  struct timeval start_time, end_time;
  double time;

  // test for simple implementation
  gettimeofday(&start_time, NULL);
  vector<vector<double>> c1(n, vector<double>(n));
  vector<vector<double>> c2(n, vector<double>(n));
  vector<vector<double>> c3(n, vector<double>(n));
  vector<vector<double>> c4(n, vector<double>(n));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      c1[i][j] = a[i][j] + b[i][j];
      c2[i][j] = a[i][j] - b[i][j];
      c3[i][j] = a[i][j] * b[i][j];
      c4[i][j] = a[i][j] / b[i][j];
    }
  }
  gettimeofday(&end_time, NULL);
  time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
    (end_time.tv_usec - start_time.tv_usec) / 1000.0;
  printf("time: %.3f ms\n", time); 

  // test for linalg
  gettimeofday(&start_time, NULL);
  auto mc1 = ma + mb;
  auto mc2 = ma - mb;
  auto mc3 = ma * mb;
  auto mc4 = ma / mb;
  gettimeofday(&end_time, NULL);
  time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
    (end_time.tv_usec - start_time.tv_usec) / 1000.0;
  printf("time: %.3f ms\n", time);

  printf("%d\n", test(c1, mc1.getData(), n, n));
  printf("%d\n", test(c2, mc2.getData(), n, n));
  printf("%d\n", test(c3, mc3.getData(), n, n));
  printf("%d\n", test(c4, mc4.getData(), n, n));
  return 0;
}
