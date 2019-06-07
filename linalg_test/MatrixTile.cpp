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
  int n = 1000;
  int r1 = 15, c1 = 15;
  vector<vector<double>> a(n, vector<double>(n));
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
  vector<vector<double>> c(n * r1, vector<double>(n * c1));
  for (int i = 0; i < n * r1; ++i) {
    for (int j = 0; j < n * c1; ++j) {
      int r2 = i % n;
      int c2 = j % n;
      c[i][j] = a[r2][c2];
    }
  }
  gettimeofday(&end_time, NULL);
  time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
    (end_time.tv_usec - start_time.tv_usec) / 1000.0;
  printf("time: %.3f ms\n", time);
  
  // test for linalg
  gettimeofday(&start_time, NULL);
  auto res = linalg::tile(ma, r1, c1);
  gettimeofday(&end_time, NULL);
  time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
    (end_time.tv_usec - start_time.tv_usec) / 1000.0;
  printf("time: %.3f ms\n", time);

  printf("%d\n", test(c, res.getData(), n * r1, n * c1));
  return 0;
}
