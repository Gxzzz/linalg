#include <bits/stdc++.h>
#include <sys/time.h>
#include "../linalg.h"
#include "../linalg_naive.h"
using namespace std;
const double eps = 1e-5;
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
  vector<vector<double>> a(n, vector<double>(n));
  vector<vector<double>> b(n, vector<double>(n));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      a[i][j] = 10.0 * rand() / RAND_MAX;
      b[i][j] = 10.0 * rand() / RAND_MAX;
    }
  }
  linalg_naive::Matrix n_ma(a);
  linalg_naive::Matrix n_mb(b);
  linalg::Matrix ma(a);
  linalg::Matrix mb(b);

  struct timeval start_time, end_time;
  double time;

  // test for naive implementation
  gettimeofday(&start_time, NULL);
  auto n_mc = linalg_naive::matmul(n_ma, n_mb);
  gettimeofday(&end_time, NULL);
  time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
    (end_time.tv_usec - start_time.tv_usec) / 1000.0;
  printf("time: %.3f ms\n", time);
  
  // test for linalg
  gettimeofday(&start_time, NULL);
  auto mc = linalg::matmul(ma, mb);
  gettimeofday(&end_time, NULL);
  time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
    (end_time.tv_usec - start_time.tv_usec) / 1000.0;
  printf("time: %.3f ms\n", time);

  printf("correctness: %d\n", test(n_mc.getData(), mc.getData(), n, n));
  return 0;
}
