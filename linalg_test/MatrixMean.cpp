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
  int n = 5000;
  vector<vector<double>> a(n, vector<double>(n));
  vector<vector<double>> b(n, vector<double>(n));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      a[i][j] = 10.0 * rand() / RAND_MAX;
    }
  }
  linalg::Matrix ma(a);
  linalg_naive::Matrix n_ma(a);

  struct timeval start_time, end_time;
  double time;

  // test for naive implementation
  gettimeofday(&start_time, NULL);
  auto n_res1 = linalg_naive::mean(n_ma, 0);
  auto n_res2 = linalg_naive::mean(n_ma, 1);
  auto n_res = linalg_naive::mean(n_ma);
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

  printf("correctness: %d\n", test(n_res1.getData(), res1.getData(), 1, n));
  printf("correctness: %d\n", test(n_res2.getData(), res2.getData(), n, 1));
  printf("correctness: %d\n", fabs(n_res - res) < eps);
  return 0;
}
