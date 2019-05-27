#include <bits/stdc++.h>
#include <sys/time.h>
#include "../src/linalg.h"
using namespace std;
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

  struct timeval start_time, end_time;
  double time;

  // test for linalg
  gettimeofday(&start_time, NULL);
  auto res1 = linalg::mean(ma, 0);
  auto res2 = linalg::mean(ma, 1);
  auto res = linalg::mean(ma);
  gettimeofday(&end_time, NULL);
  time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
    (end_time.tv_usec - start_time.tv_usec) / 1000.0;
  printf("time: %.3f ms\n", time);

  return 0;
}
