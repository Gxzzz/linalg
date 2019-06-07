#include <bits/stdc++.h>
#include "../src/linalg.h"
using namespace std;
int main() {
  vector<vector<double>> a = {
    {0, 2, 0, 1},
    {1, 1, 0, 0},
    {3, 0, 0, 0}
  };
  vector<vector<double>> b = {
    {1, 1, 0, 0},
    {0, 1, 0, 0},
    {0, 0, 0, 2},
    {0, 0, 1, 0}
  };
  vector<vector<double>> c = {
    {1, 1, 0, 0},
    {0, 1, 0, 0},
    {0, 0, 0, 2}
  };
  auto cma = linalg::csr_matrix(a);
  auto cmb = linalg::csr_matrix(b);
  auto cmc = linalg::csr_matrix(c);

  cout << cma << endl;
  cout << cma.todense() << endl;

  /*
   * a + c =
   * | 1, 3, 0, 1 |
   * | 1, 2, 0, 0 |
   * | 3, 0, 0, 2 |
   */
  cout << cma + cmc << endl;

  /* a * b = 
   * | 0, 2, 1, 0 |
   * | 1, 2, 0, 0 |
   * | 3, 3, 0, 0 |
   */
  cout << cma.matmul(cmb) << endl;

  /*
   * a * d =
   * | 3 |
   * | 3 |
   * | 3 |
   */
  vector<double> d = {1, 2, 0, -1};
  auto ad = cma.dot(d);
  for (auto v : ad) {
    printf("%.0f ", v);
  }
  puts("\n");

  /*
   * sqrt(a) = 
   * | 0    , 1.414, 0    , 1.000 |
   * | 1.000, 1.000, 0    , 0     |
   * | 1.732, 0    , 0    , 0     |
   */
  cout << cma.map([](double a) { return sqrt(a); }) << endl;

  linalg::store_csr(cmc, "csr.in");
  auto cmd = linalg::load_csr("csr.in");
  cout << cmd << endl;
  return 0;
}
