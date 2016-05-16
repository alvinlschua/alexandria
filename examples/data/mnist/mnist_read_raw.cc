#include <algorithm>
#include <fstream>
#include <iostream>

#include "tensor/tensor.h"
#include "tensor/tensor_dense.h"

namespace {
inline int flipBytes(int i) {
  std::reverse(reinterpret_cast<char*>(&i),
               reinterpret_cast<char*>(&i) + sizeof(i));
  return i;
}
}

using namespace std;
using namespace Alexandria;

int main() {
  /*
  ifstream fin("train-images-idx3-ubyte");
  ifstream fin2("train-labels-idx1-ubyte");
  */
  ifstream fin("t10k-images-idx3-ubyte");
  ifstream fin2("t10k-labels-idx1-ubyte");
  if (!fin.good()) {
    cerr << "Cannot open image file";
    return -1;
  }
  if (!fin2.good()) {
    cerr << "Cannot open label file";
    return -1;
  }

  std::vector<Tensor<uint8_t>> data;
  std::vector<uint8_t> labels;

  int magic_number = 0;
  int number_of_images = 0;
  int n_rows = 0;
  int n_cols = 0;

  fin.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
  magic_number = flipBytes(magic_number);
  fin.read(reinterpret_cast<char*>(&number_of_images),
           sizeof(number_of_images));
  number_of_images = flipBytes(number_of_images);
  fin.read(reinterpret_cast<char*>(&n_rows), sizeof(n_rows));
  n_rows = flipBytes(n_rows);
  fin.read(reinterpret_cast<char*>(&n_cols), sizeof(n_cols));
  n_cols = flipBytes(n_cols);

  int magic_number_label = 0;
  int number_of_items = 0;

  fin2.read(reinterpret_cast<char*>(&magic_number_label),
            sizeof(magic_number_label));
  magic_number_label = flipBytes(magic_number_label);
  fin2.read(reinterpret_cast<char*>(&number_of_items), sizeof(number_of_items));
  number_of_items = flipBytes(number_of_items);

  cout << "Magic Number: " << magic_number << "\n";
  cout << "Number of Images: " << number_of_images << "\n";
  cout << "Number of Rows: " << n_rows << "\n";
  cout << "Number of Cols: " << n_cols << "\n";
  cout << "------------------------------\n";
  cout << "Magic Number Label: " << magic_number_label << "\n";
  cout << "Number of Items: " << number_of_items << "\n";

  for (int i = 0; i < number_of_images; ++i) {
    std::vector<uint8_t> raw;
    for (int r = 0; r < n_rows; ++r) {
      for (int c = 0; c < n_cols; ++c) {
        unsigned char temp = 0;
        fin.read(reinterpret_cast<char*>(&temp), sizeof(temp));

        raw.emplace_back(temp);
      }
    }
    auto rows = static_cast<unsigned long>(n_rows);
    auto cols = static_cast<unsigned long>(n_cols);
    data.emplace_back(
        Tensor<uint8_t>(Tensor<uint8_t>::Dense(Shape({rows, cols}), raw)));

    // ATTACH ANSWER
    unsigned char temp;
    fin2.read(reinterpret_cast<char*>(&temp), sizeof(temp));
    labels.emplace_back(temp);
  }

  ofstream fout("mnist_testing.son");
  //ofstream fout("mnist_training.son");
  ArchiveOut ar(&fout);
  ar % labels % data;

  return 0;
}

