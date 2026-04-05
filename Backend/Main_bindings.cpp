#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations 
int add_numbers(int a, int b);

PYBIND11_MODULE(cv_backend, m) {
    m.doc() = "Test backend module";

    // Bind the functions
    m.def("add_numbers", &add_numbers, "Adds two numbers from C++");
}