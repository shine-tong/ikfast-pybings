// Python bindings for IKFast solver using pybind11
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// Include the IKFast header and solver implementation
#define IKFAST_HAS_LIBRARY
#define IKFAST_NO_MAIN
#include "../include/ikfast.h"

// IKFast functions are already declared in ikfast.h
// We just need to use them directly
using namespace ikfast;

namespace py = pybind11;

// ============================================================================
// Exception Translation Layer
// ============================================================================

/**
 * Exception translation wrapper template.
 * 
 * Wraps a callable and translates C++ exceptions to appropriate Python exceptions.
 * This ensures that errors from the C++ solver are properly propagated to Python
 * with descriptive error messages.
 * 
 * Exception mapping:
 * - std::invalid_argument → py::value_error
 * - std::out_of_range → py::index_error
 * - std::runtime_error → RuntimeError (via py::error_already_set)
 * - std::exception → RuntimeError (with context)
 * 
 * @tparam Func Callable type (lambda, function pointer, etc.)
 * @param func The function to wrap
 * @param context_name Name of the function for error context
 * @return Result of the wrapped function
 * @throws py::error_already_set with appropriate Python exception type
 */
template<typename Func>
decltype(auto) translate_exceptions(Func&& func, const std::string& context_name) {
    try {
        return func();
    } catch (const std::invalid_argument& e) {
        throw py::value_error(context_name + ": " + e.what());
    } catch (const std::out_of_range& e) {
        throw py::index_error(context_name + ": " + e.what());
    } catch (const std::runtime_error& e) {
        // For runtime_error, we need to set a Python RuntimeError
        PyErr_SetString(PyExc_RuntimeError, (context_name + ": " + e.what()).c_str());
        throw py::error_already_set();
    } catch (const std::exception& e) {
        // For generic exceptions, also set RuntimeError with context
        PyErr_SetString(PyExc_RuntimeError, (context_name + ": Unexpected error: " + e.what()).c_str());
        throw py::error_already_set();
    }
}

// ============================================================================
// Array Type Conversion Layer - Helper Functions
// ============================================================================

/**
 * Validate numpy array shape against expected dimensions.
 * 
 * @param arr The numpy array to validate
 * @param expected_shape Vector of expected dimensions (-1 means any size)
 * @param name Name of the array for error messages
 * @throws std::invalid_argument if shape doesn't match
 */
void validate_array_shape(
    py::array_t<double>& arr,
    const std::vector<ssize_t>& expected_shape,
    const std::string& name
) {
    auto buf = arr.request();
    
    // Check number of dimensions
    if (buf.ndim != static_cast<ssize_t>(expected_shape.size())) {
        throw std::invalid_argument(
            name + ": Invalid number of dimensions. Expected: " + 
            std::to_string(expected_shape.size()) + ", Got: " + 
            std::to_string(buf.ndim)
        );
    }
    
    // Check each dimension size
    for (size_t i = 0; i < expected_shape.size(); ++i) {
        if (expected_shape[i] != -1 && buf.shape[i] != expected_shape[i]) {
            std::string expected_str = "(";
            std::string actual_str = "(";
            for (size_t j = 0; j < expected_shape.size(); ++j) {
                if (j > 0) {
                    expected_str += ", ";
                    actual_str += ", ";
                }
                if (expected_shape[j] == -1) {
                    expected_str += "*";
                } else {
                    expected_str += std::to_string(expected_shape[j]);
                }
                actual_str += std::to_string(buf.shape[j]);
            }
            expected_str += ")";
            actual_str += ")";
            
            throw std::invalid_argument(
                name + ": Invalid array shape. Expected: " + 
                expected_str + ", Got: " + actual_str
            );
        }
    }
}

/**
 * Get C++ pointer from numpy array with validation.
 * 
 * @tparam T The data type (typically double)
 * @param arr The numpy array
 * @return Pointer to the array data
 */
template<typename T>
const T* get_array_ptr(py::array_t<T>& arr) {
    py::buffer_info buf = arr.request();
    return static_cast<const T*>(buf.ptr);
}

/**
 * Create numpy array from C++ data.
 * 
 * @param data Vector of data to copy into the array
 * @param shape Shape of the output array
 * @return New numpy array with float64 dtype
 */
py::array_t<double> create_numpy_array(
    const std::vector<double>& data,
    const std::vector<ssize_t>& shape
) {
    // Calculate total size
    ssize_t total_size = 1;
    for (auto dim : shape) {
        total_size *= dim;
    }
    
    // Validate data size matches shape
    if (static_cast<ssize_t>(data.size()) != total_size) {
        throw std::runtime_error(
            "create_numpy_array: Data size (" + std::to_string(data.size()) + 
            ") does not match shape size (" + std::to_string(total_size) + ")"
        );
    }
    
    // Create array with proper shape
    auto result = py::array_t<double>(shape);
    auto buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);
    
    // Copy data
    std::copy(data.begin(), data.end(), ptr);
    
    return result;
}

// Module initialization
PYBIND11_MODULE(_ikfast_pybind, m) {
    m.doc() = R"pbdoc(
        IKFast Python Bindings
        ----------------------
        
        Python bindings for the IKFast inverse kinematics solver.
        
        This module provides low-level access to the IKFast C++ solver,
        with functions for computing inverse and forward kinematics,
        as well as querying solver properties.
    )pbdoc";
    
    // Module version
    m.attr("__version__") = "0.1.0";
    
    // ========================================================================
    // IkSolutionList class binding
    // ========================================================================
    
    py::class_<IkSolutionList<IkReal>, std::shared_ptr<IkSolutionList<IkReal>>>(m, "IkSolutionList",
        R"pbdoc(
            Wrapper for a list of IK solutions.
            
            Provides a container for multiple inverse kinematics solutions.
            Supports Python iteration protocols (len, indexing, iteration).
        )pbdoc")
        .def("__len__", [](const IkSolutionList<IkReal>& self) -> size_t {
            return self.GetNumSolutions();
        },
        R"pbdoc(
            Get the number of solutions in the list.
            
            Returns:
                int: Number of solutions
        )pbdoc")
        .def("__getitem__", [](const IkSolutionList<IkReal>& self, ssize_t index) -> std::shared_ptr<const IkSolutionBase<IkReal>> {
            size_t num_solutions = self.GetNumSolutions();
            
            // Handle negative indices
            if (index < 0) {
                index += num_solutions;
            }
            
            // Check bounds
            if (index < 0 || index >= static_cast<ssize_t>(num_solutions)) {
                throw py::index_error(
                    "IkSolutionList.__getitem__: Index out of range. Valid range: [0, " +
                    std::to_string(num_solutions) + "), Got: " + std::to_string(index)
                );
            }
            
            // Return solution with no-op deleter (solution is owned by list)
            const IkSolutionBase<IkReal>& sol = self.GetSolution(index);
            return std::shared_ptr<const IkSolutionBase<IkReal>>(&sol, [](const IkSolutionBase<IkReal>*){});
        },
        R"pbdoc(
            Get solution by index.
            
            Supports negative indexing (e.g., -1 for last solution).
            
            Args:
                index: Solution index (0-based)
            
            Returns:
                IkSolution: The solution at the specified index
            
            Raises:
                IndexError: If index is out of bounds
        )pbdoc")
        .def("__iter__", [](IkSolutionList<IkReal>& self) {
            // Return a Python list of solutions for iteration
            // This is the simplest and most reliable approach
            py::list solutions;
            for (size_t i = 0; i < self.GetNumSolutions(); ++i) {
                const IkSolutionBase<IkReal>& sol = self.GetSolution(i);
                solutions.append(std::shared_ptr<const IkSolutionBase<IkReal>>(&sol, [](const IkSolutionBase<IkReal>*){})            );
            }
            return py::iter(solutions);
        },
        R"pbdoc(
            Iterate through all solutions.
            
            Returns:
                iterator: Iterator over IkSolution objects
        )pbdoc")
        .def("get_num_solutions", &IkSolutionList<IkReal>::GetNumSolutions,
        R"pbdoc(
            Get the number of solutions in the list.
            
            Returns:
                int: Number of solutions
        )pbdoc")
        .def("clear", &IkSolutionList<IkReal>::Clear,
        R"pbdoc(
            Clear all solutions from the list.
            
            Note: Any references to solutions obtained from GetSolution
            will be invalidated after calling this method.
        )pbdoc");
    
    // ========================================================================
    // IkSolution class binding
    // ========================================================================
    
    py::class_<IkSolutionBase<IkReal>, std::shared_ptr<IkSolutionBase<IkReal>>>(m, "IkSolution",
        R"pbdoc(
            Wrapper for a single IK solution.
            
            Represents one inverse kinematics solution, which may contain
            free parameters that need to be specified to get a concrete
            joint configuration.
        )pbdoc")
        .def("get_solution", [](const IkSolutionBase<IkReal>& self, py::object free_values) -> py::array_t<double> {
            return translate_exceptions([&]() {
                int dof = self.GetDOF();
                const std::vector<int>& free_indices = self.GetFree();
                
                // Prepare free values pointer
                const IkReal* pfree = nullptr;
                std::vector<IkReal> free_vec;
                
                if (!free_values.is_none()) {
                    // Convert Python object to numpy array
                    py::array_t<double> free_arr = py::cast<py::array_t<double>>(free_values);
                    auto buf = free_arr.request();
                    
                    // Validate free values size
                    if (buf.size != static_cast<ssize_t>(free_indices.size())) {
                        throw std::invalid_argument(
                            "get_solution: Invalid free_values size. Expected: " +
                            std::to_string(free_indices.size()) + ", Got: " +
                            std::to_string(buf.size)
                        );
                    }
                    
                    // Copy to vector
                    const double* ptr = static_cast<const double*>(buf.ptr);
                    free_vec.assign(ptr, ptr + buf.size);
                    pfree = free_vec.data();
                }
                
                // Get solution
                std::vector<IkReal> solution(dof);
                self.GetSolution(solution.data(), pfree);
                
                // Convert to numpy array
                return create_numpy_array(solution, {static_cast<ssize_t>(dof)});
            }, "IkSolution.get_solution");
        },
        py::arg("free_values") = py::none(),
        R"pbdoc(
            Get concrete joint angles for this solution.
            
            Args:
                free_values: Optional numpy array of free parameter values.
                            Required if the solution has free parameters.
            
            Returns:
                numpy.ndarray: Array of joint angles with shape (num_joints,) and dtype float64
            
            Raises:
                ValueError: If free_values size doesn't match number of free parameters
        )pbdoc")
        .def("get_free_indices", [](const IkSolutionBase<IkReal>& self) -> py::list {
            const std::vector<int>& free_indices = self.GetFree();
            py::list result;
            for (int idx : free_indices) {
                result.append(idx);
            }
            return result;
        },
        R"pbdoc(
            Get indices of free parameters.
            
            Returns:
                list: List of integer indices indicating which joints are free parameters
        )pbdoc")
        .def("get_dof", &IkSolutionBase<IkReal>::GetDOF,
        R"pbdoc(
            Get degrees of freedom (number of joints).
            
            Returns:
                int: Number of joints in the solution
        )pbdoc");
    
    // ========================================================================
    // ComputeIk binding (minimal for testing IkSolution)
    // ========================================================================
    
    m.def("compute_ik_raw", [](py::array_t<double> eetrans, py::array_t<double> eerot, py::object pfree) -> std::shared_ptr<IkSolutionList<IkReal>> {
        return translate_exceptions([&]() -> std::shared_ptr<IkSolutionList<IkReal>> {
            // Validate inputs
            validate_array_shape(eetrans, {3}, "eetrans");
            
            // Handle both flat [9] and matrix [3,3] rotation
            auto rot_buf = eerot.request();
            if (rot_buf.ndim == 1 && rot_buf.shape[0] == 9) {
                // Flat array is OK
            } else if (rot_buf.ndim == 2 && rot_buf.shape[0] == 3 && rot_buf.shape[1] == 3) {
                // Matrix is OK
            } else {
                throw std::invalid_argument(
                    "compute_ik_raw: Invalid eerot shape. Expected: (9,) or (3, 3), Got: (" +
                    std::to_string(rot_buf.shape[0]) + 
                    (rot_buf.ndim > 1 ? ", " + std::to_string(rot_buf.shape[1]) : "") + ")"
                );
            }
            
            // Get pointers
            const IkReal* trans_ptr = get_array_ptr(eetrans);
            const IkReal* rot_ptr = get_array_ptr(eerot);
            const IkReal* free_ptr = nullptr;
            
            // Handle optional free parameters
            std::vector<IkReal> free_vec;
            if (!pfree.is_none()) {
                py::array_t<double> free_arr = py::cast<py::array_t<double>>(pfree);
                auto free_buf = free_arr.request();
                const double* ptr = static_cast<const double*>(free_buf.ptr);
                free_vec.assign(ptr, ptr + free_buf.size);
                free_ptr = free_vec.data();
            }
            
            // Create solution list (must be kept alive)
            auto solutions = std::make_shared<IkSolutionList<IkReal>>();
            
            // Call C++ ComputeIk
            bool success = ComputeIk(trans_ptr, rot_ptr, free_ptr, *solutions);
            
            // Return the solution list (even if empty)
            return solutions;
        }, "compute_ik_raw");
    },
    py::arg("eetrans"),
    py::arg("eerot"),
    py::arg("pfree") = py::none(),
    R"pbdoc(
        Compute inverse kinematics solutions (raw interface).
        
        Args:
            eetrans: End effector translation [x, y, z] as numpy array
            eerot: End effector rotation matrix (3x3) or flattened (9,) as numpy array
            pfree: Optional free parameter values as numpy array
        
        Returns:
            IkSolutionList: List of IkSolution objects
        
        Raises:
            ValueError: If input arrays have incorrect shape
    )pbdoc");
    
    // ========================================================================
    // ComputeFk binding (minimal for testing)
    // ========================================================================
    
    m.def("compute_fk_raw", [](py::array_t<double> joints) -> py::tuple {
        return translate_exceptions([&]() -> py::tuple {
            // Validate input
            int num_joints = GetNumJoints();
            validate_array_shape(joints, {static_cast<ssize_t>(num_joints)}, "joints");
            
            // Get pointer
            const IkReal* joints_ptr = get_array_ptr(joints);
            
            // Allocate output arrays
            std::vector<IkReal> eetrans(3);
            std::vector<IkReal> eerot(9);
            
            // Call C++ ComputeFk
            ComputeFk(joints_ptr, eetrans.data(), eerot.data());
            
            // Convert to numpy arrays
            py::array_t<double> trans_arr = create_numpy_array(eetrans, {3});
            py::array_t<double> rot_arr = create_numpy_array(eerot, {9});
            
            return py::make_tuple(trans_arr, rot_arr);
        }, "compute_fk_raw");
    },
    py::arg("joints"),
    R"pbdoc(
        Compute forward kinematics (raw interface).
        
        Args:
            joints: Array of joint angles with shape (num_joints,)
        
        Returns:
            tuple: (translation, rotation) where:
                - translation: numpy array with shape (3,) and dtype float64
                - rotation: numpy array with shape (9,) and dtype float64 (flattened 3x3 matrix)
        
        Raises:
            ValueError: If joints array has incorrect shape
    )pbdoc");
    
    // Bind information query functions
    m.def("get_num_joints", &GetNumJoints,
          R"pbdoc(
              Get the number of joints in the robot.
              
              Returns:
                  int: Number of joints (typically 6 for a 6-DOF manipulator)
          )pbdoc");
    
    m.def("get_num_free_parameters", &GetNumFreeParameters,
          R"pbdoc(
              Get the number of free parameters in the IK solution.
              
              Free parameters are joint values that can be set arbitrarily
              when the robot has redundant degrees of freedom.
              
              Returns:
                  int: Number of free parameters (0 if no redundancy)
          )pbdoc");
    
    m.def("get_free_parameters", []() -> py::list {
        int* params = GetFreeParameters();
        int num_free = GetNumFreeParameters();
        py::list result;
        if (params != nullptr && num_free > 0) {
            for (int i = 0; i < num_free; ++i) {
                result.append(params[i]);
            }
        }
        return result;
    },
    R"pbdoc(
        Get the indices of free parameters.
        
        Returns:
            list: List of integer indices indicating which joints are free parameters
    )pbdoc");
    
    m.def("get_ik_type", &GetIkType,
          R"pbdoc(
              Get the IK solver type identifier.
              
              This is a constant that identifies the type of IK solver
              (e.g., Transform6D, Translation3D, etc.).
              
              Returns:
                  int: IK type constant
          )pbdoc");
    
    m.def("get_kinematics_hash", &GetKinematicsHash,
          R"pbdoc(
              Get the kinematics hash string.
              
              This hash uniquely identifies the robot kinematics configuration
              used to generate this solver.
              
              Returns:
                  str: Kinematics hash string
          )pbdoc");
    
    m.def("get_ikfast_version", &GetIkFastVersion,
          R"pbdoc(
              Get the IKFast version used to generate this solver.
              
              Returns:
                  str: IKFast version string
          )pbdoc");
    
    // ========================================================================
    // Test helper functions for array type conversion
    // ========================================================================
    
    m.def("_test_array_echo", [](py::array_t<double> arr) -> py::array_t<double> {
        // Simply return the array to test type conversion
        return arr;
    },
    R"pbdoc(
        Test helper: Echo back the input array.
        Used for testing array type conversion.
    )pbdoc");
    
    m.def("_test_array_sum", [](py::array_t<double> arr) -> double {
        // Sum all elements to test array access
        auto buf = arr.request();
        const double* ptr = static_cast<const double*>(buf.ptr);
        double sum = 0.0;
        for (ssize_t i = 0; i < buf.size; ++i) {
            sum += ptr[i];
        }
        return sum;
    },
    R"pbdoc(
        Test helper: Sum all elements in the array.
        Used for testing array data access.
    )pbdoc");
    
    m.def("_test_validate_shape", [](py::array_t<double> arr, py::list expected_shape, std::string name) {
        return translate_exceptions([&]() {
            // Test the validate_array_shape function
            std::vector<ssize_t> shape_vec;
            for (auto item : expected_shape) {
                shape_vec.push_back(py::cast<ssize_t>(item));
            }
            validate_array_shape(arr, shape_vec, name);
        }, "_test_validate_shape");
    },
    R"pbdoc(
        Test helper: Validate array shape.
        Used for testing shape validation.
    )pbdoc");
    
    m.def("_test_create_array", [](py::list data, py::list shape) -> py::array_t<double> {
        return translate_exceptions([&]() {
            // Test the create_numpy_array function
            std::vector<double> data_vec;
            for (auto item : data) {
                data_vec.push_back(py::cast<double>(item));
            }
            std::vector<ssize_t> shape_vec;
            for (auto item : shape) {
                shape_vec.push_back(py::cast<ssize_t>(item));
            }
            return create_numpy_array(data_vec, shape_vec);
        }, "_test_create_array");
    },
    R"pbdoc(
        Test helper: Create numpy array from data and shape.
        Used for testing array creation.
    )pbdoc");
    
    // ========================================================================
    // Test helper functions for exception translation
    // ========================================================================
    
    m.def("_test_throw_invalid_argument", [](std::string message) {
        return translate_exceptions([&]() -> int {
            throw std::invalid_argument(message);
        }, "_test_throw_invalid_argument");
    },
    R"pbdoc(
        Test helper: Throw std::invalid_argument.
        Used for testing exception translation to ValueError.
    )pbdoc");
    
    m.def("_test_throw_out_of_range", [](std::string message) {
        return translate_exceptions([&]() -> int {
            throw std::out_of_range(message);
        }, "_test_throw_out_of_range");
    },
    R"pbdoc(
        Test helper: Throw std::out_of_range.
        Used for testing exception translation to IndexError.
    )pbdoc");
    
    m.def("_test_throw_runtime_error", [](std::string message) {
        return translate_exceptions([&]() -> int {
            throw std::runtime_error(message);
        }, "_test_throw_runtime_error");
    },
    R"pbdoc(
        Test helper: Throw std::runtime_error.
        Used for testing exception translation to RuntimeError.
    )pbdoc");
    
    m.def("_test_throw_generic_exception", [](std::string message) {
        return translate_exceptions([&]() -> int {
            throw std::logic_error(message);  // Generic std::exception subclass
        }, "_test_throw_generic_exception");
    },
    R"pbdoc(
        Test helper: Throw generic std::exception.
        Used for testing exception translation to RuntimeError with context.
    )pbdoc");
}
