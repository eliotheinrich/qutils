find_package(Python 3.10 
    REQUIRED COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule
)

# Finding nanobind
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR
)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(
    qutils_bindings
    STABLE_ABI
    NB_STATIC
    PyQutils.hpp
    PyQutils.cpp
)

target_link_libraries(qutils_bindings PUBLIC quantum_circuit quantum_state clifford_state free_fermion)

install(
    TARGETS qutils_bindings
    LIBRARY DESTINATION qutils
)
