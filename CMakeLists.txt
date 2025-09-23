cmake_minimum_required(VERSION 3.26)

project(qutils VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)

include(CheckCXXCompilerFlag)

add_compile_options(-pthread -w)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
if(COMPILER_SUPPORTS_AVX2)
  add_compile_options(-mavx2)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g") 
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


# -------------------------------- #
# ------ Fetching externals ------ #
# -------------------------------- #

option(QUTILS_BUILD_TESTS "Build qutils tests" ON)
option(QUTILS_BUILD_PYTHON "Build Python bindings" OFF)

include(FetchContent)

set(EXTERNALS "")

FetchContent_Declare(
  glaze
  GIT_REPOSITORY https://github.com/stephenberry/glaze.git
  GIT_TAG cc4ce0efe5133c23ec55f07d9e753f7dd1abb0b8
)
list(APPEND EXTERNALS glaze)

FetchContent_MakeAvailable(${EXTERNALS})

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(fmt REQUIRED)

# -------------------------------- #
# -------- Adding code ----------- #
# -------------------------------- #

include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${glaze_SOURCE_DIR}/include
)

add_subdirectory(${PROJECT_SOURCE_DIR}/src/QuantumCircuit)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/QuantumState)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/FreeFermion)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/CliffordState)

foreach (tgt IN ITEMS quantum_state quantum_circuit free_fermion clifford_state)
    target_link_libraries(${tgt} PUBLIC Eigen3::Eigen)
    target_link_libraries(${tgt} PUBLIC fmt::fmt)
endforeach()

if (QUTILS_BUILD_TESTS)
    add_subdirectory(${PROJECT_SOURCE_DIR}/src/Tests)
endif()

if (QUTILS_BUILD_PYTHON)
    add_subdirectory(${PROJECT_SOURCE_DIR}/src/Python)
else()
    include(CMakePackageConfigHelpers)
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/qutilsConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY AnyNewerVersion
    )

    configure_package_config_file(
        "${CMAKE_CURRENT_LIST_DIR}/cmake/qutilsConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/qutilsConfig.cmake"
        INSTALL_DESTINATION lib/cmake/qutils
    )

    install(TARGETS quantum_circuit quantum_state free_fermion clifford_state 
            EXPORT qutilsTargets
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib
            RUNTIME DESTINATION bin
    )

    install(EXPORT qutilsTargets
        FILE qutilsTargets.cmake
        NAMESPACE qutils::
        DESTINATION lib/cmake/qutils
    )

    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/qutilsConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/qutilsConfigVersion.cmake"
        DESTINATION lib/cmake/qutils
    )

    install(
        DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/
        DESTINATION include/qutils
        FILES_MATCHING 
        PATTERN "*.h"
        PATTERN "*.hpp"
    )
endif()
