cmake_minimum_required(VERSION 3.26)

project(qutils LANGUAGES CXX)

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
set(CMAKE_PREFIX_PATH $ENV{CMAKE_PREFIX_PATH})
string(REPLACE ":" ";" CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


# -------------------------------- #
# ------ Fetching externals ------ #
# -------------------------------- #

foreach(PATH IN LISTS CMAKE_PREFIX_PATH)
    include_directories("${PATH}/include")
endforeach()

option(QUTILS_BUILD_TESTS "Build qutils tests" ON)
option(QUTILS_BUILD_PYTHON "Build Python bindings" OFF)

add_definitions(-DFMT_HEADER_ONLY)

include(FetchContent)

set(EXTERNALS "")

FetchContent_Declare(
  glaze
  GIT_REPOSITORY https://github.com/stephenberry/glaze.git
  GIT_TAG cc4ce0efe5133c23ec55f07d9e753f7dd1abb0b8
)
list(APPEND EXTERNALS glaze)

FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen
    GIT_TAG 3.4
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(EIGEN_BUILD_DOC OFF)
set(BUILD_TESTING OFF)
set(EIGEN_BUILD_PKGCONFIG OFF)
list(APPEND EXTERNALS eigen)

FetchContent_MakeAvailable(${EXTERNALS})

# -------------------------------- #
# -------- Adding code ----------- #
# -------------------------------- #

include_directories(
    ${eigen_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
    ${glaze_SOURCE_DIR}/include
)

add_subdirectory(${PROJECT_SOURCE_DIR}/src/QuantumCircuit)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/QuantumState)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/FreeFermion)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/CliffordState)

if (QUTILS_BUILD_TESTS)
    add_subdirectory(${PROJECT_SOURCE_DIR}/src/Tests)
endif()

if (QUTILS_BUILD_PYTHON)
    add_subdirectory(${PROJECT_SOURCE_DIR}/src/Python)
endif()
