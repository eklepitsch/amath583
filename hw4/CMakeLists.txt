cmake_minimum_required(VERSION 3.14)
project(my_project)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

include_directories(include src)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(
  matrix_class_test
  src/matrix_class_test.cpp
)
target_link_libraries(
  matrix_class_test
  GTest::gtest_main
)

add_executable(
  file_swaps_test
  src/file_swaps_test.cpp
)
target_link_libraries(
  file_swaps_test
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(matrix_class_test)
gtest_discover_tests(file_swaps_test)

add_executable(
  hw4_p4
  src/p4.cpp
)

add_executable(
  hw4_p5
  src/p5.cpp
)

add_executable(
  hw4_p6
  src/p6.cpp
)

target_link_libraries(
  hw4_p6
  pthread
)

# hw4_p7 and hw4_p9 use MPI and are compiled directly
# in build.sh.

add_executable(
  hw4_p10
  src/p10.cpp
)

add_executable(
  strassen_test
  src/strassen_test.cpp
)

target_link_libraries(
  strassen_test
  GTest::gtest_main
)