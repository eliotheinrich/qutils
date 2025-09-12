set(qutils_test_libs clifford_state quantum_state free_fermion)

foreach(tgt IN ITEMS test_graph test_quantum_circuit test_quantum_state test_free_fermion test_clifford_state benchmark)
  add_executable(${tgt} ${tgt}.cpp)
  target_link_libraries(${tgt} PRIVATE ${qutils_test_libs})
endforeach()
