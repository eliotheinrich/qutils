#set(qutils_test_libs quantum_state quantum_circuit clifford_state free_fermion)
#set(qutils_test_libs quantum_circuit clifford_state free_fermion)
set(qutils_test_libs clifford_state quantum_circuit quantum_state free_fermion)

foreach(tgt IN ITEMS test_graph test_quantum_circuit test_quantum_state benchmark)
  add_executable(${tgt} ${tgt}.cpp)
  target_link_libraries(${tgt} PRIVATE ${qutils_test_libs})
  #add_test(NAME ${tgt} COMMAND ${tgt})
endforeach()
