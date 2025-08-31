add_library(clifford_state 
    CliffordState.cpp
    QuantumCHPState.cpp
    QuantumGraphState.cpp
    Tableau.cpp
)

set_target_properties(clifford_state PROPERTIES 
    LINKER_LANGUAGE CXX 
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(clifford_state PUBLIC quantum_state)
