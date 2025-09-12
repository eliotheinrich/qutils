add_library(clifford_state 
    CliffordState.cpp
    QuantumCHPState.cpp
    QuantumGraphState.cpp
    Tableau.cpp
    TableauSIMD.cpp
)


target_link_libraries(clifford_state
    PRIVATE quantum_state
)

target_include_directories(clifford_state
    PRIVATE ${PROJECT_SOURCE_DIR}/src
    PRIVATE ${glaze_SOURCE_DIR}/include
)
