add_library(free_fermion
    FreeFermionState.cpp
    GaussianState.cpp
    MajoranaState.cpp
)

set_target_properties(free_fermion PROPERTIES 
    LINKER_LANGUAGE CXX 
    POSITION_INDEPENDENT_CODE ON
)
