// INCLUDE/field_instantiate.cpp
#include "structs.cuh"

// Include the implementation file so the compiler can see the template definitions
#include "field.cpp"
#include "field.cu"

// Explicit template instantiations for Field<T>
template class Field<real_t>;
template class Field<int>;
template class Field<bool>;
template class Field<real2_t>;
template class Field<real3_t>;
template class Field<real4_t>;