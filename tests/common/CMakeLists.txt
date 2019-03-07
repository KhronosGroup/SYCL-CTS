add_library(main_function_object OBJECT main.cpp)
add_library(main_function INTERFACE)
add_library(CTS::main_function ALIAS main_function)
target_sources(main_function INTERFACE $<TARGET_OBJECTS:main_function_object>)
