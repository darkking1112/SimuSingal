cmake_minimum_required(VERSION 3.16)
project(native_plugin_demo LANGUAGES C)

add_library(demo_plugin SHARED demo_plugin.c)
target_compile_definitions(demo_plugin PRIVATE DEMO_BUILD)
set_target_properties(demo_plugin PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED YES
    C_VISIBILITY_PRESET hidden
)
if(MSVC)
    target_compile_options(demo_plugin PRIVATE /W4)
else()
    target_compile_options(demo_plugin PRIVATE -Wall -Wextra -Werror)
endif()
