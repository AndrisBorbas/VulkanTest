cmake_minimum_required(VERSION 3.19)
project(VulkanTest)

# Get OS information
if (CMAKE_HOST_WIN32)
    set(WINDOWS 1)
elseif (CMAKE_HOST_APPLE)
    set(MACOS 1)
elseif (CMAKE_HOST_UNIX)
    set(LINUX 1)
endif ()

if (CMAKE_COMPILER_IS_GNUCXX OR (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang"))
    message("Compiler is GNU compatible.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-missing-field-initializers -Wno-unused-function -Wno-unused-variable -DVK_PROTOTYPES=1 -pthread")
elseif (MSVC)
    message("Compiler is MSVC.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS /wd4267 /wd4244 /wd4146")
endif ()

set(CMAKE_CXX_STANDARD 20)

include_directories(Libraries/glfw-3.3.4.bin.WIN64)
include_directories(Libraries/glfw-3.3.4.bin.WIN64/include)
include_directories(Libraries/glfw-3.3.4.bin.WIN64/include/GLFW)
include_directories(Libraries/glm/glm)
include_directories(Libraries/glm/glm/detail)
include_directories(Libraries/glm/glm/ext)
include_directories(Libraries/glm/glm/gtc)
include_directories(Libraries/glm/glm/gtx)
include_directories(Libraries/glm/glm/simd)
include_directories(VulkanTest)

add_executable(VulkanTest
        Libraries/glfw-3.3.4.bin.WIN64/include/GLFW/glfw3.h
        Libraries/glfw-3.3.4.bin.WIN64/include/GLFW/glfw3native.h
        Libraries/glm/glm/glm.hpp
        VulkanTest/main.cpp
        VulkanTest/Shaders.hpp
        VulkanTest/utils.hpp
        VulkanTest/VulkanDevices.hpp
        VulkanTest/VulkanInit.hpp
        VulkanTest/VulkanRendering.hpp
        VulkanTest/VulkanInit.cpp
        VulkanTest/Defines.hpp
        )

### GLFW
set(GLFW3_LIBRARY C:/SDKs/glfw-3.3.4.bin.WIN64/lib-vc2019/glfw3.lib)
include_directories(${PROJECT_NAME} C:/SDKs/glfw-3.3.4.bin.WIN64/include)


####  GLM
set(GLM_INCLUDE_DIRS C:/SDKs/glm)
include_directories(${GLM_INCLUDE_DIRS})

####Vulkan
find_package(Vulkan REQUIRED)
target_include_directories(${PROJECT_NAME} PUBLIC ${Vulkan_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} Vulkan::Vulkan ${GLFW3_LIBRARY})

#add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD
#        COMMAND ${CMAKE_COMMAND} -E copy_directory
#        ${CMAKE_SOURCE_DIR}/${PROJECT_NAME}/shaders/ $<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders)
#
#if (WIN32)
#    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
#            COMMAND cmd /C ${CMAKE_SOURCE_DIR}/${PROJECT_NAME}/compileShaders.bat)
#
#endif ()

include(Shaders.cmake)

# Find all shaders.
file(GLOB vertex-shaders ${CMAKE_SOURCE_DIR}/${PROJECT_NAME}/shaders/*.vert)
file(GLOB fragment-shaders ${CMAKE_SOURCE_DIR}/${PROJECT_NAME}/shaders/*.frag)
file(GLOB compute-shaders ${CMAKE_SOURCE_DIR}/${PROJECT_NAME}/shaders/*.comp)

# Add them to the build.
foreach (vertex-shader ${vertex-shaders})
    get_filename_component(p ${vertex-shader} NAME)
    add_shader(${PROJECT_NAME} ${p})
endforeach (vertex-shader)

foreach (fragment-shader ${fragment-shaders})
    get_filename_component(p ${fragment-shader} NAME)
    add_shader(${PROJECT_NAME} ${p})
endforeach (fragment-shader)

foreach (compute-shader ${compute-shaders})
    get_filename_component(p ${compute-shader} NAME)
    add_shader(${PROJECT_NAME} ${p})
endforeach (compute-shader)


set(output-assets $<TARGET_FILE_DIR:${PROJECT_NAME}>/${TARGET}/assets)


# Copy assets from sample to the appropriate location.
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/assets/textures)
    file(MAKE_DIRECTORY ${output-assets})
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/assets/textures ${output-assets}/textures)
endif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/assets/textures)


if (CMAKE_CONFIGURATION_TYPES)
    string(TOLOWER "${CMAKE_CONFIGURATION_TYPES}" CMAKE_CONFIGURATION_TYPES_LOWER)
else ()
    string(TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_CONFIGURATION_TYPES_LOWER)
endif ()

foreach (config ${CMAKE_CONFIGURATION_TYPES_LOWER})
    if (${config} MATCHES "debug")
        #add preprocessor definition something like this bellow
        add_definitions(-DENABLE_VALIDATION_LAYERS)
    elseif (${config} MATCHES "release")
        #and so on...
    endif ()
endforeach ()
