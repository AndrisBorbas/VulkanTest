#pragma once

#include <vulkan/vulkan.hpp>

#include <array>
#include <glm/glm.hpp>
#include <vector>

struct Vertex
{
	glm::vec2 pos;
	glm::vec3 color;

	static vk::VertexInputBindingDescription getBindingDescription()
	{
		vk::VertexInputBindingDescription bindingDescription;
		bindingDescription.binding   = 0;
		bindingDescription.stride    = sizeof(Vertex);
		bindingDescription.inputRate = vk::VertexInputRate::eVertex;

		return bindingDescription;
	}

	static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions()
	{
		std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions;

		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].binding  = 0;
		attributeDescriptions[0].format   = vk::Format::eR32G32Sfloat;
		attributeDescriptions[0].offset   = offsetof(Vertex, pos);

		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].binding  = 0;
		attributeDescriptions[1].format   = vk::Format::eR32G32B32Sfloat;
		attributeDescriptions[1].offset   = offsetof(Vertex, color);

		return attributeDescriptions;
	}
};

struct UniformBufferObject
{
	alignas(16) glm::mat4 model;
	alignas(16) glm::mat4 view;
	alignas(16) glm::mat4 proj;
};

uint32_t findMemoryType(vk::PhysicalDevice physicalDevice,
						uint32_t typeFilter,
						vk::MemoryPropertyFlags properties);
