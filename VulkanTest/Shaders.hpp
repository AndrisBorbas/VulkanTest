#pragma once

#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#define VULKAN_HPP_NO_NODISCARD_WARNINGS

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
		vk::VertexInputBindingDescription bindingDescription{
			.binding = 0, .stride = sizeof(Vertex), .inputRate = vk::VertexInputRate::eVertex};

		return bindingDescription;
	}

	static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions()
	{
		std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions{};
		attributeDescriptions[0] = {.location = 0,
									.binding  = 0,
									.format   = vk::Format::eR32G32Sfloat,
									.offset   = offsetof(Vertex, pos)};
		attributeDescriptions[1] = {.location = 1,
									.binding  = 0,
									.format   = vk::Format::eR32G32B32Sfloat,
									.offset   = offsetof(Vertex, color)};

		return attributeDescriptions;
	}
};

uint32_t findMemoryType(vk::PhysicalDevice physicalDevice,
						uint32_t typeFilter,
						vk::MemoryPropertyFlags properties)
{
	vk::PhysicalDeviceMemoryProperties memProperties;
	physicalDevice.getMemoryProperties(&memProperties);

	for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
		if ((typeFilter & (1 << i))
			&& (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
			return i;
		}
	}

	throw std::runtime_error("failed to find suitable memory type!");
}
