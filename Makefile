STB_INCLUDE_PATH=external/stb

CPPFLAGS=-std=c++17 -O2 -g -I$(STB_INCLUDE_PATH)
LDFLAGS=-lglfw -lvulkan -ldl -lpthread -lX11 -lXrandr -lXi
APPS= \
	vulkan-test \
	vulkan-triangle


all: $(APPS)

vulkan-test: test.cpp
	g++ $(CPPFLAGS) -o vulkan-test test.cpp $(LDFLAGS)

vulkan-triangle: triangle.cpp
	g++ $(CPPFLAGS) -o vulkan-triangle triangle.cpp $(LDFLAGS)

.PHONY: test clean

clean:
	rm -f $(APPS)
