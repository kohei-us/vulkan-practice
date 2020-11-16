CFLAGS=-std=c++17 -O2 -g
LDFLAGS=-lglfw -lvulkan -ldl -lpthread -lX11 -lXrandr -lXi
APPS= \
	vulkan-test \
	vulkan-triangle


all: $(APPS)

vulkan-test: test.cpp
	g++ $(CFLAGS) -o vulkan-test test.cpp $(LDFLAGS)

vulkan-triangle: triangle.cpp
	g++ $(CFLAGS) -o vulkan-triangle triangle.cpp $(LDFLAGS)

.PHONY: test clean

clean:
	rm -f $(APPS)
