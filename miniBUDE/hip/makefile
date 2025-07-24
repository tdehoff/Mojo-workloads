CC = hipcc

# Include fast-math
# FLAGS=-std=c++17 -O3 -ffast-math
# Without fast-math
FLAGS=-std=c++17 -O3

SRCS = main.cpp
HEADERS = bude.h fasten.hpp
TARGET = bude_hip

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(CC) $(FLAGS) -o $@ $(SRCS)

clean:
	rm -f $(TARGET)