CC = clang++
CFLAGS = -std=c++11 -Wall -Wextra -pedantic -O3 -march=native -mtune=native

all: rows_to_tiles
rows_to_tiles: rows_to_tiles.cpp
	$(CC) $(CFLAGS) -o rows_to_tiles rows_to_tiles.cpp