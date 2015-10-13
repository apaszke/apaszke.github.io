CC = gcc
CFLAGS = -Wall -pedantic

.PHONY: all clean

all: read write

clean:
	@rm *.o 2>/dev/null || true
	@rm read 2>/dev/null || true
	@rm write 2>/dev/null || true

read: shared_read.o shared_common.o
	$(CC) $(CFLAGS) shared_read.o shared_common.o -o read

write: shared_write.o shared_common.o
	$(CC) $(CFLAGS) shared_write.o shared_common.o -o write
