CC = gcc
CFLAGS = -Wall -Wextra -O3 -pthread -I.
LDFLAGS = -lz -pthread

# Source files
SRCS = fgr.c fgr2.c
OBJS = $(SRCS:.c=.o)
BINS = $(SRCS:.c=)

# Default target
all: $(BINS)

# Compilation rules
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Linking rules
fgr: fgr.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

fgr2: fgr2.o kthread.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Clean target
clean:
	rm -f $(OBJS) $(BINS) kthread.o

.PHONY: all clean 