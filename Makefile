# Use ?= so an environment- or command-line-supplied compiler is honoured
# (e.g. `make CC=clang`, or a cross-compiler in CI).
CC ?= gcc
CFLAGS ?= -Wall -Wextra -O3
# -MMD -MP emit .d files so editing a header (kseq.h, khashl.h, ...) triggers a
# rebuild instead of silently leaving stale objects behind.
ALL_CFLAGS = $(CFLAGS) -pthread -I. -MMD -MP
LDFLAGS ?=
LDLIBS = -lz -pthread

PREFIX ?= /usr/local
BINDIR = $(DESTDIR)$(PREFIX)/bin

# fgr2 is the supported tool. fgr is legacy (non-streaming, forward-strand-only)
# and is built only via `make legacy`.
BINS = fgr2
LEGACY_BINS = fgr

OBJS = fgr2.o kthread.o
LEGACY_OBJS = fgr.o
DEPS = $(OBJS:.o=.d) $(LEGACY_OBJS:.o=.d)

.PHONY: all legacy clean check install uninstall debug

all: $(BINS)

legacy: $(LEGACY_BINS)

# Compilation rule
%.o: %.c
	$(CC) $(ALL_CFLAGS) -c $< -o $@

# Linking rules
fgr2: fgr2.o kthread.o
	$(CC) $(ALL_CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

fgr: fgr.o
	$(CC) $(ALL_CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# Unoptimised build with symbols and sanitizers, for debugging.
debug:
	$(MAKE) clean
	$(MAKE) CFLAGS="-Wall -Wextra -O0 -g -fsanitize=address,undefined"

# Run the test suite (see tests/run_tests.sh).
check: $(BINS)
	@./tests/run_tests.sh

install: $(BINS)
	install -d $(BINDIR)
	install -m 755 fgr2 $(BINDIR)/fgr2
	install -m 755 run_fgr2.py $(BINDIR)/run_fgr2.py
	install -m 755 calculate_similarity.py $(BINDIR)/calculate_similarity.py

uninstall:
	rm -f $(BINDIR)/fgr2 $(BINDIR)/run_fgr2.py $(BINDIR)/calculate_similarity.py

clean:
	rm -f $(OBJS) $(LEGACY_OBJS) $(DEPS) $(BINS) $(LEGACY_BINS)

-include $(DEPS)
