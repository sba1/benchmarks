#
# This makefile contains the rules to compile different flavors of the
# ann.c source code.
#
# In order to test a new flavor, add a new target and define a corresponding
# CFLAGS marco that contains the special compiler flags. Use macros inside
# the code and the -D switch to test diferent code implementations.
#
# Currently, support for x86_64 and armv7l (cortex-a53 on the raspberry pi 3)
# is built in.
#

TARGETS = ann1 ann2 ann3 ann4 ann5 ann6

CFLAGS = -std=c11 -Wall

ifeq ($(shell arch),x86_64)

CFLAGS += -msse4
CFLAGS1 = -O0
CFLAGS2 = -Os
CFLAGS3 = -O2 
CFLAGS3 = -O3
CFLAGS4 = -O2 -ftree-vectorize # -fopt-info-vec-all
CFLAGS5 = -O2 -ftree-vectorize -ffast-math # -fopt-info-vec-all
CFLAGS6 = -O2 -DUSE_INTRINSICS
TARGETS += ann7
CFLAGS7 = -O2 -DUSE_INTRINSICS -funroll-all-loops
TARGETS += ann8
CFLAGS8 = -O2 -DUSE_DP_INTRINSICS -funroll-all-loops

else ifeq ($(shell arch),armv7l)

CFLAGS1 = -O0
CFLAGS2 = -Os
CFLAGS3 = -O3
CFLAGS4 = -O2 -mcpu=cortex-a53 -mfloat-abi=hard -mfpu=neon-fp-armv8
CFLAGS5 = -O2 -mcpu=cortex-a53 -mfloat-abi=hard -mfpu=neon-fp-armv8 -ftree-vectorize
CFLAGS6 = -O2 -mcpu=cortex-a53 -mfloat-abi=hard -mfpu=neon-fp-armv8 -ftree-vectorize -ffast-math

else

CFLAGS1 = -O0
CFLAGS2 = -Os
CFLAGS3 = -O3
CFLAGS4 = -O2
CFLAGS5 = -O2 -ftree-vectorize
CFLAGS6 = -O2 -ftree-vectorize -ffast-math

endif

.PHONY: all
all: $(TARGETS)

# Macro to compile a flavor of ann
define cc_ann
$(1): ann.c
	gcc $(CFLAGS) $$< -o $$@ $(CFLAGS$(1:ann%=%)) -DCFLAGS='"$(CFLAGS$(1:ann%=%))"' -lm
endef

$(foreach target,$(TARGETS),$(eval $(call cc_ann, $(target))))

.PHONY: clean
clean:
	rm -f $(TARGETS)

.PHONY: benchmark
benchmark: $(TARGETS)
	./ann1 --print-header-only && $(foreach e,$(TARGETS), ./$(e) && ) true
