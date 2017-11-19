This repository is supposed to contain some benchmarks
for various things. However, at this time of writing,
only a benchmark for the dot product in context of
artificial networks exists in the ```ann``` folder.

Look into the source and makefile for more explanations
how to modify the examples.

In order to try out, just enter

```
$ make -C ann benchmark
```

On my computer, an i5-2540M equipped one, this leads to something
like this:

```
Time(ms) Stddev   Output    Options
0.559013 0.000009 26.368052 "-O0"
0.178039 0.000005 26.368052 "-Os"
0.179200 0.000005 26.368052 "-O3"
0.179745 0.000007 26.368052 "-O2 -ftree-vectorize "
0.049078 0.000003 26.368038 "-O2 -ftree-vectorize -ffast-math "
0.047697 0.000004 26.368038 "-O2 -DUSE_INTRINSICS"
0.040484 0.000004 26.368038 "-O2 -DUSE_INTRINSICS -funroll-all-loops"
0.083281 0.000004 26.368044 "-O2 -DUSE_DP_INTRINSICS -funroll-all-loops"
```
