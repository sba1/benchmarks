/**
 * This is a benchmark for how compiler options influence the execution
 * speed of artifical neuronal networks. For now, only the inference is
 * measured. Training is not implemented. Weights as well as input are
 * randomly chosen (but fixed with a seed).
 *
 * The code is based on https://github.com/codeplea/genann.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <memory.h>
#include <time.h>

/* For the intrinsic version */
#if defined(USE_INTRINSICS) || defined(USE_DP_INTRINSICS)
#include <mmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <smmintrin.h>
#endif

/**
 * Our neuronal network representation. Only fully connected networks are
 * supported for now.
 */
typedef struct
{
  int inputs;
  int hiddens;
  int hidden_layers;
  int outputs;
  int total_nodes;
  int *nodes_in_layer;
  int weights;
  float *weight;

  float *output;
} ann;

/**
 * Free all memory associated with the ann. The structure holding the members
 * is not freed.
 *
 * @param ann
 */
void ann_free(ann *ann)
{
  if (!ann)
  {
    return;
  }
  free(ann->nodes_in_layer);
  ann->nodes_in_layer = NULL;
  free(ann->weight);
  ann->weight = NULL;
  free(ann->output);
  ann->output = NULL;  
}

/**
 * Initialize the artificial neuronal network with the given parameter and
 * allocate enough space. A fully connected neuronal network is instanciated.
 *
 * @param inputs number of input neurons
 * @param outputs number of output neurons
 * @param hiddens number of neurons per hidden layer
 * @param hidden_layers number of hidden layers
 * @return an error code or 0 on success.
 */
int ann_init(ann *ann, int inputs, int outputs, int hiddens, int hidden_layers)
{
  int l;
  int weights = 0;

  memset(ann, 0, sizeof(*ann));
  ann->inputs = inputs;
  ann->outputs = outputs;
  ann->hidden_layers = hidden_layers;
  ann->hiddens = hiddens;
  ann->total_nodes = inputs + outputs + hidden_layers * hiddens;

  if (!(ann->nodes_in_layer = (int*)malloc(sizeof(int) * (hidden_layers + 2))))
  {
    return -1;
  }

  ann->nodes_in_layer[0] = inputs;
  for (l = 1; l < hidden_layers + 1; l++)
  {
    ann->nodes_in_layer[l] = hiddens;
    weights += ann->nodes_in_layer[l-1] * hiddens;
  }
  ann->nodes_in_layer[l] = outputs;
  weights += hiddens * outputs;

  ann->weights = weights;
  if (!(ann->weight = (float*)malloc(sizeof(float) * weights)))
  {
    free(ann->weight);
    return -1;
  }

  /* Initialize weights randomly */
  for (int i = 0; i < weights; i++)
  {
    ann->weight[i] = (rand() - RAND_MAX / 2) / (float)RAND_MAX;
  }

  if (!(ann->output = (float*)malloc(sizeof(float) * ann->total_nodes)))
  {
    free(ann->weight);
    ann->weight = NULL;
    free(ann->output);
    ann->output = NULL;
    return -1;
  }

  return 0;
}

/**
 * Calculate the dot product for the given vectors a and b that contain
 * size elements.
 *
 * @param a
 * @param b
 * @param size
 */
float dotp(const float *a, const float *b, int size)
{
#if defined(USE_INTRINSICS)
  __m128 prod, *ma = (__m128 *)a, *mb=(__m128 *)b;
  __m128 sum = {0, 0, 0, 0};

  for (int i = 0; i < size / 4; i++)
  {
    prod = _mm_mul_ps(ma[i], mb[i]);
    sum = _mm_add_ps(sum, prod);
  }

  /* Reduce sum vector, needs at least sse3 */
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);

  return sum[0];
#elif defined(USE_DP_INTRINSICS)
  __m128 *ma = (__m128 *)a, *mb=(__m128 *)b;
  float sum = 0.0;

  for (int i = 0; i < size / 4; i++)
  {
    /* Needs at least SSE4 */
    __m128 dp = _mm_dp_ps(ma[i], mb[i], 0xf1);
    sum += dp[0];
  }
  return sum;
#else
  float sum = 0;
  for (int i = 0; i < size; i++)
  {
    sum += *a++ * *b++;
  }
  return sum;
#endif
}

/**
 * Perform a full inference step for the network for the given input.
 *
 * @param ann the artificial network on which to perform the inference
 * @param inputs pointer to the first input value
 * @return the pointer to the first output value
 */
float *ann_inference(ann *ann, const float *inputs)
{
  float *w, *i, *o;

  memcpy(ann->output, inputs, sizeof(*inputs) * ann->inputs);

  w = ann->weight;
  i = ann->output;
  o = ann->output + ann->inputs;

  for (int l = 1; l < ann->hidden_layers + 2; l++)
  {
    /* Calculate the output of a single layer */
    for (int h = 0; h < ann->nodes_in_layer[l]; h++)
    {
      /* Calculate the output of a single neuron */
      int nodes_in_prev_layer = ann->nodes_in_layer[l-1];
      float sum = dotp(i, w, nodes_in_prev_layer);
      w += nodes_in_prev_layer;
      /* No activation function for now */
      *o++ = sum;
    }
    i += ann->nodes_in_layer[l-1];
  }
  return ann->output + ann->total_nodes - ann->outputs;
}

/******************************************************************************/

#define INPUTS (18*18)
#define HIDDEN (30*20)

/* Number of retries of the inference step */
#define RERUNS 1000

static float input[INPUTS];

int main(int argc, char *argv[])
{
  int err;
  ann ann;
  float *output;
  double runtime[RERUNS] = {0.0};
  double mean = 0.0;
  double stddev = 0.0;

  clock_t start, end;
  
  if (argc > 1)
  {
    if (!strcmp(argv[1], "--print-header-only"))
    {
      printf("Time(ms) Stddev   Output    Options\n");
      return 0;
    }
  }
  
  srand(1);

  if ((err = ann_init(&ann, INPUTS, 2, HIDDEN, 1)))
  {
    fprintf(stderr, "Couldn't initialize ann\n");
    goto out;
  }

  /* Put some random values into the input, in reality this would be e.g., 
  * a picture */
  for (int i = 0; i < INPUTS; i++)
  {
    input[i] = rand() / (float)RAND_MAX;
  }

  for (int i = 0; i < RERUNS; i++)
  {
    start = clock();
    output = ann_inference(&ann, input);
    end = clock();

    runtime[i] = (double)(end - start) / CLOCKS_PER_SEC;
  }

  /* Calculate run time statistics */
  for (int i = 0; i < RERUNS; i++)
  {
    mean += runtime[i];
  }
  mean = mean / RERUNS;

  for (int i = 0; i < RERUNS; i++)
  {
    stddev += pow(runtime[i] - mean, 2);
  }
  stddev = sqrt(stddev / RERUNS);

  printf("%lf %lf %f \"%s\"\n", mean * 1000, stddev, output[0], CFLAGS);
  ann_free(&ann);
  return 0;
out:
  return 1;
}
