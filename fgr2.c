#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>
#include "ketopt.h" // command-line argument parser
#include "kthread.h" // multi-threading models: pipeline and multi-threaded for loop

#include "kseq.h" // FASTA/Q parser
KSEQ_INIT(gzFile, gzread)

#include "khashl.h" // hash table
#define KC_BITS 14
#define KC_MAX ((1<<KC_BITS) - 1)
#define kc_c4_eq(a, b) ((a)>>KC_BITS == (b)>>KC_BITS) // lower 8 bits for counts; higher bits for k-mer
#define kc_c4_hash(a) ((a)>>KC_BITS)

#define ROTL64(x,r) ((x << r) | (x >> (64 - r)))  // Rotate left Lifu Song

KHASHL_SET_INIT(, kc_c4_t, kc_c4, uint64_t, kc_c4_hash, kc_c4_eq)

#define CALLOC(ptr, len) ((ptr) = (__typeof__(ptr))calloc((len), sizeof(*(ptr))))
#define MALLOC(ptr, len) ((ptr) = (__typeof__(ptr))malloc((len) * sizeof(*(ptr))))
#define REALLOC(ptr, len) ((ptr) = (__typeof__(ptr))realloc((ptr), (len) * sizeof(*(ptr))))

const unsigned char seq_nt4_table[256] = { // translate ACGT to 0123
	0, 1, 2, 3,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  3, 3, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  3, 3, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4
};

static inline uint64_t hash64(uint64_t key, uint64_t mask) // invertible integer hash function
{
	key = (~key + (key << 21)) & mask; // key = (key << 21) - key - 1;
	key = key ^ key >> 24;
	key = ((key + (key << 3)) + (key << 8)) & mask; // key * 265
	key = key ^ key >> 14;
	key = ((key + (key << 2)) + (key << 4)) & mask; // key * 21
	key = key ^ key >> 28;
	key = (key + (key << 31)) & mask;
	return key;
}

static inline uint64_t hash64i(uint64_t key, uint64_t mask)
{
	uint64_t tmp;

	// Invert key = key + (key << 31)
	tmp = (key - (key << 31)) & mask; key = (key - (tmp << 31)) & mask;

	// Invert key = key ^ (key >> 28)
	tmp = key ^ key >> 28; key = key ^ tmp >> 28;

	// Invert key *= 21
	key = (key * 14933078535860113213ull) & mask;

	// Invert key = key ^ (key >> 14)
	tmp = key ^ key >> 14; tmp = key ^ tmp >> 14; tmp = key ^ tmp >> 14; key = key ^ tmp >> 14;

	// Invert key *= 265
	key = (key * 15244667743933553977ull) & mask;

	// Invert key = key ^ (key >> 24)
	tmp = key ^ key >> 24; key = key ^ tmp >> 24;

	// Invert key = (~key) + (key << 21)
	tmp = ~key; tmp = ~(key - (tmp << 21)) & mask; tmp = ~(key - (tmp << 21)) & mask; key = ~(key - (tmp << 21)) & mask;

	return key;
}

static inline uint64_t hash_wang(uint64_t key)
{
	// Implementation of Thomas Wang's 64-bit integer hash function
	key = (~key) + (key << 21);
	key = key ^ (key >> 24);
	key = (key + (key << 3)) + (key << 8);
	key = key ^ (key >> 14);
	key = (key + (key << 2)) + (key << 4);
	key = key ^ (key >> 28);
	key = key + (key << 31);
	return key;
}



static inline uint64_t fmix64(uint64_t k)
{
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k;
}

uint64_t MurmurHash3_x64_64(const void* key, int len, uint64_t seed)
{
    const uint8_t* data = (const uint8_t*)key;
    const int nblocks = len / 8;
    int i;

    uint64_t h1 = seed;
    uint64_t h2 = seed;

    const uint64_t c1 = 0x87c37b91114253d5ULL;
    const uint64_t c2 = 0x4cf5ad432745937fULL;

    // Body
    const uint64_t* blocks = (const uint64_t*)(data);

    for (i = 0; i < nblocks; i++) {
        uint64_t k1 = blocks[i];

        k1 *= c1;
        k1 = ROTL64(k1, 31);
        k1 *= c2;

        h1 ^= k1;
        h1 = ROTL64(h1, 27);
        h1 += h2;
        h1 = h1 * 5 + 0x52dce729;

        h2 = ROTL64(h2, 31);
        h2 += h1;
        h2 = h2 * 5 + 0x38495ab5;
    }

    // Tail
    const uint8_t* tail = (const uint8_t*)(data + nblocks * 8);
    uint64_t k1 = 0;

    switch (len & 7) {
    case 7: k1 ^= ((uint64_t)tail[6]) << 48;
            /* FALLTHROUGH */
    case 6: k1 ^= ((uint64_t)tail[5]) << 40;
            /* FALLTHROUGH */
    case 5: k1 ^= ((uint64_t)tail[4]) << 32;
            /* FALLTHROUGH */
    case 4: k1 ^= ((uint64_t)tail[3]) << 24;
            /* FALLTHROUGH */
    case 3: k1 ^= ((uint64_t)tail[2]) << 16;
            /* FALLTHROUGH */
    case 2: k1 ^= ((uint64_t)tail[1]) << 8;
            /* FALLTHROUGH */
    case 1: k1 ^= ((uint64_t)tail[0]) << 0;
            k1 *= c1; k1 = ROTL64(k1, 31); k1 *= c2; h1 ^= k1;
    };

    // Finalization
    h1 ^= len;
    h2 ^= len;

    h1 += h2;
    h2 += h1;

    h1 = fmix64(h1);
    h2 = fmix64(h2);

    h1 += h2;
    h2 += h1;

    return h1;
}

static inline uint64_t hash_murmur3(uint64_t key)
{
	// uint64_t h;
	// uint32_t seed = 42; // Arbitrary seed
	// MurmurHash3_x64_128(&key, sizeof(key), seed, &h);
	// return h;

	return MurmurHash3_x64_64(&key, sizeof(key), 42); // Use 42 as a seed, or choose another value
}

// Function pointer for the selected hash function
static uint64_t (*selected_hash_func)(uint64_t) = NULL;

typedef struct {
	int p; // suffix length; at least 8
	kc_c4_t **h; // 1<<p hash tables
} kc_c4x_t;

static kc_c4x_t *c4x_init(int p)
{
	int i;
	kc_c4x_t *h;
	CALLOC(h, 1);
	MALLOC(h->h, 1<<p);
	h->p = p;
	for (i = 0; i < 1<<p; ++i)
		h->h[i] = kc_c4_init();
	return h;
}

typedef struct {
	int n, m;
	uint64_t *a;
} buf_c4_t;

static inline void c4x_insert_buf(buf_c4_t *buf, int p, uint64_t y) // insert a k-mer $y to a linear buffer
{
	int pre = y & ((1<<p) - 1);
	buf_c4_t *b = &buf[pre];
	if (b->n == b->m) {
		b->m = b->m < 8? 8 : b->m + (b->m>>1);
		REALLOC(b->a, b->m);
	}
	b->a[b->n++] = y;
}

static void count_seq_buf(buf_c4_t *buf, int k, int p, int len, const char *seq) // insert k-mers in $seq to linear buffer $buf
{
	int i, l;
	uint64_t x[2], mask = (1ULL<<k*2) - 1, shift = (k - 1) * 2;
	for (i = l = 0, x[0] = x[1] = 0; i < len; ++i) {
		int c = seq_nt4_table[(uint8_t)seq[i]];
		if (c < 4) { // not an "N" base
			x[0] = (x[0] << 2 | c) & mask;                  // forward strand
			x[1] = x[1] >> 2 | (uint64_t)(3 - c) << shift;  // reverse strand
			if (++l >= k) { // we find a k-mer
				uint64_t y = x[0] < x[1]? x[0] : x[1];
				// Store the hashed k-mer value
				c4x_insert_buf(buf, p, hash64(y, mask));
			}
		} else l = 0, x[0] = x[1] = 0; // if there is an "N", restart
	}
}

typedef struct { // global data structure for kt_pipeline()
	int k, block_len, n_thread;
	kseq_t *ks;
	kc_c4x_t *h;
} pldat_t;

typedef struct { // data structure for each step in kt_pipeline()
	pldat_t *p;
	int n, m, sum_len, nk;
	int *len;
	char **seq;
	buf_c4_t *buf;
} stepdat_t;

static void worker_for(void *data, long i, int tid) // callback for kt_for()
{
	stepdat_t *s = (stepdat_t*)data;
	buf_c4_t *b = &s->buf[i];
	kc_c4_t *h = s->p->h->h[i];
	int j;
	// Unused parameter
	(void)tid;
	
	for (j = 0; j < b->n; ++j) {
		khint_t k;
		int absent;
        // Shift the hashed k-mer and insert into the hash table
		k = kc_c4_put(h, b->a[j] << KC_BITS, &absent);
		if ((kh_key(h, k)&KC_MAX) < KC_MAX) ++kh_key(h, k);
	}
}

static void *worker_pipeline(void *data, int step, void *in) // callback for kt_pipeline()
{
	pldat_t *p = (pldat_t*)data;
	if (step == 0) { // step 1: read a block of sequences
		int ret;
		stepdat_t *s;
		CALLOC(s, 1);
		s->p = p;
		while ((ret = kseq_read(p->ks)) >= 0) {
			int l = p->ks->seq.l;
			if (l < p->k) continue;
			if (s->n == s->m) {
				s->m = s->m < 16? 16 : s->m + (s->n>>1);
				REALLOC(s->len, s->m);
				REALLOC(s->seq, s->m);
			}
			MALLOC(s->seq[s->n], l);
			memcpy(s->seq[s->n], p->ks->seq.s, l);
			s->len[s->n++] = l;
			s->sum_len += l;
			s->nk += l - p->k + 1;
			if (s->sum_len >= p->block_len)
				break;
		}
		if (s->sum_len == 0) free(s);
		else return s;
	} else if (step == 1) { // step 2: extract k-mers
		stepdat_t *s = (stepdat_t*)in;
		int i, n = 1<<p->h->p, m;
		CALLOC(s->buf, n);
		m = (int)(s->nk * 1.2 / n) + 1;
		for (i = 0; i < n; ++i) {
			s->buf[i].m = m;
			MALLOC(s->buf[i].a, m);
		}
		for (i = 0; i < s->n; ++i) {
			count_seq_buf(s->buf, p->k, p->h->p, s->len[i], s->seq[i]);
			free(s->seq[i]);
		}
		free(s->seq); free(s->len);
		return s;
	} else if (step == 2) { // step 3: insert k-mers to hash table
		stepdat_t *s = (stepdat_t*)in;
		int i, n = 1<<p->h->p;
		kt_for(p->n_thread, worker_for, s, n);
		for (i = 0; i < n; ++i) free(s->buf[i].a);
		free(s->buf); free(s);
	}
	return 0;
}

static kc_c4x_t *count_file(const char *fn, int k, int p, int block_size, int n_thread)
{
	pldat_t pl;
	gzFile fp;
	if ((fp = gzopen(fn, "r")) == 0) return 0;
	pl.ks = kseq_init(fp);
	pl.k = k;
	pl.n_thread = n_thread;
	pl.h = c4x_init(p);
	pl.block_len = block_size;
	kt_pipeline(3, worker_pipeline, &pl, 3);
	kseq_destroy(pl.ks);
	gzclose(fp);
	return pl.h;
}

typedef struct {
	uint64_t *c; // Changed from c[256]
} buf_cnt_t;

typedef struct {
	const kc_c4x_t *h;
	buf_cnt_t *cnt;
} hist_aux_t;

static void worker_hist(void *data, long i, int tid) // callback for kt_for()
{
	hist_aux_t *a = (hist_aux_t*)data;
	uint64_t *cnt = a->cnt[tid].c;
	kc_c4_t *g = a->h->h[i];
	khint_t k;
	for (k = 0; k < kh_end(g); ++k)
		if (kh_exist(g, k)) {
			int c = kh_key(g, k) & KC_MAX;
			++cnt[c];
		}
}

/**
 * Find the first coverage value where the k-mer count increases.
 * This can indicate a potential threshold for meaningful coverage.
 * 
 * @param counts Array of count values indexed by coverage
 * @param hist_size Size of the counts array
 * @return The first coverage value where count increases, or 0 if not found
 */
static int find_first_increasing_coverage(uint64_t *counts, int hist_size)
{
    // Skip very low coverage values that might be noisy
    int start_index = 1;
    
    // Find the first valid point to start from
    while (start_index < hist_size && counts[start_index] == 0) {
        start_index++;
    }
    
    // We want to find the first time the count increases after it has been decreasing
    uint64_t prev_count = counts[start_index];
    int decreasing_streak = 0;
    
    for (int i = start_index + 1; i < hist_size; i++) {
        // Skip zero counts
        if (counts[i] == 0) continue;
        
        // Check if we're still decreasing
        if (counts[i] < prev_count) {
            decreasing_streak++;
            prev_count = counts[i];
            continue;
        }
        
        // We found an increase after at least one decrease
        if (decreasing_streak > 0 && counts[i] > prev_count) {
            return i; // Return the coverage value where the increase occurs
        }
        
        prev_count = counts[i];
    }
    
    return 0; // No increasing point found
}

static void print_hist(const char *output_filename, uint64_t *pre_computed_cnt, int hist_size, int coverage_threshold, int auto_detected)
{
	FILE *hist_file = NULL;

	// Open output file if filename is provided
	if (output_filename) {
		char *hist_filename = NULL;
		hist_filename = malloc(strlen(output_filename) + 6); // +6 for ".hist\0"
		if (!hist_filename) {
			perror("Failed to allocate memory for histogram filename");
			return;
		}
		sprintf(hist_filename, "%s.hist", output_filename);
		hist_file = fopen(hist_filename, "w");
		if (!hist_file) {
			perror("Failed to open histogram output file");
			free(hist_filename);
			return;
		}
		printf("Writing histogram data to %s\n", hist_filename);
		free(hist_filename);
	}

	// Write header comments if output file is used
	if (hist_file) {
		fprintf(hist_file, "# K-mer coverage histogram\n");
		if (auto_detected) {
			fprintf(hist_file, "# Auto-detected coverage threshold: %d\n", coverage_threshold);
		} else {
			fprintf(hist_file, "# User-specified coverage threshold: %d\n", coverage_threshold);
		}
		fprintf(hist_file, "# Format: <coverage>\t<number_of_kmers>\n");
	}

	// Write counts to file or stdout
	for (int i = 1; i < hist_size; ++i) {
		if (pre_computed_cnt[i] > 0) { // Only print non-zero counts
			if (hist_file) {
				fprintf(hist_file, "%d\t%ld\n", i, (long)pre_computed_cnt[i]);
			} else {
				printf("%d\t%ld\n", i, (long)pre_computed_cnt[i]);
			}
		}
	}
	
	// Close output file if opened
	if (hist_file) {
		fclose(hist_file);
	}
}

// Function to convert k-mer integer to sequence
static char *uint64_t_to_seq(uint64_t y, int k)
{
	char *seq = (char*)malloc(k + 1);
	int i;
	for (i = k - 1; i >= 0; --i) {
		seq[i] = "ACGT"[y & 0x3];
		y >>= 2;
	}
	seq[k] = '\0';
	return seq;
}

typedef struct {
	uint64_t y;     // k-mer integer
	uint64_t hash;  // hash value of the k-mer
	uint16_t count; // coverage
} kmer_t;

// Max-heap functions
static void heapify_down(kmer_t *heap, int heap_size, int i)
{
	int largest = i;
	int left = 2 * i + 1;
	int right = 2 * i + 2;
	if (left < heap_size && heap[left].hash > heap[largest].hash)
		largest = left;
	if (right < heap_size && heap[left].hash > heap[largest].hash)
		largest = right;
	if (largest != i) {
		kmer_t temp = heap[i];
		heap[i] = heap[largest];
		heap[largest] = temp;
		heapify_down(heap, heap_size, largest);
	}
}

static void heapify_up(kmer_t *heap, int i)
{
	while (i > 0 && heap[i].hash > heap[(i - 1) / 2].hash) {
		kmer_t temp = heap[i];
		heap[i] = heap[(i - 1) / 2];
		heap[(i - 1) / 2] = temp;
		i = (i - 1) / 2;
	}
}

// Comparator for qsort (ascending order)
int compare_kmers(const void *a, const void *b) {
    kmer_t *k1 = (kmer_t *)a;
    kmer_t *k2 = (kmer_t *)b;
    if (k1->hash < k2->hash) return -1;
    if (k1->hash > k2->hash) return 1;
    return 0;
}

// Function to select top N k-mers with minimal hash values and coverage >= c
static void select_top_kmers(const kc_c4x_t *h, int N, int coverage_threshold, int k, FILE *output_fp)
{
	int i;
	// Max-heap to store k-mers
	kmer_t *heap = (kmer_t*)malloc(N * sizeof(kmer_t));
	int heap_size = 0;
	uint64_t mask = (1ULL << (k * 2)) - 1; // Mask to confine to k-mer bit space

	for (i = 0; i < 1 << h->p; ++i) {
		kc_c4_t *g = h->h[i];
		khint_t k_iter;
		for (k_iter = 0; k_iter < kh_end(g); ++k_iter) {
			if (kh_exist(g, k_iter)) {
				uint64_t key = kh_key(g, k_iter);
				uint16_t count = key & KC_MAX;
				if (count >= coverage_threshold) {
					// Extract the higher bits of the hashed k-mer from the key
					uint64_t hashed_kmer_high = key >> KC_BITS;
					// Reconstruct the full hashed k-mer
					uint64_t hashed_kmer = (hashed_kmer_high << h->p) | i;
					// Recover the original k-mer using hash64i
					uint64_t y = hash64i(hashed_kmer, mask);
					// Compute the hash of the original k-mer for comparison
					uint64_t hashed_y = selected_hash_func(y); // Already hashed

					if (heap_size < N) {
						heap[heap_size].y = y;
						heap[heap_size].hash = hashed_y;
						heap[heap_size].count = count;
						heapify_up(heap, heap_size);
						heap_size++;
					} else if (hashed_y < heap[0].hash) {
						heap[0].y = y;
						heap[0].hash = hashed_y;
						heap[0].count = count;
						heapify_down(heap, heap_size, 0);
					}
				}
			}
		}
	}

	// Sort the heap array based on the hash values
	qsort(heap, heap_size, sizeof(kmer_t), compare_kmers);

	// Output k-mers from the sorted heap
	fprintf(output_fp, "#Top %d k-mers with minimal hash values and coverage >= %d:\n", N, coverage_threshold);
	for (i = 0; i < heap_size; ++i) {
		char *seq = uint64_t_to_seq(heap[i].y, k);
		fprintf(output_fp, "%s\t%llu\t%d\n", seq, (unsigned long long)heap[i].hash, heap[i].count);
		free(seq);
	}
    
	// Ensure that the output is flushed to the file
	fflush(output_fp);

	free(heap);
}

// Global variable to store output filename
char *output_filename = NULL;

int main(int argc, char *argv[])
{
	kc_c4x_t *h;
	int i, c, k = 31, p = KC_BITS, block_size = 10000000, n_thread = 4;
	int N = 10000, coverage_threshold = 0; // Change default to 0 to indicate auto-detection
	int use_wang_hash = 0;
	int auto_coverage_threshold = 1; // Flag to indicate auto-detection should be used
	ketopt_t o = KETOPT_INIT;
	while ((c = ketopt(&o, argc, argv, 1, "k:p:b:t:N:c:wo:", 0)) >= 0) {
		if (c == 'k') k = atoi(o.arg);
		else if (c == 'p') p = atoi(o.arg);
		else if (c == 'b') block_size = atoi(o.arg);
		else if (c == 't') n_thread = atoi(o.arg);
		else if (c == 'N') N = atoi(o.arg);
		else if (c == 'c') {
			coverage_threshold = atoi(o.arg);
			auto_coverage_threshold = 0; // User specified a threshold, don't auto-detect
		}
		else if (c == 'w') use_wang_hash = 1;
		else if (c == 'o') output_filename = strdup(o.arg);
	}
	if (argc - o.ind < 1) {
		fprintf(stderr, "Usage: fgr2 [options] <in.fa|in.fq|in.fa.gz|in.fq.gz>\n");
		fprintf(stderr, "Options:\n");
		fprintf(stderr, "  -k INT     k-mer size [%d]\n", k);
		fprintf(stderr, "  -p INT     prefix length [%d]\n", p);
		fprintf(stderr, "  -b INT     block size [%d]\n", block_size);
		fprintf(stderr, "  -t INT     number of worker threads [%d]\n", n_thread);
		fprintf(stderr, "  -N INT     number of k-mers to output [%d]\n", N);
		fprintf(stderr, "  -c INT     minimum coverage threshold [default: auto-detect]\n");
		fprintf(stderr, "             (If not specified, the threshold is automatically detected from the histogram)\n");
		fprintf(stderr, "  -w         use Thomas Wang's hash function (default: MurmurHash3)\n");
		fprintf(stderr, "  -o FILE    Output file to write the top N k-mers with minimal hash values and coverage >= c\n");
		fprintf(stderr, "             Also creates FILE.hist with k-mer histogram data\n\n");
		fprintf(stderr, "Supported input formats:\n");
		fprintf(stderr, "  - FASTA format (*.fa, *.fa.gz)\n");
		fprintf(stderr, "  - FASTQ format (*.fq, *.fq.gz)\n\n");
		fprintf(stderr, "Author: Lifu Song <songlf@tib.cas.cn>\n");
		return 1;
	}
	if (p < KC_BITS) {
		fprintf(stderr, "ERROR: -p should be at least %d\n", KC_BITS);
		return 1;
	}

	if (use_wang_hash) {
		selected_hash_func = hash_wang;
		printf("Using Thomas Wang's hash function.\n");
	} else {
		selected_hash_func = hash_murmur3;
		printf("Using MurmurHash3 hash function.\n");
	}

	h = count_file(argv[o.ind], k, p, block_size, n_thread);
	
	// Set up structures for histogram to detect coverage threshold if needed
	hist_aux_t a;
	uint64_t *cnt = NULL;
    int j;
    const int hist_size = 1 << KC_BITS;

    // Allocate histogram memory
    CALLOC(cnt, hist_size);
    if (!cnt) { 
        perror("Failed to allocate memory for histogram"); 
        return 1;
    }

	a.h = h;
	CALLOC(a.cnt, n_thread);
    if (!a.cnt) {
        perror("Failed to allocate memory for thread histograms");
        free(cnt);
        return 1;
    }

    for (j = 0; j < n_thread; ++j) {
        CALLOC(a.cnt[j].c, hist_size);
        if (!a.cnt[j].c) {
            perror("Failed to allocate memory for a thread's histogram buffer");
            for (int k_idx = 0; k_idx < j; ++k_idx) free(a.cnt[k_idx].c);
            free(a.cnt);
            free(cnt);
            return 1;
        }
    }

	// Calculate histogram
	kt_for(n_thread, worker_hist, &a, 1<<h->p);

	for (j = 0; j < n_thread; ++j) {
		for (i = 0; i < hist_size; ++i)
            cnt[i] += a.cnt[j].c[i];
        free(a.cnt[j].c);
    }
	free(a.cnt);
	
	// Auto-detect coverage threshold if needed
	if (auto_coverage_threshold) {
		int detected_threshold = find_first_increasing_coverage(cnt, hist_size);
		if (detected_threshold > 0) {
			coverage_threshold = detected_threshold;
			printf("Auto-detected coverage threshold: %d\n", coverage_threshold);
		} else {
			coverage_threshold = 1; // Default if auto-detection fails
			printf("No coverage threshold detected automatically. Using default: %d\n", coverage_threshold);
		}
	} else {
		printf("Using user-specified coverage threshold: %d\n", coverage_threshold);
	}
	
	// Write histogram data to file or stdout
	print_hist(output_filename, cnt, hist_size, coverage_threshold, auto_coverage_threshold);
	
	// Free histogram memory
	free(cnt);

	FILE *output_fp = stdout; // Default to standard output

	// Open output file if -o is specified
	if (output_filename) {
		output_fp = fopen(output_filename, "w");
		if (!output_fp) {
			fprintf(stderr, "Error: Cannot open output file '%s'\n", output_filename);
			return 1;
		}
	}

    // Select and output N k-mers with minimal hash values and coverage >= c
	if (N > 0)
		select_top_kmers(h, N, coverage_threshold, k, output_fp);

	// Close output file if it was opened
	if (output_fp != stdout) {
		fclose(output_fp);
	}

	// Free allocated memory for output filename
	if (output_filename)
		free(output_filename);

	for (i = 0; i < 1<<p; ++i) kc_c4_destroy(h->h[i]);
	free(h->h); free(h);
	return 0;
}