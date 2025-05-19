#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdint.h>
#include <stdbool.h>
#include <ctype.h>
#include <zlib.h>
#include <time.h>
#include <unistd.h>
#include "kseq.h" // Include kseq.h

// For getopt
extern char *optarg;
extern int optind;

#define MAX_KMER_SIZE 31   // Maximum k-mer length
#define HASH_TABLE_SIZE 10000019  // A large prime number for hash table size
#define BUFFER_SIZE 4096    // Buffer size for file reading

// Structure to hold k-mer counts
typedef struct {
    uint64_t kmer;
    uint32_t count;
} KmerCount;

// Node for handling collisions in hash table
typedef struct KmerNode {
    uint64_t kmer;
    uint32_t count;
    struct KmerNode *next;
} KmerNode;

// Declare kseq_t types for gzFile
KSEQ_INIT(gzFile, gzread)

// Thread data structure
typedef struct {
    char *sequence;
    size_t start;
    size_t end;
    int k;
    KmerNode **hash_table;
    pthread_mutex_t *mutexes;
} ThreadData;

// MurmurHash3 64-bit hash function
uint64_t murmur3_64(uint64_t key) {
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return key;
}

// Thomas Wang's 64-bit hash function
uint64_t wang_hash64(uint64_t key) {
    key = (~key) + (key << 21); // key = (key << 21) - key - 1;
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8); // key * 265
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4); // key * 21
    key = key ^ (key >> 28);
    key = key + (key << 31);
    return key;
}

// Function pointer for the chosen hash function
uint64_t (*hash_function)(uint64_t);

// Convert DNA sequence to 2-bit representation
uint64_t dna_to_int(const char *seq, int len) {
    uint64_t result = 0;
    for (int i = 0; i < len; i++) {
        result <<= 2;
        switch (seq[i]) {
            case 'A': case 'a': result |= 0; break;
            case 'C': case 'c': result |= 1; break;
            case 'G': case 'g': result |= 2; break;
            case 'T': case 't': result |= 3; break;
            default: return UINT64_MAX; // Non-ACGT character
        }
    }
    return result;
}

// Initialize hash table
KmerNode **init_hash_table() {
    KmerNode **table = calloc(HASH_TABLE_SIZE, sizeof(KmerNode *));
    if (!table) {
        perror("Hash table allocation failed");
        exit(EXIT_FAILURE);
    }
    return table;
}

// Insert or increment k-mer count
void insert_kmer(KmerNode **table, pthread_mutex_t *mutexes, uint64_t kmer) {
    uint64_t hash = hash_function(kmer) % HASH_TABLE_SIZE;
    pthread_mutex_lock(&mutexes[hash % 256]); // Use a subset of mutexes
    KmerNode *node = table[hash];
    while (node) {
        if (node->kmer == kmer) {
            node->count++;
            pthread_mutex_unlock(&mutexes[hash % 256]);
            return;
        }
        node = node->next;
    }
    // K-mer not found, add new node
    KmerNode *new_node = malloc(sizeof(KmerNode));
    if (!new_node) {
        perror("KmerNode allocation failed");
        pthread_mutex_unlock(&mutexes[hash % 256]);
        exit(EXIT_FAILURE);
    }
    new_node->kmer = kmer;
    new_node->count = 1;
    new_node->next = table[hash];
    table[hash] = new_node;
    pthread_mutex_unlock(&mutexes[hash % 256]);
}

// Function to read sequences using kseq.h
char *read_sequences_kseq(const char *filename, size_t *total_length) {
    gzFile fp = gzopen(filename, "r");
    if (fp == NULL) {
        perror("Error opening file");
        return NULL;
    }

    kseq_t *seq = kseq_init(fp);
    if (seq == NULL) {
        fprintf(stderr, "Error initializing kseq.\n");
        gzclose(fp);
        return NULL;
    }

    char *all_sequences = NULL;
    size_t current_capacity = 0;
    *total_length = 0;
    int l;

    while ((l = kseq_read(seq)) >= 0) {
        if (*total_length + seq->seq.l + 1 > current_capacity) {
            current_capacity = (*total_length + seq->seq.l + 1) * 1.5; // Allocate 50% more space
            if (current_capacity < BUFFER_SIZE) current_capacity = BUFFER_SIZE;
            char *new_sequences = realloc(all_sequences, current_capacity);
            if (new_sequences == NULL) {
                perror("Memory reallocation error");
                free(all_sequences);
                kseq_destroy(seq);
                gzclose(fp);
                return NULL;
            }
            all_sequences = new_sequences;
        }
        memcpy(all_sequences + *total_length, seq->seq.s, seq->seq.l);
        *total_length += seq->seq.l;
    }

    if (l < -1) { // -1 is normal EOF, -2 truncated quality, -3 error reading sequence
        fprintf(stderr, "Error reading sequence: kseq_read returned %d\n", l);
        free(all_sequences);
        kseq_destroy(seq);
        gzclose(fp);
        return NULL;
    }
    
    if (all_sequences != NULL) {
         all_sequences[*total_length] = '\0'; // Null-terminate the concatenated string
    } else if (*total_length == 0) { // Handle empty input file
        all_sequences = malloc(1);
        if (all_sequences == NULL) {
            perror("Memory allocation error for empty sequence");
            kseq_destroy(seq);
            gzclose(fp);
            return NULL;
        }
        all_sequences[0] = '\0';
    }


    kseq_destroy(seq);
    gzclose(fp);
    return all_sequences;
}

void *count_kmers(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    const char *seq = data->sequence;
    int k = data->k;
    size_t start = data->start;
    size_t end = data->end;
    KmerNode **hash_table = data->hash_table;
    pthread_mutex_t *mutexes = data->mutexes;

    size_t processed = 0;
    for (size_t i = start; i <= end - k; i++) {
        uint64_t kmer = dna_to_int(&seq[i], k);
        if (kmer == UINT64_MAX) continue; // Skip invalid k-mers
        insert_kmer(hash_table, mutexes, kmer);
        processed++;
        // Optional: progress reporting
        if (processed % 1000000 == 0) {
            printf("Thread %p processed %zu k-mers\n", (void*)pthread_self(), processed);
        }
    }
    return NULL;
}

// Comparator function for qsort (sort by hash value in ascending order)
int compare_kmer_hashes(const void *a, const void *b) {
    const KmerCount *kmer_a = (const KmerCount *)a;
    const KmerCount *kmer_b = (const KmerCount *)b;
    uint64_t hash_a = hash_function(kmer_a->kmer);
    uint64_t hash_b = hash_function(kmer_b->kmer);
    if (hash_a < hash_b) return -1;
    else if (hash_a > hash_b) return 1;
    else return 0;
}

void print_usage(const char *program_name) {
    fprintf(stderr, "Usage: %s [OPTIONS] <sequence_file>\n", program_name);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -k <int>        K-mer size (default: 31)\n");
    fprintf(stderr, "  -t <int>        Number of threads (default: 1)\n");
    fprintf(stderr, "  -o <file>       Output file for all k-mer counts\n");
    fprintf(stderr, "  -F <file>       Output file for top N k-mers with smallest hash values\n");
    fprintf(stderr, "  -N <int>        Number of top k-mers with smallest hash values to output (default: 10000)\n");
    fprintf(stderr, "  -c <int>        Coverage cutoff for k-mers (default: 0)\n");
    fprintf(stderr, "  -w              Use Wang's hash function (default: MurmurHash3)\n");
    fprintf(stderr, "  -h              Display this help message and exit\n");
}

int main(int argc, char *argv[]) {
    int k = 31; // Default k-mer size is now 31
    int num_threads = 1;
    const char *sequence_file = NULL;
    const char *output_file = NULL;
    const char *F_output_file = NULL; // Output file for top N k-mers
    int N = 10000; // Default number of k-mers to output
    int coverage_cutoff = 0; // Coverage cutoff
    bool use_wang_hash = false; // Flag to use Wang's hash function
    int opt;

    // Parse command-line options
    while ((opt = getopt(argc, argv, "k:t:o:F:N:c:wh")) != -1) {
        switch (opt) {
            case 'k':
                k = atoi(optarg);
                break;
            case 't':
                num_threads = atoi(optarg);
                break;
            case 'o':
                output_file = optarg;
                break;
            case 'F':
                F_output_file = optarg;
                break;
            case 'N':
                N = atoi(optarg);
                break;
            case 'c':
                coverage_cutoff = atoi(optarg);
                break;
            case 'w':
                use_wang_hash = true;
                break;
            case 'h':
                print_usage(argv[0]);
                return EXIT_SUCCESS;
            default:
                print_usage(argv[0]);
                return EXIT_FAILURE;
        }
    }

    // Check if required arguments are provided
    if (optind >= argc || k <= 0 || k > MAX_KMER_SIZE || num_threads <= 0) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    sequence_file = argv[optind];

    if (k <= 0 || k > MAX_KMER_SIZE) {
        fprintf(stderr, "Invalid k-mer size. Must be between 1 and %d.\n", MAX_KMER_SIZE);
        return EXIT_FAILURE;
    }
    if (num_threads <= 0) {
        fprintf(stderr, "Invalid number of threads.\n");
        return EXIT_FAILURE;
    }

    // Set the hash function based on the user's choice
    hash_function = use_wang_hash ? wang_hash64 : murmur3_64;

    // Generate default output filenames if not provided
    char *default_output_file = NULL;
    char *default_F_output_file = NULL;

    if (output_file == NULL) {
        default_output_file = malloc(strlen(sequence_file) + strlen(".kmers") + 1);
        if (default_output_file == NULL) {
            perror("Memory allocation for default output file name failed");
            return EXIT_FAILURE;
        }
        sprintf(default_output_file, "%s.kmers", sequence_file);
        output_file = default_output_file;
    }

    if (F_output_file == NULL) {
        default_F_output_file = malloc(strlen(sequence_file) + strlen(".fgr") + 1);
        if (default_F_output_file == NULL) {
            perror("Memory allocation for default F output file name failed");
            free(default_output_file); // free previously allocated memory if any
            return EXIT_FAILURE;
        }
        sprintf(default_F_output_file, "%s.fgr", sequence_file);
        F_output_file = default_F_output_file;
    }

    // Read sequences from file
    size_t total_length;
    char *sequence = read_sequences_kseq(sequence_file, &total_length);
    if (!sequence) {
        fprintf(stderr, "Failed to read sequences.\n");
        return EXIT_FAILURE;
    }
    printf("Total sequence length: %zu\n", total_length);

    // Initialize hash table and mutexes
    KmerNode **hash_table = init_hash_table();
    pthread_mutex_t mutexes[256]; // Use 256 mutexes for simplicity
    for (int i = 0; i < 256; i++) {
        pthread_mutex_init(&mutexes[i], NULL);
    }

    // Start timing
    clock_t start = clock();

    // Create threads
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];
    size_t chunk_size = total_length / num_threads;

    for (int i = 0; i < num_threads; i++) {
        thread_data[i].sequence = sequence;
        thread_data[i].k = k;
        thread_data[i].hash_table = hash_table;
        thread_data[i].mutexes = mutexes;
        thread_data[i].start = i * chunk_size;
        thread_data[i].end = (i == num_threads - 1) ? total_length - 1 : (i + 1) * chunk_size + k - 1;

        if (pthread_create(&threads[i], NULL, count_kmers, &thread_data[i]) != 0) {
            perror("Thread creation failed");
            return EXIT_FAILURE;
        }
    }

    // Wait for threads to finish
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    // Stop timing
    clock_t end = clock();
    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("K-mer counting completed in %.2f seconds\n", cpu_time_used);

    // Output results to file if specified
    if (output_file) {
        FILE *out = fopen(output_file, "w");
        if (!out) {
            perror("Error opening output file");
            return EXIT_FAILURE;
        }
        for (size_t i = 0; i < HASH_TABLE_SIZE; i++) {
            KmerNode *node = hash_table[i];
            while (node) {
                // Convert integer k-mer back to DNA sequence
                char kmer_seq[MAX_KMER_SIZE + 1];
                uint64_t temp = node->kmer;
                for (int j = k - 1; j >= 0; j--) {
                    int base = temp & 0x3;
                    kmer_seq[j] = "ACGT"[base];
                    temp >>= 2;
                }
                kmer_seq[k] = '\0';
                fprintf(out, "%s\t%u\n", kmer_seq, node->count);
                node = node->next;
            }
        }
        fclose(out);
        printf("K-mer counts written to %s\n", output_file);
    } else {
        printf("No output file specified. K-mer counts not written to file.\n");
    }

    // If -F option is provided, output top N k-mers with smallest hash values
    if (F_output_file) {
        KmerCount *top_kmers = malloc(N * sizeof(KmerCount));
        if (!top_kmers) {
            perror("Memory allocation failed for top_kmers");
            return EXIT_FAILURE;
        }
        int top_count = 0;

        for (size_t i = 0; i < HASH_TABLE_SIZE; i++) {
            KmerNode *node = hash_table[i];
            while (node) {
                if (node->count > (uint32_t)coverage_cutoff) {
                    uint64_t hash = hash_function(node->kmer);
                    if (top_count < N) {
                        top_kmers[top_count].kmer = node->kmer;
                        top_kmers[top_count].count = node->count;
                        top_count++;
                        if (top_count == N) {
                            qsort(top_kmers, N, sizeof(KmerCount), compare_kmer_hashes);
                        }
                    } else if (hash < hash_function(top_kmers[N-1].kmer)) {
                        top_kmers[N-1].kmer = node->kmer;
                        top_kmers[N-1].count = node->count;
                        qsort(top_kmers, N, sizeof(KmerCount), compare_kmer_hashes);
                    }
                }
                node = node->next;
            }
        }

        // Output the top N k-mers to the specified file
        FILE *F_out = fopen(F_output_file, "w");
        if (!F_out) {
            perror("Error opening output file for top N k-mers with smallest hash values");
            free(top_kmers);
            return EXIT_FAILURE;
        }

        for (int i = 0; i < top_count; i++) {
            // Convert integer k-mer back to DNA sequence
            char kmer_seq[MAX_KMER_SIZE + 1];
            uint64_t temp = top_kmers[i].kmer;
            for (int j = k - 1; j >= 0; j--) {
                int base = temp & 0x3;
                kmer_seq[j] = "ACGT"[base];
                temp >>= 2;
            }
            kmer_seq[k] = '\0';
            uint64_t hash = hash_function(top_kmers[i].kmer);
            fprintf(F_out, "%s\t%llu\t%u\n", kmer_seq, (unsigned long long)hash, top_kmers[i].count);
        }
        fclose(F_out);
        printf("Top %d k-mers with smallest hash values written to %s using %s hash function\n", 
               top_count, F_output_file, use_wang_hash ? "Wang's" : "MurmurHash3");

        free(top_kmers);
    }

    // Clean up
    free(sequence);
    for (size_t i = 0; i < HASH_TABLE_SIZE; i++) {
        KmerNode *node = hash_table[i];
        while (node) {
            KmerNode *tmp = node;
            node = node->next;
            free(tmp);
        }
    }
    free(hash_table);
    for (int i = 0; i < 256; i++) {
        pthread_mutex_destroy(&mutexes[i]);
    }

    // Free default filenames if they were allocated
    if (default_output_file) {
        free(default_output_file);
    }
    if (default_F_output_file) {
        free(default_F_output_file);
    }

    return EXIT_SUCCESS;
}