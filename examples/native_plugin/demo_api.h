#ifndef DEMO_API_H
#define DEMO_API_H

#include <stdint.h>

#if defined(_WIN32)
#  if defined(DEMO_BUILD)
#    define DEMO_API __declspec(dllexport)
#  else
#    define DEMO_API __declspec(dllimport)
#  endif
#  define DEMO_CALL __cdecl
#else
#  define DEMO_API __attribute__((visibility("default")))
#  define DEMO_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define DEMO_OK 0
#define DEMO_INVALID_ARGUMENT 1
#define DEMO_BUFFER_TOO_SMALL 2

DEMO_API uint32_t DEMO_CALL demo_abi_version(void);

/* Counts are float elements, not bytes or complex samples.
 * Input and output buffers belong to the caller. No pointer is retained.
 * written is required and is zero on failure. Empty input is valid.
 * On BUFFER_TOO_SMALL the output remains unchanged.
 */
DEMO_API int32_t DEMO_CALL demo_copy_f32(
    const float *input,
    uint64_t count,
    float *output,
    uint64_t capacity,
    uint64_t *written);

#ifdef __cplusplus
}
#endif

#endif
