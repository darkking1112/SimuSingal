#include "demo_api.h"
#include <stddef.h>
#include <string.h>

uint32_t DEMO_CALL demo_abi_version(void)
{
    return 1u;
}

int32_t DEMO_CALL demo_copy_f32(
    const float *input,
    uint64_t count,
    float *output,
    uint64_t capacity,
    uint64_t *written)
{
    if (written == NULL) {
        return DEMO_INVALID_ARGUMENT;
    }
    *written = 0;
    if (count == 0) {
        return DEMO_OK;
    }
    if (input == NULL || output == NULL || count > SIZE_MAX / sizeof(float)) {
        return DEMO_INVALID_ARGUMENT;
    }
    if (capacity < count) {
        return DEMO_BUFFER_TOO_SMALL;
    }
    memmove(output, input, (size_t)count * sizeof(float));
    *written = count;
    return DEMO_OK;
}
