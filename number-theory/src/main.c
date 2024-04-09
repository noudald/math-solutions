#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>


#define INIT_DIGIT_SIZE 64

// TODO:
// * Implement operators: addition, multiplication, modulo.
// * Documentation.
// * Testing.


typedef struct BigInt_t {
    uint32_t * digits;
    uint32_t ndigits;
    uint32_t sign;
    // 0: BigInt is valid.
    // 1: BigInt is not valid.
    uint32_t valid;
} BigInt;


typedef struct BigIntResult_t {
    // 0: Valid result.
    // 1: Cannot run method because BigInt is not valid.
    // 2: String is too short to convert to BigInt.
    // 3: Cannot (re)allocate digits.
    uint32_t result;
    char * msg;
} BigIntResult;


BigIntResult bi_init(BigInt * bi) {
    bi->digits = malloc(INIT_DIGIT_SIZE * sizeof(uint32_t));
    for (uint32_t i = 0; i < INIT_DIGIT_SIZE; i++) {
        bi->digits[i] = 0;
    }

    bi->ndigits = INIT_DIGIT_SIZE;
    bi->sign = 0;
    bi->valid = 0;

    return (BigIntResult){
        .result = 0,
        .msg = "BigInt successfully initialized.",
    };
}


uint32_t bi_is_valid(BigInt * bi) {
    if (bi->valid) {
        return 0;
    } else {
        return 1;
    }
}


BigIntResult bi_alloc_digits(BigInt * bi, uint32_t num_of_digits) {
    if (!bi_is_valid(bi)) {
        return (BigIntResult) {
            .result = 1,
            .msg = "Cannot convert allocate digits to BigInt because BigInt is not valid.",
        };
    }

    bi->digits = realloc(bi->digits, num_of_digits * sizeof(uint32_t));
    if (bi->digits == NULL) {
        bi->valid = 1;
        return (BigIntResult) {
            .result = 3,
            .msg = "Could not allocate enough memory to assign BigInt.",
        };
    }

    return (BigIntResult) {
        .result = 0,
        .msg = "Successfully allocated digits for BigInt.",
    };
}


BigIntResult bi_from_str(BigInt *bi, char *str) {
    uint32_t num_of_digits = strlen(str);

    if (num_of_digits == 0) {
        return (BigIntResult) {
            .result = 2,
            .msg = "Empty string cannot be converted to BigInt.",
        };
    }

    uint32_t offset = 0;
    if (str[0] == '-') {
        bi->sign = 1;
        offset = 1;
        num_of_digits -= 1;
    } else {
        bi->sign = 0;
    }

    BigIntResult bi_result = bi_alloc_digits(bi, num_of_digits);
    if (bi_result.result) {
        return bi_result;
    }

    for (uint32_t i = 0; i < num_of_digits; i++) {
        bi->digits[i] = (uint32_t) (str[num_of_digits + offset - i - 1] - '0');
    }
    bi->ndigits = num_of_digits;

    return (BigIntResult){
        .result = 0,
        .msg = "Successfully converted string to BigInt.",
    };
}


uint32_t num_of_digits(uint32_t n) {
    uint32_t comp = 10, i = 1;
    while (comp < n) {
        i++;
        comp *= 10;
    }
    return i;
}


BigIntResult bi_from_int32(BigInt *bi, int32_t n) {
    if (!bi_is_valid(bi)) {
        return (BigIntResult) {
            .result = 1,
            .msg = "Cannot convert int32 to BigInt because BigInt is not valid.",
        };
    }

    if (n > 0) {
        bi->sign = 0;
    } else {
        bi->sign = 1;
        n = -n;
    }
    bi->ndigits = num_of_digits((uint32_t)n);

    bi->digits = realloc(bi->digits, bi->ndigits * sizeof(uint32_t));
    if (bi->digits == NULL) {
        bi->valid = 1;
        return (BigIntResult) {
            .result = 3,
            .msg = "Could not allocate enough memory to assign BigInt.",
        };
    }

    for (uint32_t i = 0; i < num_of_digits; i++) {
        bi->digits[i] = (uint32_t) (str[num_of_digits + offset - i - 1] - '0');
    }
    bi->ndigits = num_of_digits;

    return (BigIntResult){
        .result = 0,
        .msg = "Successfully converted string to BigInt.",
    };
}


BigIntResult bi_print(BigInt * bi) {
    if (bi->valid) {
        return (BigIntResult) {
            .result = 1,
            .msg = "BigInt is not valid.",
        };
    }

    if (bi->sign) {
        printf("-");
    }

    for (uint32_t i = 0; i < bi->ndigits; i++) {
        printf("%d", bi->digits[bi->ndigits - i - 1]);
    }

    printf("\n");

    return (BigIntResult) {
        .result = 0,
        .msg = "Successfully printed BigInt.",
    };
}


BigIntResult bi_free(BigInt *bi) {
    if (bi->valid) {
        return (BigIntResult) {
            .result = 1,
            .msg = "BigInt is not valid.",
        };
    }

    free(bi->digits);
    bi->valid = 1;

    return (BigIntResult) {
        .result = 0,
        .msg = "Successfully freed BigInt."
    };
}


int main(void) {
    BigInt bi1;

    bi_init(&bi1);
    bi_from_str(&bi1, "-1234567890");
    bi_print(&bi1);

    BigInt bi2;

    bi_init(&bi2);
    bi_from_str(&bi2, "1234567890");
    bi_print(&bi2);

    bi_free(&bi1);
    bi_free(&bi2);

    return 0;
}
