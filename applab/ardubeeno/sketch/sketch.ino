// ArduBeeno: the LED matrix shows the state from python/main.py.
// The number of mites counted so far (a tick while it is 0). While a mite is detected,
// a mite icon alternates with the number.

#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>

const int COLS = 13;

Arduino_LED_Matrix matrix;
volatile bool mite = false;
volatile int count = 0;

// 8 rows x 13 columns, 3-bit brightness (0-7).
uint8_t mite_frame[104] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,7,0,0,0,0,0,0,0,7,0,0,
    0,0,0,7,7,7,7,7,7,7,0,0,0,
    0,7,7,7,7,7,7,7,7,7,7,7,0,
    0,0,7,7,7,7,7,7,7,7,7,0,0,
    0,7,0,7,7,7,7,7,7,7,0,7,0,
    0,0,7,0,0,0,0,0,0,0,7,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0
};

uint8_t clean_frame[104] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,3,0,0,
    0,0,0,0,0,0,0,0,0,3,0,0,0,
    0,0,0,0,0,0,0,0,3,0,0,0,0,
    0,0,3,0,0,0,0,3,0,0,0,0,0,
    0,0,0,3,0,0,3,0,0,0,0,0,0,
    0,0,0,0,3,3,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0
};

// 3x5 digits, one row per byte, bit 2 = left column.
const uint8_t DIGITS[10][5] = {
    {7,5,5,5,7}, {2,6,2,2,7}, {7,1,7,4,7}, {7,1,7,1,7}, {5,5,7,1,1},
    {7,4,7,1,7}, {7,4,7,5,7}, {7,1,2,2,2}, {7,5,7,5,7}, {7,5,7,1,7}
};

uint8_t number_frame[104];

// Draws n (capped at 999) centred, 3 columns per digit with 1 blank column between digits.
void render_number(int n) {
    if (n > 999) n = 999;
    char text[4];
    snprintf(text, sizeof(text), "%d", n);
    int len = strlen(text);
    int x0 = (COLS - (len * 4 - 1)) / 2;
    memset(number_frame, 0, sizeof(number_frame));
    for (int i = 0; i < len; i++) {
        const uint8_t *d = DIGITS[text[i] - '0'];
        for (int r = 0; r < 5; r++) {
            for (int c = 0; c < 3; c++) {
                if (d[r] & (4 >> c)) number_frame[(r + 1) * COLS + x0 + i * 4 + c] = 7;
            }
        }
    }
}

void set_mite(bool state) {
    mite = state;
}

void set_count(int n) {
    count = n < 0 ? 0 : n;
}

void setup() {
    matrix.begin();
    matrix.setGrayscaleBits(3);
    matrix.draw(clean_frame);
    Bridge.begin();
    Bridge.provide("set_mite", set_mite);
    Bridge.provide("set_count", set_count);
}

void loop() {
    int n = count;
    uint8_t *idle = clean_frame;
    if (n > 0) {
        render_number(n);
        idle = number_frame;
    }
    if (mite) {
        matrix.draw((millis() / 500) % 2 ? mite_frame : idle);
    } else {
        matrix.draw(idle);
    }
    delay(50);
}
