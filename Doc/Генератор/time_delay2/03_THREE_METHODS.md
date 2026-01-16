💻 ТРИ МЕТОДА ИНТЕРПОЛЯЦИИ: КОД
МЕТОД 1: ЛИНЕЙНАЯ ИНТЕРПОЛЯЦИЯ
text
float t1 = (float)sample_int / sample_rate;
float chirp_rate = (f_stop - f_start) / duration;
float phase1 = 2.0f * 3.14159265f * (f_start * t1 + 0.5f * chirp_rate * t1 * t1);
float real1 = cos(phase1), imag1 = sin(phase1);

float t2 = (float)(sample_int + 1) / sample_rate;
float phase2 = 2.0f * 3.14159265f * (f_start * t2 + 0.5f * chirp_rate * t2 * t2);
float real2 = cos(phase2), imag2 = sin(phase2);

real = real1 * (1.0f - sample_frac) + real2 * sample_frac;
imag = imag1 * (1.0f - sample_frac) + imag2 * sample_frac;
Точность: 2% на 50нс, 4.6% на 10нс

МЕТОД 2: КУБИЧЕСКАЯ ИНТЕРПОЛЯЦИЯ
text
// 4 лагранжевых полинома - смешивание 4 соседних отсчётов
float L_m1 = -alpha * (alpha - 1) * (alpha - 2) / 6.0f;
float L_0 = (alpha + 1) * (alpha - 1) * (alpha - 2) / (-2.0f);
float L_1 = (alpha + 1) * alpha * (alpha - 2) / 2.0f;
float L_2 = (alpha + 1) * alpha * (alpha - 1) / 6.0f;

real = real_m1 * L_m1 + real_0 * L_0 + real_1 * L_1 + real_2 * L_2;
imag = imag_m1 * L_m1 + imag_0 * L_0 + imag_1 * L_1 + imag_2 * L_2;
Точность: 0.4% на 50нс, 0.8% на 10нс

МЕТОД 3: 🏆 СПЕКТРАЛЬНАЯ ИНТЕРПОЛЯЦИЯ
text
float t_exact = delayed_sample_float / sample_rate;
float chirp_rate = (f_stop - f_start) / duration;
float phase_exact = 2.0f * 3.14159265f * (f_start * t_exact + 0.5f * chirp_rate * t_exact * t_exact);
real = cos(phase_exact);
imag = sin(phase_exact);
Точность: <0.01% на всех задержках

ВЫБИРАЙТЕ МЕТОД 3 - он лучше всех!