#include <vector>
#include <stdarg.h>
#include "string_formater.h"

// I found this online, I not really sure what it does #slackoverflowcopier #copy&paste ✌️
std::string string_formater::formater(const std::string &format, ...) {
    va_list args;
    va_start (args, format);
    size_t len = std::vsnprintf(NULL, 0, format.c_str(), args);
    va_end (args);
    std::vector<char> vec(len + 1);
    va_start (args, format);
    std::vsnprintf(&vec[0], len + 1, format.c_str(), args);
    va_end (args);
    return &vec[0];
}
